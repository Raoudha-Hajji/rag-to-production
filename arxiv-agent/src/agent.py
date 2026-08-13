"""
The agent loop: sends the user's question to the LLM along with tool
definitions, lets the model decide whether to call a tool, executes
that tool if requested, feeds the result back, and repeats until the
model gives a final answer.

messages is the transcript of all of this, it is not just the newest entry. 
That is how the llm remembers, by reading the whole history each turn.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from tools import AVAILABLE_TOOLS, _embed_model

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "llama-3.1-8b-instant"

_SCOPE_REFERENCE_TOPICS = [
    "retrieval augmented generation",
    "large language models",
    "NLP and machine learning research",
    "AI agents and tool use",
    "transformers and embeddings",
    "vector databases and semantic search",
]
_scope_embeddings = _embed_model.encode(_SCOPE_REFERENCE_TOPICS)

SCOPE_THRESHOLD = 0.35  


def is_in_scope(question: str) -> bool:
    q_emb = _embed_model.encode([question])
    sims = cosine_similarity(q_emb, _scope_embeddings)[0]
    return max(sims) >= SCOPE_THRESHOLD


OUT_OF_SCOPE_MESSAGE = (
    "I'm a research assistant focused on AI/NLP papers — that question "
    "looks outside my area. Try asking about topics like retrieval-"
    "augmented generation, LLM agents, or related ML research."
)

# Tool schemas: this is how we DESCRIBE each tool to the LLM so it can
# decide when and how to call it.
# this is effectively the prompt engineering for tool selection.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": (
                "Search a LOCAL pre-indexed collection of ~300 recent NLP/AI "
                "papers about retrieval-augmented generation and LLM agents. "
                "Fast, but limited to this fixed snapshot — may not have very "
                "recent papers or papers outside this specific topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, in natural language.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_arxiv_live",
            "description": (
                "Search arXiv's live API directly for ANY topic, not just "
                "what's in the local index. Slower (real network call), but "
                "covers all of arXiv and finds the most recent papers. Use "
                "this when the local search doesn't have enough, or when "
                "the question needs very recent or out-of-scope papers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, in natural language.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a research assistant that answers questions about
AI/NLP papers using the tools available to you.

Scope:
- You specialize in AI/NLP/ML research topics — things like retrieval-augmented
  generation, LLM agents, transformers, embeddings, etc.
- If the question is NOT related to AI/NLP/ML research (e.g. general knowledge,
  definitions unrelated to this field, casual conversation, or any topic
  outside computer science/AI research), do NOT call any tools. Simply
  explain that you're a research assistant focused on AI/NLP papers and
  can't help with that topic.
- Do not search for papers just because a question contains a word that
  might loosely overlap with paper content (e.g. "ball" matching physics
  papers about colliders is not a real match — use judgment about topical
  relevance, not just keyword overlap).

Rules (for in-scope questions):
- Always use a tool to find real papers before answering factual questions.
- Never invent paper titles, authors, findings, or URLs.
- Try `search_papers` (the local index) first, since it's faster.
- If the local results are insufficient, outdated, empty, or off-topic for
  the question, you MUST immediately call `search_arxiv_live` yourself in
  this same turn. NEVER ask the user for permission to search again, and
  NEVER say things like "would you like me to search" or "I can try
  searching again if you'd like" — just do it.
- Only give a final text answer once you have either found genuinely
  relevant papers, or exhausted both tools without finding anything relevant.
- Base your answer only on information found in the retrieved papers. Do
  not cite, reference, or mention any paper that isn't clearly relevant to
  the question — irrelevant search results should be ignored, not cited.
- A paper only counts as relevant if it substantively addresses the
  question's actual subject — not just because its title or abstract
  happens to share a keyword. A paper about image quality assessment
  that happens to have "Dog" in its title is NOT relevant to "what is
  a dog." Judge relevance by topic and content, not keyword overlap.
- When referencing a paper in your answer text, mention only the paper's
  title (e.g. "According to 'RETA-LLM: A Retrieval-Augmented Large
  Language Model Toolkit'..."). Do NOT list author names in the answer
  text — full author details are already shown separately in the sources
  section, so repeating them inline is redundant and harder to read.
- Do not invent citations or URLs.
- Answer clearly and concisely.
- If after searching both sources you still can't find a relevant answer,
  say so plainly and do not list unrelated papers as if they were sources.
"""

def extract_sources(tool_result: str):
    """
    Extract paper title, authors, and URL from a tool result.
    """
    sources = []
    blocks = tool_result.split("\n\n")

    for block in blocks:
        title = None
        authors = None
        url = None

        for line in block.splitlines():
            if line.startswith("Title: "):
                title = line.replace("Title: ", "", 1).strip()
            elif line.startswith("Authors: "):
                authors = line.replace("Authors: ", "", 1).strip()
            elif line.startswith("URL: "):
                url = line.replace("URL: ", "", 1).strip()

        if title and url:
            sources.append(
                {
                    "title": title,
                    "authors": authors or "Unknown authors",
                    "url": url,
                }
            )

    return sources


def deduplicate_sources(sources):
    """
    Remove duplicate papers using their URL.
    """
    unique_sources = []
    seen_urls = set()

    for source in sources:
        url = source["url"]
        if url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(source)

    return unique_sources


def run_agent(user_question: str, max_turns: int = 5, verbose: bool = True):
    """
    Runs the agent loop until it produces a final text answer (no more
    tool calls) or hits max_turns as a safety limit.
    """
    # --- Scope check happens BEFORE any LLM call ---
    if not is_in_scope(user_question):
        if verbose:
            print(f"\n[Scope guard] Question rejected as out-of-scope: {user_question!r}")
        return OUT_OF_SCOPE_MESSAGE, []

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    sources = []

    for turn in range(max_turns):
        # Retry loop: Groq's models occasionally produce malformed tool
        # calls. Retrying with a slightly higher temperature (more
        # randomness) often produces a well-formed call on the 2nd try.
        response = None
        last_error = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=0.3 + (attempt * 0.2),
                    max_tokens=800,  # caps runaway generations
                )
                break
            except Exception as e:
                last_error = e
                if verbose:
                    print(f"[Retry {attempt + 1}/3] Tool call generation failed: {e}")

        if response is None:
            if verbose:
                print(f"[Agent failed] {last_error}")
            return (
                "I ran into an issue processing that question — it might be "
                "outside my research focus, or the request was too large. "
                "Try rephrasing, or ask about a specific AI/NLP topic.",
                deduplicate_sources(sources),
            )

        message = response.choices[0].message

        # Case 1: the model wants to call one or more tools
        if message.tool_calls:
            # We must append the assistant's tool-call message to history
            # before appending tool results, or the API will reject the
            # conversation as malformed.
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if verbose:
                    print(
                        f"\n[Agent Turn {turn + 1}] "
                        f"Calling tool: {tool_name}({tool_args})"
                    )

                tool_function = AVAILABLE_TOOLS[tool_name]
                tool_result = tool_function(**tool_args)

                if verbose:
                    print(f"[Tool Result Preview] {tool_result[:150]}...")

                found_sources = extract_sources(tool_result)
                sources.extend(found_sources)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

            continue

        # Case 2: the model gave a final text answer, no more tools needed
        else:
            if verbose:
                print(f"\n[Agent finished after {turn + 1} turn(s)]")

            sources = deduplicate_sources(sources)
            return message.content, sources

    # Max turns reached
    sources = deduplicate_sources(sources)
    return (
        "Agent stopped: reached max turns without a final answer.",
        sources,
    )


if __name__ == "__main__":
    test_questions = [
        "What are common approaches for combining retrieval with LLM agents?",
        "what is a ball",
        "what is a dog",
    ]

    for question in test_questions:
        print(f"\n{'=' * 60}\nQUESTION: {question}\n{'=' * 60}")
        answer, sources = run_agent(question)
        print("\n=== FINAL ANSWER ===")
        print(answer)
        print("\n=== SOURCES ===")
        for i, source in enumerate(sources, 1):
            print(f"[{i}] {source['title']}")
            print(f"    {source['url']}")
