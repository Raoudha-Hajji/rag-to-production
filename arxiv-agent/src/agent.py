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
from tools import AVAILABLE_TOOLS

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "llama-3.1-8b-instant"

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

Rules:
- Always use a tool to find real papers before answering factual questions.
- Never invent paper titles, authors, findings, or URLs.
- Try `search_papers` (the local index) first, since it's faster.
- If the local results are insufficient, outdated, or off-topic for the
  question, call `search_arxiv_live` to search more broadly.
- Base your answer only on information found in the retrieved papers.
- Do not invent citations or URLs.
- Answer clearly and concisely.
- If after searching you still can't find a good answer, say so honestly
  instead of guessing.
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
                )
                break
            except Exception as e:
                last_error = e
                if verbose:
                    print(f"[Retry {attempt + 1}/3] Tool call generation failed: {e}")

        if response is None:
            return (
    f"Agent failed after retries: {last_error}", deduplicate_sources(sources),)

        message = response.choices[0].message

        # Case 1: the model wants to call one or more tools
        if message.tool_calls:
            # We must append the assistant's tool-call message to history
            # before appending tool results, or the API will reject the
            # conversation as malformed.
            messages.append(message)

            for tool_call in message.tool_calls:

                tool_name = tool_call.function.name

                tool_args = json.loads(
                    tool_call.function.arguments
                )

                if verbose:
                    print(
                        f"\n[Agent Turn {turn + 1}] "
                        f"Calling tool: {tool_name}({tool_args})"
                    )

                # Get actual Python function
                tool_function = AVAILABLE_TOOLS[tool_name]

                # Execute tool
                tool_result = tool_function(**tool_args)

                if verbose:
                    print(
                        f"[Tool Result Preview] "
                        f"{tool_result[:150]}..."
                    )

                found_sources = extract_sources(tool_result)

                sources.extend(found_sources)

                # Feed the tool's output back into the conversation so the
                # model can see it and decide what to do next.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

            # Loop again: the model will now see the tool result and
            # decide whether it has enough info, or needs another tool call.
            continue

        # Case 2: the model gave a final text answer, no more tools needed
        else:
            if verbose:
                print(
                    f"\n[Agent finished after "
                    f"{turn + 1} turn(s)]")

                # Remove duplicate papers
            sources = deduplicate_sources(sources)

            return message.content, sources

    #Max turns reached
    sources = deduplicate_sources(sources)

    return (
        "Agent stopped: reached max turns without a final answer.",
        sources,
     )



if __name__ == "__main__":

    question = (
        "What are common approaches for combining "
        "retrieval with LLM agents?"
    )

    answer, sources = run_agent(question)

    print("\n=== FINAL ANSWER ===")
    print(answer)

    print("\n=== SOURCES ===")

    for i, source in enumerate(sources, 1):
        print(f"[{i}] {source['title']}")
        print(f"    {source['url']}")