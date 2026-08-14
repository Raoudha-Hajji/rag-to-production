# 🔎 ArXiv Research Assistant Agent

An agentic RAG (Retrieval-Augmented Generation) assistant that answers questions about AI/NLP research by searching a local pre-indexed paper collection and falling back to live arXiv search when needed — with source citations for every answer.

Part of the [`rag-to-production`](https://github.com/Raoudha-Hajji/rag-to-production) portfolio sprint.

**🌐 Live demo:** https://arxiv-search-agent.streamlit.app 

---

## What it does

Ask a question about AI/NLP research, and the agent:

1. Searches a locally indexed collection of ~300 recent arXiv papers (fast, semantic search)
2. If the local results are weak, outdated, or off-topic, autonomously falls back to a live arXiv API search — no local reindexing required
3. Synthesizes an answer grounded only in the retrieved papers, with clickable source citations
4. Refuses to answer (rather than hallucinate) when nothing relevant is found, or when the question is outside its research scope

This is what makes it *agentic* rather than a fixed retrieval pipeline: the LLM itself decides which tool to call, when to call a second tool, and when it has enough information to answer — all without any user intervention.

## How it works

```
User question
     │
     ▼
Scope guard (embedding similarity check)
     │  → rejects off-topic questions before any LLM call
     ▼
LLM agent loop (Groq / Llama 3.1 8B)
     │
     ├─► search_papers()      → Chroma local vector index (SBERT embeddings)
     │
     └─► search_arxiv_live()  → arXiv API (real-time, uncapped by local index)
     │
     ▼
Grounded answer + deduplicated source citations
```

**Key design decisions:**

- **Two-tool retrieval strategy** : a fast local index for the common case, with a live API fallback for freshness and out-of-scope-index topics. Avoids the need for scheduled reindexing.
- **Relevance filtering, not just top-k retrieval** : local search filters out weak semantic matches by distance threshold, rather than always returning the "closest available" results even when nothing is actually relevant.
- **Scope guard before the LLM call** : an embedding-similarity pre-filter rejects clearly out-of-domain questions before spending any tokens or risking a hallucinated/ungrounded answer.
- **Grounded citation instructions** : the system prompt explicitly forbids citing papers that don't substantively address the question, not just ones that share a keyword.

## Tech stack

| Component | Choice | Why |
|---|---|---|
| LLM | Llama 3.1 8B Instant (via Groq) | Fast inference, free tier suitable for a demo |
| Embeddings | `all-MiniLM-L6-v2` (Sentence-Transformers) | Small, fast, strong baseline for semantic search |
| Vector store | ChromaDB (persistent, local) | Simple, no external infra needed for a portfolio-scale index |
| Live retrieval | arXiv API (via `feedparser`) | Free, no auth required, covers all of arXiv |
| UI | Streamlit | Fast to build, easy to deploy and share |

## Project structure

```
arxiv-agent/
├── data/
│   └── chroma_db/          # persisted vector index
├── src/
│   ├── fetch_papers.py     # pulls papers from arXiv API into data/papers.json
│   ├── build_index.py      # embeds papers with SBERT, stores in Chroma
│   ├── tools.py            # search_papers() and search_arxiv_live() tool functions
│   ├── agent.py            # tool-calling agent loop, system prompt, scope guard
│   └── app.py               # Streamlit demo interface
├── requirements.txt
└── README.md
```

## Running it locally

```bash
# clone and enter the project
git clone https://github.com/Raoudha-Hajji/rag-to-production.git
cd rag-to-production/arxiv-agent

# set up environment
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# add your Groq API key
echo GROQ_API_KEY=your_key_here > .env

# (optional) rebuild the index from scratch
python src/fetch_papers.py
python src/build_index.py

# run the app
streamlit run src/app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

## Known limitations

This is a portfolio-scale prototype, not a production system:

- Groq's free tier has a low rate limit (6,000 tokens/minute) — heavy or concurrent usage can hit it
- The local index is a static snapshot (~300 papers); freshness relies entirely on the live-search fallback rather than scheduled reindexing
- No authentication, usage limits, or multi-user session isolation
- Single-process Streamlit deployment — not built for concurrent traffic at scale

These are intentional scope decisions for a focused demo — see the [`rag-to-production`](https://github.com/Raoudha-Hajji/rag-to-production) repo for the other sub-projects in this sprint (data pipeline, LLM evaluation harness).
