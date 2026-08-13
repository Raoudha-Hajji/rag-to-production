"""
the agent's tools are functions the agent can choose to call to gather information.
Each functoin has one clear job and the llm (agent) decides when to call on each of them based on the user's question.
The decision making is the model's job.
"""

import chromadb
from sentence_transformers import SentenceTransformer
import feedparser
import urllib.parse

#Loading the models once
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
_client = chromadb.PersistentClient(path="data/chroma_db")
_collection = _client.get_collection(name="arxiv_papers")

def search_papers(query: str, n_results: int = 5) -> str:
    """
    Searches the papers we indexed.
    Limited search.
    """
    query_embedding = _embed_model.encode([query]).tolist()
    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    if not results["documents"][0]:
        return "No relevant papers found in the local index."

    output = []

    for doc, meta in zip(
        results["documents"][0],
        results["metadatas"][0],
    ):
        authors = meta.get("authors", "Unknown authors")

        output.append(
            f"Title: {meta['title']}\n"
            f"Authors: {authors}\n"
            f"URL: {meta['url']}\n"
            f"Abstract: {doc[:300]}..."
        )

    return "\n\n".join(output)

def search_arxiv_live(query: str, max_results: int = 5) -> str:
    """
    Searches arXiv's live API directly.
    Not limited, includes recent papers but slower.
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
    feed = feedparser.parse(url)

    if not feed.entries:
        return "No results found on arXiv."

    output = []

    for entry in feed.entries:
        title = entry.title.replace("\n", " ").strip()
        abstract = entry.summary.replace("\n", " ").strip()

        authors = ", ".join(
            author.name for author in entry.authors
        )

        output.append(
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"URL: {entry.id}\n"
            f"Abstract: {abstract[:300]}..."
        )

    return "\n\n".join(output)


# This maps tool names (strings) to the actual Python functions,
AVAILABLE_TOOLS = {
    "search_papers": search_papers,
    "search_arxiv_live": search_arxiv_live,
}