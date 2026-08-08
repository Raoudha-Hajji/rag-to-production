"""
Fetching papers from the arXiv API returns results as atom feed (XML).
Clean it into python dicts and save them as JSON file.
"""

import feedparser
import json
import time
import urllib.parse

ARXIV_API_URL = "http://export.arxiv.org/api/query"

def fetch_arxiv_papers(query: str, max_results: int = 300, category: str = "cs.CL"):
    """
    query: keyword search
    max_result: nb of papers to pull
    category: category to restrict the search to
    """
    papers = []
    batch_size = 100
    search_query = f'cat:{category} AND all:{query}'

    for start in range(0, max_results, batch_size):
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": min(batch_size, max_results - start),
            "sortBy": "submittedDate",
            "sortOrder":"descending",
        }
        url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"

        print(f"Fetching results {start} to {start + batch_size }...")
        feed = feedparser.parse(url)

        if not feed.entries:
            print("No more results.")
            break
        for entry in feed.entries:
            papers.append({
                "arxiv_id": entry.id.split("/abs/")[-1],
                "title": entry.title.replace("\n", " ").strip(),
                "abstract": entry.summary.replace("\n", " ").strip(),
                "authors": [author.name for author in entry.authors],
                "published": entry.published,
                "url": entry.id,
                "category": category,
            })
            time.sleep(3)

    return papers
    
if __name__ == "__main__":
    # Topic: RAG / LLM agents — ties directly into what this project itself is
    QUERY = "retrieval augmented generation OR large language model agent"
    CATEGORY = "cs.CL"
    MAX_RESULTS = 300

    papers = fetch_arxiv_papers(QUERY, max_results=MAX_RESULTS, category=CATEGORY)

    output_path = "data/papers.json"
    with open(output_path, "w") as f:
        json.dump(papers, f, indent=2)

    print(f"\nSaved {len(papers)} papers to {output_path}")