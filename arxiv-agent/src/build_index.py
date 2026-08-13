"""
Embed the abstracts from papers.json using SBERT, and store them in a local chroma vector db.

this makes the retrival part if RAG
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/papers.json"
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "arxiv_papers"

def load_papers(path: str):
    with open(path, "r") as f:
        return json.load(f)

def build_index():
    papers = load_papers(DATA_PATH)
    print(f"Loaded {len(papers)} papers.")

    # Print SBERT model
    print("Loading SBERT model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    #Chroma client saves the DB to disk, so no need to re-embed each run
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    #Make or reuse a "table" for the vectors (safe to rerun the script)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    #Embed tiles + abstracts together to improve retrieval 
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]

    print("Generating embeddings (this may take a minute)...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Chroma needs: unique ids, the vectors, and metadata for each entry.
    ids = [p["arxiv_id"] for p in papers]
    metadatas = [ 
        {
            "title": p["title"],
            "authors": ", ".join(p["authors"]),
            "published": p["published"],
            "url": p["url"],
        }
        for p in papers
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Indexed {len(papers)} papers into Chroma at '{CHROMA_PATH}'.")


if __name__ == "__main__":
    build_index()