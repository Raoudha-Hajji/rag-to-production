"""manual test: embed a query, search Chroma, print top results."""

import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection(name="arxiv_papers")

query = "How do agents decide when to search for more information?"
query_embedding = model.encode([query]).tolist()

results = collection.query(query_embeddings=query_embedding, n_results=3)

for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"\n--- Result {i+1} ---")
    print(f"Title: {meta['title']}")
    print(f"URL: {meta['url']}")
    print(f"Text: {doc[:200]}...")