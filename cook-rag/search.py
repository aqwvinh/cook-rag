from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---- required env vars -------------------------------------------------
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENV"]
INDEX_NAME = os.getenv("INDEX_NAME", "cook-rag-index")
NAMESPACE = os.getenv("NAMESPACE", "cook-rag-namespace")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
# ------------------------------------------------------------------------

# 1 · instantiate model & index handle
model = SentenceTransformer(MODEL_NAME)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)


def search(query: str, top_k: int = TOP_K):
    """Embed → query Pinecone → pretty-print matches."""
    vec = model.encode(query).tolist()
    res = index.query(
        vector=vec,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=True,  # we saved chunk text + source as metadata
        include_values=False,
    )

    if not res["matches"]:
        print("No hits 🤷")
        return

    for rank, m in enumerate(res["matches"], 1):
        text = m["metadata"].get("text", "[chunk text missing]")
        source = m["metadata"].get("source", "unknown.pdf")
        print(f"{rank:>2}. score={m['score']:.3f} │ {source}")
        print("   ", text[:250].replace("\n", " "), "…\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run search.py "your question here"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    search(query)
