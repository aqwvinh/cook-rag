from __future__ import annotations

import io
import itertools
import os
import uuid

import tiktoken
import tqdm
from google.cloud import storage
from pinecone import Pinecone
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

GCS_BUCKET = os.environ["GCS_BUCKET"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENV"]
INDEX_NAME = os.getenv("INDEX_NAME", "cook-rag-index")
NAMESPACE = os.getenv("NAMESPACE", "cook-rag-namespace")

ENC = tiktoken.get_encoding("cl100k_base")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TOKENS_PER_CHUNK = int(os.getenv("CHUNK_TOKENS", "800"))
BATCH_UPSERT = 64  # vectors per Pinecone upsert


print("Connecting to GCS bucket:", GCS_BUCKET)
gcs_client = storage.Client()
bucket = gcs_client.bucket(GCS_BUCKET)

print("Loading embedder:", MODEL_NAME)
embedder = SentenceTransformer(MODEL_NAME)
dim = embedder.get_sentence_embedding_dimension()

print("Connecting to Pinecone …")
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)
print("Using index:", INDEX_NAME, "| dimension:", dim)


def chunks_from_pdf(data: bytes) -> list[str]:
    """Return ≈800-token chunks extracted from one PDF."""
    reader = PdfReader(io.BytesIO(data))
    all_text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
    if not all_text:
        return []

    tokens = ENC.encode(all_text)
    return [
        ENC.decode(tokens[i : i + TOKENS_PER_CHUNK]).strip()
        for i in range(0, len(tokens), TOKENS_PER_CHUNK)
        if tokens[i : i + TOKENS_PER_CHUNK]
    ]


def grouper(iterable, size):
    """('ABCDEF', 2) → ('AB','CD','EF')"""
    it = iter(iterable)
    while grp := tuple(itertools.islice(it, size)):
        yield grp


# iterate PDFs
blobs = [b for b in bucket.list_blobs() if b.name.lower().endswith(".pdf")]
if not blobs:
    print("No PDFs found, aborting.")
    exit(1)

total = 0
for blob in blobs:
    print(f"\n{blob.name}")
    pdf_bytes = blob.download_as_bytes()
    chunk_texts = chunks_from_pdf(pdf_bytes)
    print(f"   {len(chunk_texts)} chunks")

    # batch-embed and upsert
    for batch in grouper(chunk_texts, BATCH_UPSERT):
        vectors = embedder.encode(batch).tolist()
        ids = [str(uuid.uuid4()) for _ in batch]
        metas = [{"source": blob.name}] * len(batch)
        index.upsert(list(zip(ids, vectors, metas)), namespace=NAMESPACE)
        total += len(batch)
        tqdm.tqdm.write(f"upserted {len(batch)} vectors")

print(f"\nFinished. Upserted {total} vectors.")
print(index.describe_index_stats(namespace=NAMESPACE))
