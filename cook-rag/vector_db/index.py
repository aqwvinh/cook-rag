import os

from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "cook-rag-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws", region="us-east-1"
        ),  # AWS only for the free tier
    )
