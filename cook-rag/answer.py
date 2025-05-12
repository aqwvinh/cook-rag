"""
1. Embed the user query with the same Sentence-Transformers model
2. Retrieve top-K chunks from Pinecone
3. Send context + question to a local Meta-Llama-3.1-8B-Instruct LLM
   via Hugging Face Transformers pipeline
"""

import os
import sys
import textwrap

import torch
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import TextGenerationPipeline, pipeline

# ── Load secrets & config ────────────────────────────────────────────────
load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENV"]
INDEX_NAME = os.getenv("INDEX_NAME", "cook-rag-index")
NAMESPACE = os.getenv("NAMESPACE", "cook-rag-namespace")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "800"))

LLM_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
# ────────────────────────────────────────────────────────────────────────

# 1) Initialize embedder & Pinecone client
embedder = SentenceTransformer(EMBED_MODEL)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)

# 2) Instantiate the Transformers pipeline for Llama-Instruct
llm_pipeline: TextGenerationPipeline = pipeline(
    "text-generation",
    model=LLM_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


def retrieve_context(query: str) -> list[str]:
    """Embed query → retrieve top-K chunk texts from Pinecone."""
    vec = embedder.encode(query).tolist()
    res = index.query(
        vector=vec,
        top_k=TOP_K,
        namespace=NAMESPACE,
        include_metadata=True,
        include_values=False,
    )
    return [m["metadata"].get("text", "") for m in res["matches"]]


def answer(question: str):
    chunks = retrieve_context(question)
    if not chunks:
        print("No context found")
        return

    system = "You are a helpful cooking assistant. Use only the provided context from recipe books."
    context = "\n---\n".join(chunks)
    prompt = textwrap.dedent(f"""
        [INST] <<SYS>>
        {system}
        <</SYS>>

        ### Context
        {context}

        ### Question
        {question}

        ### Answer
        [/INST]
    """)

    outputs = llm_pipeline(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
    )
    # the generated text includes the prompt; strip it off
    generated = outputs[0]["generated_text"]
    answer = generated[len(prompt) :].strip()
    print("\nAnswer:\n", answer)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run answer.py "Your question here"')
        sys.exit(1)
    answer(" ".join(sys.argv[1:]))
