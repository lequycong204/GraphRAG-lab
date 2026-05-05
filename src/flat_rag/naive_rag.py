"""Query module for Flat‑RAG.

Provides `answer_query` which loads the Chroma collection, retrieves the most
relevant chunks for a user query, builds a prompt and calls the LLM.

The indexing (building/updating the vector store) is handled by
`src.flat_rag.index` which wraps the functions in `vector_db.py`.
"""
import os
import sys
from pathlib import Path
from typing import Any

import chromadb
from openai import OpenAI

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Load configuration (fallback to env vars if config module missing)
try:
    from config import (
        NVIDIA_API_KEY,
        NVIDIA_BASE_URL,
        NVIDIA_MODEL,
        CHROMA_PERSIST_DIR,
    )
except ImportError:
    print("Error: Could not import configuration from src.config")

# Helper to create a Chroma client
def _get_client() -> chromadb.Client:
    # Use PersistentClient so that the stored vectors in the Chroma DB are loaded
    dir_path = Path(CHROMA_PERSIST_DIR).resolve()
    dir_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(dir_path))

def _get_collection(client: chromadb.Client) -> Any:
    """Return the "flat_rag" collection, creating it if it does not exist."""
    try:
        return client.get_collection(name="flat_rag")
    except Exception as e:
        print(f"Error: Can not get collection: {e}")

# Retrieve top‑k similar chunks for a query string
def retrieve_chunks(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    client = _get_client()
    collection = _get_collection(client)
    if collection is None:
        return []

    from src.flat_rag.vector_db import _embed

    results = collection.query(
        query_embeddings=_embed([query]),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks: list[dict[str, Any]] = []
    for doc, meta, dist in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        chunks.append({"text": doc, "metadata": meta, "distance": dist})
    return chunks

# Build a prompt for the LLM using retrieved chunks
def build_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    context = "\n\n---\n\n".join(chunk["text"] for chunk in chunks)
    prompt = (
        f"You are an assistant answering a user query based on the provided context.\n"
        f"Answer concisely and only use information from the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    return prompt

# Create OpenAI‑compatible client
def _get_llm_client() -> OpenAI:
    if not NVIDIA_API_KEY:
        raise RuntimeError("Missing NVIDIA_API_KEY environment variable")
    kwargs: dict[str, Any] = {"api_key": NVIDIA_API_KEY}
    if NVIDIA_BASE_URL:
        kwargs["base_url"] = NVIDIA_BASE_URL
    return OpenAI(**kwargs)

def get_model_name() -> str:
    # Preserve same logic as extract_entity for consistency
    if NVIDIA_BASE_URL and "integrate.api.nvidia.com" in NVIDIA_BASE_URL and "/" not in NVIDIA_MODEL:
        return f"openai/{NVIDIA_MODEL}"
    return NVIDIA_MODEL

def answer_query(query: str, top_k: int = 5) -> str:
    chunks = retrieve_chunks(query, top_k)
    if not chunks:
        return "No relevant information found."
    prompt = build_prompt(query, chunks)
    client = _get_llm_client()
    response = client.chat.completions.create(
        model=get_model_name(),
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""  # type: ignore

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Answer a user query using Flat‑RAG (Chroma DB).")
    parser.add_argument("query", help="User question to answer.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of similar chunks to retrieve.")
    args = parser.parse_args()
    answer = answer_query(args.query, args.top_k)
    print(answer)