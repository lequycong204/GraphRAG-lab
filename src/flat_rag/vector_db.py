'''
Run: python -m src.flat_rag.vector_db build 
'''

import os
import json
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Ensure the project root is in sys.path for config import
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load configuration (fallback to env vars if config module missing)
try:
    from src.config import (
        NVIDIA_API_KEY,
        NVIDIA_BASE_URL,
        CHROMA_PERSIST_DIR,
    )
except ImportError:
    # Fallback to root config module
    try:
        from config import (
            NVIDIA_API_KEY,
            NVIDIA_BASE_URL,
            CHROMA_PERSIST_DIR,
        )
    except ImportError:
        NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
        NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL")
        CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma-db")

# Simple embedding placeholder – replace with real model call if available
def _embed(texts: List[str]) -> List[List[float]]:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings

def _get_client(persist_dir: str = CHROMA_PERSIST_DIR) -> chromadb.Client:
    dir_path = Path(persist_dir).resolve()
    dir_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(dir_path))

def build_vector_db(
    chunks_file: str = "data/output/chunks.json",
    persist_dir: str = CHROMA_PERSIST_DIR,
    batch_size: int = 64,
) -> None:
    """Read chunk records, compute embeddings, and store them in a Chroma collection.

    This creates (or overwrites) a collection named "flat_rag".
    """
    client = _get_client(persist_dir)
    # Delete existing collection if present to ensure a clean rebuild
    if "flat_rag" in client.list_collections():
        client.delete_collection("flat_rag")
    collection = client.create_collection(name="flat_rag")

    # Load chunks
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]
    ids = [str(i) for i in range(len(chunks))]
    metadatas = [chunk.get("metadata", {}) for chunk in chunks]

    # Process in batches to avoid large payloads
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_meta = metadatas[start:end]
        embeddings = _embed(batch_texts)
        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_meta,
        )
    print(f"Built Chroma vector store with {len(texts)} vectors.")


def update_vector_db(
    new_chunks_file: str = "data/output/chunks.json",
    persist_dir: str = CHROMA_PERSIST_DIR,
    batch_size: int = 64,
) -> None:
    """Incrementally add new chunks to the existing collection.
    It assumes the collection already exists.
    """
    client = _get_client(persist_dir)
    collection = client.get_collection(name="flat_rag")

    with open(new_chunks_file, "r", encoding="utf-8") as f:
        new_chunks = json.load(f)

    # Determine current max id to avoid collisions
    existing_ids = set(collection.get()['ids']) if collection.get()['ids'] else set()
    start_index = len(existing_ids)

    texts = [c["text"] for c in new_chunks]
    ids = [str(start_index + i) for i in range(len(new_chunks))]
    metadatas = [c.get("metadata", {}) for c in new_chunks]

    for s in range(0, len(texts), batch_size):
        e = s + batch_size
        batch_texts = texts[s:e]
        batch_ids = ids[s:e]
        batch_meta = metadatas[s:e]
        embeddings = _embed(batch_texts)
        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_meta,
        )
    print(f"Added {len(texts)} new vectors to the store.")


def clear_vector_db(persist_dir: str = CHROMA_PERSIST_DIR) -> None:
    """Delete the entire Chroma persistence directory.
    Use with care – this removes all stored vectors.
    """
    client = _get_client(persist_dir)
    if "flat_rag" in client.list_collections():
        client.delete_collection("flat_rag")
        print("Cleared flat_rag collection.")
    else:
        print("No flat_rag collection found to clear.")

import sys

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manage Chroma vector store for Flat‑RAG.")
    subparsers = parser.add_subparsers(dest="command")  # optional subcommand

    build = subparsers.add_parser("build", help="Build the vector store from chunks.")
    build.add_argument("--chunks-file", default="data/output/chunks.json")
    build.add_argument("--persist-dir", default=CHROMA_PERSIST_DIR)
    build.add_argument("--batch-size", type=int, default=64)

    update = subparsers.add_parser("update", help="Add new chunks to the existing store.")
    update.add_argument("--chunks-file", default="data/output/chunks.json")
    update.add_argument("--persist-dir", default=CHROMA_PERSIST_DIR)
    update.add_argument("--batch-size", type=int, default=64)

    clear = subparsers.add_parser("clear", help="Delete the vector store.")
    clear.add_argument("--persist-dir", default=CHROMA_PERSIST_DIR)

    # If no arguments (only script name), default to building the store
    if len(sys.argv) == 1:
        # Implicit build with defaults
        build_vector_db("data/output/chunks.json", CHROMA_PERSIST_DIR, 64)
    else:
        args = parser.parse_args()
        if args.command == "build":
            build_vector_db(args.chunks_file, args.persist_dir, args.batch_size)
        elif args.command == "update":
            update_vector_db(args.chunks_file, args.persist_dir, args.batch_size)
        elif args.command == "clear":
            clear_vector_db(args.persist_dir)
