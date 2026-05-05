# src/chunk.py
"""Create overlapped chunks from Markdown files in data/.

Run:
    python -m src.chunk --input-dir data --output-dir data/output

All chunks are saved into one JSON file containing metadata and chunk text.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def create_chunks(text: str, max_chars: int = 6000, overlap_chars: int = 300) -> list[dict[str, Any]]:
    """Split text into fixed-size chunks with character overlap."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be greater than or equal to 0")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    chunks: list[dict[str, Any]] = []
    step = max_chars - overlap_chars
    start = 0
    chunk_index = 1

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                    "length_chars": len(chunk_text),
                    "text": chunk_text,
                }
            )
            chunk_index += 1
        if end == len(text):
            break
        start += step

    return chunks


def chunk_text(text: str, max_chars: int = 6000, overlap_chars: int = 300) -> list[str]:
    """Return only chunk texts for callers that do not need metadata."""
    return [chunk["text"] for chunk in create_chunks(text, max_chars, overlap_chars)]


def build_chunk_records(
    md_path: Path,
    input_dir: Path,
    max_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    content = md_path.read_text(encoding="utf-8")
    chunks = create_chunks(content, max_chars=max_chars, overlap_chars=overlap_chars)
    source_file = md_path.relative_to(input_dir.parent).as_posix()
    records: list[dict[str, Any]] = []

    for chunk in chunks:
        chunk_index = chunk["chunk_index"]
        records.append(
            {
                "metadata": {
                    "source_file": source_file,
                    "chunk_index": chunk_index,
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                    "length_chars": chunk["length_chars"],
                    "overlap_chars": overlap_chars,
                },
                "text": chunk["text"],
            }
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Split Markdown files into overlapped JSON chunks.")
    parser.add_argument(
        "--input-dir",
        default="data",
        help="Directory containing *.md files to chunk (default: data).",
    )
    parser.add_argument(
        "--output-file",
        default="data/output/chunks.json",
        help="Path to the single JSON file where chunks will be written (default: data/output/chunks.json).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2500,
        help="Maximum characters per chunk (default: 6000).",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=200,
        help="Number of overlapping characters between adjacent chunks (default: 300).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_file = Path(args.output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict[str, Any]] = []
    for md_path in sorted(input_dir.rglob("*.md")):
        if output_file.parent in md_path.resolve().parents:
            continue
        records = build_chunk_records(
            md_path=md_path,
            input_dir=input_dir,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )
        all_chunks.extend(records)
        print(f"Chunked {md_path.name}: {len(records)} chunks")

    output_file.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(all_chunks)} chunks -> {output_file}")


if __name__ == "__main__":
    main()
