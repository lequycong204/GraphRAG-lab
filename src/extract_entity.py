# src/extract_entity.py
"""Extract knowledge triples from chunked corpus using an OpenAI-compatible LLM.

Environment variables are loaded through config.py:
- NVIDIA_API_KEY: API key for the LLM provider.
- NVIDIA_BASE_URL: OpenAI-compatible base URL.
- NVIDIA_MODEL: model name, defaults in config.py.

Run:
    python -m src.extract_entity --chunks-file data/output/chunks.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_MODEL


SYSTEM_PROMPT = """Bạn là hệ thống trích xuất knowledge graph triples từ văn bản tiếng Việt.
Nhiệm vụ: đọc một chunk văn bản và trả về danh sách triples dạng JSON array.
Mỗi triple có đúng 3 trường: subject, predicate, object.

<rule>
- Chỉ trích xuất thông tin có trong chunk, không suy đoán.
- Subject và object phải là thực thể hoặc giá trị cụ thể.
- Predicate viết bằng UPPER_SNAKE_CASE tiếng Anh, ví dụ: FOUNDED_BY, FOUNDED_IN, IS_A, PART_OF, DEVELOPED_BY.
- Nếu một câu có nhiều object cho cùng một quan hệ, tách thành nhiều triples.
- Không trả markdown, không giải thích, chỉ trả JSON hợp lệ.
</rule>

<example>
Input: OpenAI được thành lập bởi Sam Altman và Elon Musk vào năm 2015.
Output:
[
  {"subject":"OpenAI","predicate":"FOUNDED_BY","object":"Sam Altman"},
  {"subject":"OpenAI","predicate":"FOUNDED_BY","object":"Elon Musk"},
  {"subject":"OpenAI","predicate":"FOUNDED_IN","object":"2015"}
]
</example>
"""


def get_client() -> OpenAI:
    if not NVIDIA_API_KEY:
        raise RuntimeError("Missing NVIDIA_API_KEY environment variable")
    kwargs: dict[str, Any] = {"api_key": NVIDIA_API_KEY}
    if NVIDIA_BASE_URL:
        kwargs["base_url"] = NVIDIA_BASE_URL
    return OpenAI(**kwargs)


def get_model_name() -> str:
    if NVIDIA_BASE_URL and "integrate.api.nvidia.com" in NVIDIA_BASE_URL and "/" not in NVIDIA_MODEL:
        return f"openai/{NVIDIA_MODEL}"
    return NVIDIA_MODEL


def parse_llm_json(content: str) -> list[dict[str, str]]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content.removeprefix("json").strip()
    data = json.loads(content)
    triples = data.get("triples", data) if isinstance(data, dict) else data
    if not isinstance(triples, list):
        return []

    parsed: list[dict[str, str]] = []
    for item in triples:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        predicate = str(item.get("predicate", "")).strip()
        obj = str(item.get("object", "")).strip()
        if subject and predicate and obj:
            parsed.append({"subject": subject, "predicate": predicate, "object": obj})
    return parsed


# Global token counters
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0


def extract_triples_from_text(client: OpenAI, text: str) -> list[dict[str, str]]:
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS
    response = client.chat.completions.create(
        model=get_model_name(),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Trích xuất triples từ chunk văn bản sau. "
                    "Trả về JSON object có key `triples`, value là array triples.\n\n"
                    f"Chunk:\n{text}"
                ),
            },
        ],
    )
    usage = getattr(response, "usage", None)
    if usage:
        TOTAL_PROMPT_TOKENS += usage.prompt_tokens or 0
        TOTAL_COMPLETION_TOKENS += usage.completion_tokens or 0
    return parse_llm_json(response.choices[0].message.content or "{}")


def load_chunks(chunks_file: Path) -> list[dict[str, Any]]:
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise ValueError("chunks_file must contain a JSON array")
    return chunks


def extract_triples_from_chunks(chunks_file: Path) -> list[dict[str, Any]]:
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS
    TOTAL_PROMPT_TOKENS = 0
    TOTAL_COMPLETION_TOKENS = 0

    client = get_client()
    all_triples: list[dict[str, Any]] = []

    for chunk in load_chunks(chunks_file):
        if not isinstance(chunk, dict):
            continue
        text = str(chunk.get("text", "")).strip()
        metadata = chunk.get("metadata", {})
        if not text or not isinstance(metadata, dict):
            continue

        triples = extract_triples_from_text(client, text)
        for triple in triples:
            all_triples.append(
                {
                    **triple,
                    "metadata": {
                        "source_file": metadata.get("source_file"),
                        "chunk_index": metadata.get("chunk_index"),
                        "start_char": metadata.get("start_char"),
                        "end_char": metadata.get("end_char"),
                    },
                }
            )

    return deduplicate_triples(all_triples)


def deduplicate_triples(triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str, int | None]] = set()
    unique: list[dict[str, Any]] = []
    for triple in triples:
        metadata = triple.get("metadata", {})
        key = (
            triple["subject"],
            triple["predicate"],
            triple["object"],
            str(metadata.get("source_file")),
            metadata.get("chunk_index"),
        )
        if key not in seen:
            seen.add(key)
            unique.append(triple)
    return unique


def print_token_usage() -> None:
    total = TOTAL_PROMPT_TOKENS + TOTAL_COMPLETION_TOKENS
    print(
        f"\n>>> Token usage: {total:,} total "
        f"(prompt: {TOTAL_PROMPT_TOKENS:,} / completion: {TOTAL_COMPLETION_TOKENS:,})"
    )


def main() -> None:
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS

    parser = argparse.ArgumentParser(description="Extract triples from chunk JSON file.")
    parser.add_argument(
        "--chunks-file",
        default="data/output/chunks.json",
        help="Input JSON file generated by src.chunk (default: data/output/chunks.json).",
    )
    parser.add_argument(
        "--output-file",
        default="data/output/triples.json",
        help="Output JSON file for triples with chunk metadata (default: data/output/triples.json).",
    )
    args = parser.parse_args()

    chunks_file = Path(args.chunks_file).resolve()
    output_file = Path(args.output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    triples = extract_triples_from_chunks(chunks_file)
    output_file.write_text(json.dumps(triples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Extracted {len(triples)} triples -> {output_file}")
    print_token_usage()


if __name__ == "__main__":
    main()
