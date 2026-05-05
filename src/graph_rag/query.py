"""
Run:
    python src/graph_rag/query.py "OpenAI liên quan đến ai?" --top-k 3 --max-depth 2
    python src/graph_rag/query.py "OpenAI liên quan đến ai?" --nodes "OpenAI" --top-k 3 --max-depth 2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_MODEL
from src.graph_rag.graph_traversal import retrieve_graph_context
from src.graph_rag.graph_traversal import get_driver as get_neo4j_driver
from config import NEO4J_DATABASE


def get_llm_client() -> OpenAI:
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


def parse_json_list(content: str) -> list[str]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").removeprefix("json").strip()
    data = json.loads(content)
    entities = data.get("entities", data) if isinstance(data, dict) else data
    if not isinstance(entities, list):
        return []
    return [str(entity).strip() for entity in entities if str(entity).strip()]


def extract_entities_from_question(question: str) -> list[str]:
    client = get_llm_client()
    response = client.chat.completions.create(
        model=get_model_name(),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là hệ thống trích xuất thực thể từ câu hỏi tiếng Việt. "
                    "Chỉ trả về JSON object dạng {\"entities\": [\"...\"]}. "
                    "Thực thể phải là tên riêng, khái niệm, tổ chức, người, sản phẩm hoặc chủ đề chính trong câu hỏi. "
                    "Không giải thích."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return parse_json_list(response.choices[0].message.content or "{}")


def entity_exists(tx, name: str) -> bool:
    result = tx.run("MATCH (n:Entity {name: $name}) RETURN count(n) > 0 AS exists", name=name)
    record = result.single()
    return bool(record and record["exists"])


def find_matching_entities(candidates: list[str], limit: int = 5) -> list[str]:
    if not candidates:
        return []

    driver = get_neo4j_driver()
    matches: list[str] = []
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            for candidate in candidates:
                if session.execute_read(entity_exists, candidate):
                    matches.append(candidate)
                    continue

                records = session.run(
                    """
                    MATCH (n:Entity)
                    WHERE toLower(n.name) CONTAINS toLower($candidate)
                       OR toLower($candidate) CONTAINS toLower(n.name)
                    RETURN n.name AS name
                    ORDER BY size(n.name) DESC
                    LIMIT $limit
                    """,
                    candidate=candidate,
                    limit=limit,
                )
                for record in records:
                    name = record["name"]
                    if name not in matches:
                        matches.append(name)
    finally:
        driver.close()
    return matches[:limit]


def resolve_start_nodes(question: str, nodes: list[str] | None = None) -> list[str]:
    if nodes:
        return nodes
    extracted_entities = extract_entities_from_question(question)
    matched_entities = find_matching_entities(extracted_entities)
    return matched_entities or extracted_entities


def format_graph_context(graph_context: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in graph_context:
        start_node = item.get("start_node", "")
        neighbors = item.get("neighbors", [])
        if not neighbors:
            lines.append(f"Start node: {start_node}\nNo neighbors found.")
            continue

        lines.append(f"Start node: {start_node}")
        for neighbor in neighbors:
            node = neighbor.get("node", "")
            relationship = neighbor.get("relationship", "")
            direction = neighbor.get("direction", "")
            depth = neighbor.get("depth", "")
            metadata = neighbor.get("metadata", {}) or {}
            source_file = metadata.get("source_file", "unknown")
            chunk_index = metadata.get("chunk_index", "unknown")

            if direction == "outgoing":
                triple_text = f"{start_node} -[{relationship}]-> {node}"
            else:
                triple_text = f"{node} -[{relationship}]-> {start_node}"

            lines.append(
                f"- depth={depth}; {triple_text}; source={source_file}; chunk_index={chunk_index}"
            )
    return "\n".join(lines)


def build_prompt(question: str, graph_context: list[dict[str, Any]]) -> str:
    context_text = format_graph_context(graph_context)
    return (
        "Bạn là trợ lý trả lời câu hỏi dựa trên knowledge graph đã trích xuất.\n"
        "Chỉ sử dụng thông tin trong phần Graph context. Nếu không đủ thông tin, hãy nói rõ là không đủ dữ kiện.\n"
        "Trả lời ngắn gọn, chính xác bằng tiếng Việt.\n\n"
        f"Graph context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def answer_graph_query(
    question: str,
    nodes: list[str] | None = None,
    max_depth: int = 2,
    top_k: int = 3,
) -> dict[str, Any]:
    start_nodes = resolve_start_nodes(question, nodes)
    graph_context = retrieve_graph_context(nodes=start_nodes, max_depth=max_depth, top_k_neighbors=top_k)
    prompt = build_prompt(question, graph_context)
    client = get_llm_client()
    response = client.chat.completions.create(
        model=get_model_name(),
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content or ""
    return {
        "question": question,
        "nodes": start_nodes,
        "graph_context": graph_context,
        "answer": answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question using Neo4j graph context and an LLM.")
    parser.add_argument("question", help="User question to answer.")
    parser.add_argument("--nodes", nargs="+", default=None, help="Optional start entity names for BFS retrieval.")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-file", default="data/output/graph_answer.json")
    args = parser.parse_args()

    result = answer_graph_query(
        question=args.question,
        nodes=args.nodes,
        max_depth=args.max_depth,
        top_k=args.top_k,
    )

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(result["answer"])
    print(f"Saved graph answer -> {output_file}")


if __name__ == "__main__":
    main()
