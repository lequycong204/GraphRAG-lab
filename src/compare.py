import json
import sys
import time
from pathlib import Path
from typing import Any

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.flat_rag.naive_rag import answer_query as flat_answer
from src.graph_rag.query import answer_graph_query


def load_test_set(path: str = "eval/test_set.json") -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def run_test(test: dict[str, Any]) -> dict[str, Any]:
    question = test["question"]

    flat_start = time.perf_counter()
    flat_result = flat_answer(question, top_k=5)
    flat_time = time.perf_counter() - flat_start

    graph_start = time.perf_counter()
    graph_result = answer_graph_query(question, max_depth=2, top_k=3)
    graph_time = time.perf_counter() - graph_start

    return {
        "id": test["id"],
        "question": question,
        "category": test.get("category", ""),
        "expected_strength": test.get("expected_strength", ""),
        "flat_rag": {
            "answer": flat_result[:200] + ("..." if len(flat_result) > 200 else ""),
            "time": f"{flat_time:.2f}s",
        },
        "graph_rag": {
            "answer": (graph_result.get("answer", "")[:200] + ("..." if len(graph_result.get("answer", "")) > 200 else "")),
            "time": f"{graph_time:.2f}s",
        },
    }


def generate_markdown_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| # | Câu hỏi | Category | Expected | Flat‑RAG time | Graph‑RAG time |",
        "|---|---------|----------|----------|---------------|----------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['question'][:60]}... | {r['category']} | "
            f"{r['expected_strength']} | {r['flat_rag']['time']} | {r['graph_rag']['time']} |"
        )

    lines.extend(
        [
            "",
            "### Flat‑RAG answers",
            "",
            "| # | Câu hỏi | Trả lời |",
            "|---|---------|---------|",
        ]
    )
    for r in results:
        lines.append(f"| {r['id']} | {r['question'][:60]}... | {r['flat_rag']['answer']} |")

    lines.extend(
        [
            "",
            "### Graph‑RAG answers",
            "",
            "| # | Câu hỏi | Trả lời |",
            "|---|---------|---------|",
        ]
    )
    for r in results:
        lines.append(f"| {r['id']} | {r['question'][:60]}... | {r['graph_rag']['answer']} |")

    return "\n".join(lines)


def update_readme(table_content: str) -> None:
    readme_path = root_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text("# GraphRAG Lab\n\n", encoding="utf-8")

    content = readme_path.read_text(encoding="utf-8")
    marker_begin = "<!-- COMPARE_TABLE_BEGIN -->"
    marker_end = "<!-- COMPARE_TABLE_END -->"

    new_section = f"{marker_begin}\n\n## So sánh Flat‑RAG vs Graph‑RAG\n\n{table_content}\n\n{marker_end}"

    if marker_begin in content:
        before = content.split(marker_begin)[0]
        after = content.split(marker_end)[1] if marker_end in content else ""
        content = before + new_section + after
    else:
        content = content.rstrip() + "\n\n" + new_section + "\n"

    readme_path.write_text(content, encoding="utf-8")
    print(f"Updated README.md -> {readme_path}")


def main() -> None:
    test_set = load_test_set()

    all_results: list[dict[str, Any]] = []
    for i, test in enumerate(test_set):
        print(f"[{i + 1}/{len(test_set)}] {test['question'][:60]}...")
        result = run_test(test)
        all_results.append(result)
        print(f"  Flat‑RAG  ({result['flat_rag']['time']}): {result['flat_rag']['answer'][:80]}...")
        print(f"  Graph‑RAG ({result['graph_rag']['time']}): {result['graph_rag']['answer'][:80]}...")
        print()

    # Save raw results
    output_path = root_dir / "eval" / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Generate and save markdown table
    markdown = generate_markdown_table(all_results)
    update_readme(markdown)
    print(f"Saved raw results -> {output_path}")


if __name__ == "__main__":
    main()