"""
Run:
    python src/graph_rag/graph_traversal.py --nodes "OpenAI" --top-k 3 --max-depth 2
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


def get_driver():
    if not NEO4J_PASSWORD:
        raise RuntimeError("Missing NEO4J_PASSWORD environment variable")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def fetch_neighbor_edges(tx, node_name: str) -> list[dict[str, Any]]:
    result = tx.run(
        """
        MATCH (current:Entity {name: $node_name})-[relationship]-(neighbor:Entity)
        RETURN
            neighbor.name AS node,
            type(relationship) AS relationship,
            CASE WHEN startNode(relationship) = current THEN 'outgoing' ELSE 'incoming' END AS direction,
            properties(relationship) AS metadata
        ORDER BY node
        """,
        node_name=node_name,
    )
    return [record.data() for record in result]


def bfs_neighbors(
    start_node: str,
    max_depth: int = 2,
    top_k_neighbors: int = 3,
) -> dict[str, Any]:
    driver = get_driver()
    visited = {start_node}
    queue: deque[tuple[str, int]] = deque([(start_node, 0)])
    neighbors: list[dict[str, Any]] = []

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            while queue and len(neighbors) < top_k_neighbors:
                current_node, depth = queue.popleft()
                if depth >= max_depth:
                    continue

                edges = session.execute_read(fetch_neighbor_edges, current_node)
                for edge in edges:
                    next_node = edge["node"]
                    if next_node in visited:
                        continue

                    visited.add(next_node)
                    neighbors.append(
                        {
                            "node": next_node,
                            "depth": depth + 1,
                            "relationship": edge["relationship"],
                            "direction": edge["direction"],
                            "metadata": edge.get("metadata", {}),
                        }
                    )
                    queue.append((next_node, depth + 1))

                    if len(neighbors) >= top_k_neighbors:
                        break
    finally:
        driver.close()

    return {"start_node": start_node, "neighbors": neighbors}


def retrieve_graph_context(
    nodes: list[str],
    max_depth: int = 2,
    top_k_neighbors: int = 3,
) -> list[dict[str, Any]]:
    return [bfs_neighbors(node, max_depth=max_depth, top_k_neighbors=top_k_neighbors) for node in nodes]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve nearest graph neighbors from Neo4j with BFS.")
    parser.add_argument("--nodes", nargs="+", required=True)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-file", default="data/output/graph_context.json")
    args = parser.parse_args()

    context = retrieve_graph_context(
        nodes=args.nodes,
        max_depth=args.max_depth,
        top_k_neighbors=args.top_k,
    )
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(context, ensure_ascii=False, indent=2))
    print(f"Saved graph context -> {output_file}")


if __name__ == "__main__":
    main()
