"""
Run:
    python src/graph_rag/knowledge_graph.py --triples-file data/output/triples.json --clear-existing
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


def sanitize_relationship_type(predicate: str) -> str:
    relation = re.sub(r"\W+", "_", predicate.strip().upper()).strip("_")
    if not relation:
        return "RELATED_TO"
    if relation[0].isdigit():
        relation = f"REL_{relation}"
    return relation


def load_triples(triples_file: Path) -> list[dict[str, Any]]:
    triples = json.loads(triples_file.read_text(encoding="utf-8"))
    if not isinstance(triples, list):
        raise ValueError("triples_file must contain a JSON array")
    return [triple for triple in triples if isinstance(triple, dict)]


def get_driver():
    if not NEO4J_PASSWORD:
        raise RuntimeError("Missing NEO4J_PASSWORD environment variable")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def clear_graph(tx) -> None:
    tx.run("MATCH (n) DETACH DELETE n")


def create_entity_constraint(tx) -> None:
    tx.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")


def ingest_batch(tx, triples: list[dict[str, Any]]) -> None:
    for triple in triples:
        subject = str(triple.get("subject", "")).strip()
        predicate = str(triple.get("predicate", "")).strip()
        obj = str(triple.get("object", "")).strip()
        if not subject or not predicate or not obj:
            continue

        relationship_type = sanitize_relationship_type(predicate)
        metadata = triple.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        properties = {
            "predicate": predicate,
            "source_file": metadata.get("source_file"),
            "chunk_index": metadata.get("chunk_index"),
            "start_char": metadata.get("start_char"),
            "end_char": metadata.get("end_char"),
        }
        properties = {key: value for key, value in properties.items() if value is not None}

        tx.run(
            f"""
            MERGE (subject:Entity {{name: $subject}})
            MERGE (object:Entity {{name: $object}})
            MERGE (subject)-[relationship:{relationship_type}]->(object)
            SET relationship += $properties
            """,
            subject=subject,
            object=obj,
            properties=properties,
        )


def build_knowledge_graph(
    triples_file: str = "data/output/triples.json",
    batch_size: int = 100,
    clear_existing: bool = False,
) -> int:
    triples = load_triples(Path(triples_file))
    driver = get_driver()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            session.execute_write(create_entity_constraint)
            if clear_existing:
                session.execute_write(clear_graph)
            for start in range(0, len(triples), batch_size):
                session.execute_write(ingest_batch, triples[start : start + batch_size])
    finally:
        driver.close()
    return len(triples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Neo4j knowledge graph from extracted triples.")
    parser.add_argument("--triples-file", default="data/output/triples.json")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--clear-existing", action="store_true")
    args = parser.parse_args()

    count = build_knowledge_graph(
        triples_file=args.triples_file,
        batch_size=args.batch_size,
        clear_existing=args.clear_existing,
    )
    print(f"Loaded {count} triples into Neo4j.")


if __name__ == "__main__":
    main()
