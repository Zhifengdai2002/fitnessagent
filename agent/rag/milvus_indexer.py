"""CLI helper to build Milvus RAG collections.

Usage:
    python -m agent.rag.milvus_indexer --recreate
"""

from __future__ import annotations

import argparse
from collections import Counter

from agent.rag.documents import build_primary_exercise_documents, build_primary_food_documents, build_primary_knowledge_documents
from agent.rag.milvus_store import (
    build_milvus_collection,
    exercise_collection_name,
    food_collection_name,
    knowledge_collection_name,
    milvus_enabled,
)
from agent.rag.vector_store import embedding_backend_name, embedding_dimensions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FitnessAgent Milvus RAG collections.")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collections before inserting.")
    args = parser.parse_args()

    if not milvus_enabled():
        raise SystemExit("Milvus is not configured. Set RAG_BACKEND=milvus and MILVUS_URI first.")

    exercise_documents = build_primary_exercise_documents()
    food_documents = build_primary_food_documents()
    knowledge_documents = build_primary_knowledge_documents()
    exercise_ok = build_milvus_collection(
        exercise_documents,
        collection_name=exercise_collection_name(),
        recreate=args.recreate,
    )
    food_ok = build_milvus_collection(
        food_documents,
        collection_name=food_collection_name(),
        recreate=args.recreate,
    )
    knowledge_ok = build_milvus_collection(
        knowledge_documents,
        collection_name=knowledge_collection_name(),
        recreate=args.recreate,
    )
    if not exercise_ok or not food_ok or not knowledge_ok:
        raise SystemExit("Failed to build one or more Milvus collections.")

    print(
        "Built Milvus collections: "
        f"{exercise_collection_name()}={len(exercise_documents)} docs {_source_counts(exercise_documents)}, "
        f"{food_collection_name()}={len(food_documents)} docs {_source_counts(food_documents)}, "
        f"{knowledge_collection_name()}={len(knowledge_documents)} docs {_source_counts(knowledge_documents)}. "
        f"Embedding={embedding_backend_name()} dim={embedding_dimensions()}"
    )


def _source_counts(documents: list[dict]) -> dict[str, int]:
    return dict(Counter(str(document.get("metadata", {}).get("source") or "unknown") for document in documents))


if __name__ == "__main__":
    main()
