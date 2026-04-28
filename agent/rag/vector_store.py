"""Tiny persistent vector store for local RAG documents.

This intentionally avoids a heavyweight service in phase one. It uses a
deterministic hashing embedding so the retrieval path is testable offline; the
module boundary can later be backed by Chroma, Milvus, or an external embedding
model without changing planner/tool code.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "rag_index"
EXERCISE_INDEX_PATH = INDEX_DIR / "exercise_index.json"
EMBEDDING_DIMENSIONS = 384


def build_index(documents: list[dict[str, Any]], index_path: Path = EXERCISE_INDEX_PATH) -> dict[str, Any]:
    """Build and persist a local vector index."""

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    indexed_documents = []
    for document in documents:
        text = document_text(document)
        indexed_documents.append(
            {
                "document": document,
                "embedding": embed_text(text),
            }
        )
    index = {
        "version": 1,
        "embedding": "hashing-bow-v1",
        "dimensions": EMBEDDING_DIMENSIONS,
        "documents": indexed_documents,
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def load_index(index_path: Path = EXERCISE_INDEX_PATH) -> dict[str, Any]:
    """Load a persisted index."""

    if not index_path.exists():
        return {}
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def search_index(
    query: str,
    *,
    index: dict[str, Any],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return nearest documents with cosine scores."""

    query_vector = embed_text(query)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for position, item in enumerate(index.get("documents", [])):
        if not isinstance(item, dict):
            continue
        document = item.get("document")
        embedding = item.get("embedding")
        if not isinstance(document, dict) or not isinstance(embedding, list):
            continue
        score = cosine_similarity(query_vector, embedding)
        scored.append((score, position, document))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "score": score,
            "document": document,
        }
        for score, _, document in scored[: max(1, limit)]
        if score > 0
    ]


def document_text(document: dict[str, Any]) -> str:
    return "\n".join(
        str(piece)
        for piece in [
            document.get("title", ""),
            document.get("text", ""),
            json.dumps(document.get("metadata", {}), ensure_ascii=False),
        ]
        if piece
    )


def embed_text(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    for token in tokenize(text):
        index = stable_hash(token) % EMBEDDING_DIMENSIONS
        vector[index] += token_weight(token)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [value / norm for value in vector]


def tokenize(text: str) -> list[str]:
    normalized = text.lower().replace("_", " ").replace("-", " ")
    tokens = re.findall(r"[a-z0-9]+", normalized)
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(SYNONYMS.get(token, []))
    return expanded


def stable_hash(token: str) -> int:
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:12], 16)


def token_weight(token: str) -> float:
    if token in HIGH_VALUE_TOKENS:
        return 1.8
    if len(token) <= 2:
        return 0.4
    return 1.0


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


SYNONYMS = {
    "shoulder": ["delts", "deltoid"],
    "shoulders": ["delts", "deltoid"],
    "delt": ["shoulder", "deltoid"],
    "delts": ["shoulder", "deltoid"],
    "back": ["lats", "row", "pull"],
    "lat": ["back", "lats", "pull"],
    "lats": ["back", "lat", "pull"],
    "chest": ["pec", "push"],
    "pec": ["chest", "push"],
    "glute": ["glutes", "hip"],
    "glutes": ["glute", "hip"],
    "quad": ["quads", "squat"],
    "quads": ["quad", "squat"],
    "hamstring": ["hamstrings", "hinge"],
    "hamstrings": ["hamstring", "hinge"],
    "abs": ["core"],
    "core": ["abs"],
    "press": ["push"],
    "row": ["pull"],
    "squat": ["knee", "lower"],
    "hinge": ["hip", "posterior"],
}

HIGH_VALUE_TOKENS = {
    "beginner",
    "intermediate",
    "advanced",
    "shoulder",
    "shoulders",
    "chest",
    "back",
    "quads",
    "glutes",
    "hamstrings",
    "core",
    "push",
    "pull",
    "squat",
    "hinge",
    "press",
    "row",
}
