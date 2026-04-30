"""Optional Milvus backend for FitnessAgent RAG.

Milvus stores the same document payload used by the local JSON index, but the
retriever can query it as a real vector database. The embedding is still the
current deterministic local embedding, which keeps tests offline and lets us
swap in a real embedding model later without changing planner code.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from agent.config import load_settings
from agent.rag.vector_store import EMBEDDING_DIMENSIONS, document_text, embed_text, stable_hash


class MilvusUnavailable(RuntimeError):
    """Raised when Milvus is requested but not available."""


def milvus_enabled() -> bool:
    settings = load_settings()
    return settings.rag_backend in {"auto", "milvus"} and settings.has_milvus


def exercise_collection_name() -> str:
    return load_settings().milvus_exercise_collection


def food_collection_name() -> str:
    return load_settings().milvus_food_collection


@lru_cache(maxsize=1)
def milvus_client() -> Any:
    settings = load_settings()
    if not milvus_enabled():
        raise MilvusUnavailable("Milvus is not configured")
    try:
        from pymilvus import MilvusClient
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise MilvusUnavailable("pymilvus is not installed") from exc

    kwargs: dict[str, Any] = {"uri": settings.milvus_uri}
    if settings.milvus_token:
        kwargs["token"] = settings.milvus_token
    return MilvusClient(**kwargs)


def build_milvus_collection(
    documents: list[dict[str, Any]],
    *,
    collection_name: str,
    recreate: bool = True,
) -> bool:
    """Create and fill a Milvus collection. Returns False if unavailable."""

    if not documents or not milvus_enabled():
        return False
    try:
        client = milvus_client()
        if recreate and client.has_collection(collection_name):
            client.drop_collection(collection_name)
        if not client.has_collection(collection_name):
            create_milvus_collection(client, collection_name)
        client.insert(collection_name=collection_name, data=_document_rows(documents))
    except Exception:
        return False
    return True


def ensure_milvus_collection(
    documents: list[dict[str, Any]],
    *,
    collection_name: str,
) -> bool:
    """Create and seed a collection only when it does not already exist."""

    if not documents or not milvus_enabled():
        return False
    try:
        client = milvus_client()
        if client.has_collection(collection_name):
            return True
        create_milvus_collection(client, collection_name)
        client.insert(collection_name=collection_name, data=_document_rows(documents))
    except Exception:
        return False
    return True


def search_milvus_collection(
    query: str,
    *,
    collection_name: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search a Milvus collection and return local-index-compatible results."""

    if not query.strip() or not milvus_enabled():
        return []
    try:
        client = milvus_client()
        if not client.has_collection(collection_name):
            return []
        response = client.search(
            collection_name=collection_name,
            data=[embed_text(query)],
            limit=max(1, limit),
            output_fields=["document_json"],
        )
    except Exception:
        return []
    return _search_results(response)


def create_milvus_collection(client: Any, collection_name: str) -> None:
    """Create the document schema used by FitnessAgent RAG collections."""

    try:
        from pymilvus import DataType
    except Exception as exc:  # pragma: no cover - optional dependency
        raise MilvusUnavailable("pymilvus is not installed") from exc

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSIONS)
    schema.add_field(field_name="document_json", datatype=DataType.VARCHAR, max_length=65_535)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        consistency_level="Strong",
    )


def _document_rows(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        text = document_text(document)
        rows.append(
            {
                "id": _document_id(document, index),
                "vector": embed_text(text),
                "document_json": json.dumps(document, ensure_ascii=False),
            }
        )
    return rows


def _document_id(document: dict[str, Any], fallback: int) -> int:
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    raw = document.get("raw") if isinstance(document.get("raw"), dict) else {}
    key = str(
        metadata.get("id")
        or metadata.get("name")
        or raw.get("id")
        or raw.get("name")
        or document.get("title")
        or fallback
    )
    return stable_hash(key) % 9_000_000_000_000_000_000


def _search_results(response: Any) -> list[dict[str, Any]]:
    hits = response[0] if isinstance(response, list) and response else response
    results: list[dict[str, Any]] = []
    for hit in hits or []:
        document_json = _hit_field(hit, "document_json")
        if not document_json:
            continue
        try:
            document = json.loads(document_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict):
            continue
        results.append(
            {
                "score": _hit_score(hit),
                "document": document,
            }
        )
    return results


def _hit_field(hit: Any, field_name: str) -> Any:
    if isinstance(hit, dict):
        entity = hit.get("entity")
        if isinstance(entity, dict) and field_name in entity:
            return entity[field_name]
        return hit.get(field_name)
    entity = getattr(hit, "entity", None)
    if isinstance(entity, dict):
        return entity.get(field_name)
    if hasattr(hit, "get"):
        try:
            return hit.get(field_name)
        except Exception:
            return None
    return None


def _hit_score(hit: Any) -> float:
    if isinstance(hit, dict):
        return float(hit.get("score") or hit.get("distance") or 0.0)
    for attr in ("score", "distance"):
        value = getattr(hit, attr, None)
        if value is not None:
            return float(value)
    return 0.0
