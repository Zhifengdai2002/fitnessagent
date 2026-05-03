"""Exercise retriever backed by the local vector index."""

from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Iterable

from agent.rag.documents import (
    build_exercise_documents,
    build_food_documents,
    build_knowledge_documents,
    build_primary_exercise_documents,
    build_primary_food_documents,
    build_primary_knowledge_documents,
)
from agent.rag.milvus_store import (
    build_milvus_collection,
    ensure_milvus_collection,
    exercise_collection_name,
    food_collection_name,
    knowledge_collection_name,
    milvus_enabled,
    search_milvus_collection,
)
from agent.rag.vector_store import (
    EXERCISE_INDEX_PATH,
    FOOD_INDEX_PATH,
    KNOWLEDGE_INDEX_PATH,
    INDEX_VERSION,
    build_index,
    embedding_backend_name,
    embedding_dimensions,
    load_index,
    search_index,
)
from agent.services.preference_scoring import food_is_avoided, score_exercise_preference, score_food_preference
from agent.services.redis_store import load_cache_item, redis_cache_ttl_seconds, save_cache_item

LOCAL_FALLBACK_SOURCES = {"local", "local_fallback"}
RAG_SEARCH_CACHE_NAMESPACE = "rag_search"
RAG_SEARCH_CACHE_VERSION = 2


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_many(values: Iterable[str] | None) -> set[str]:
    if not values:
        return set()
    return {_normalize(str(value)) for value in values if str(value).strip()}


@lru_cache(maxsize=1)
def exercise_index() -> dict[str, Any]:
    index = load_index(EXERCISE_INDEX_PATH)
    if _index_is_current(index):
        return index
    return build_index(build_exercise_documents())


@lru_cache(maxsize=1)
def food_index() -> dict[str, Any]:
    index = load_index(FOOD_INDEX_PATH)
    if _index_is_current(index):
        return index
    return build_index(build_food_documents(), FOOD_INDEX_PATH)


@lru_cache(maxsize=1)
def knowledge_index() -> dict[str, Any]:
    index = load_index(KNOWLEDGE_INDEX_PATH)
    if _index_is_current(index):
        return index
    return build_index(build_knowledge_documents(), KNOWLEDGE_INDEX_PATH)


def _index_is_current(index: dict[str, Any]) -> bool:
    return bool(
        index.get("documents")
        and index.get("version") == INDEX_VERSION
        and index.get("embedding") == embedding_backend_name()
        and index.get("dimensions") == embedding_dimensions()
    )


def rebuild_exercise_index() -> dict[str, Any]:
    from agent.rag.documents import (
        load_exercise_documents_source,
        load_local_exercise_fallback_source,
        load_primary_exercise_documents_source,
    )

    load_exercise_documents_source.cache_clear()
    load_primary_exercise_documents_source.cache_clear()
    load_local_exercise_fallback_source.cache_clear()
    exercise_index.cache_clear()
    documents = build_exercise_documents()
    primary_documents = build_primary_exercise_documents()
    build_milvus_collection(primary_documents, collection_name=exercise_collection_name())
    return build_index(documents, EXERCISE_INDEX_PATH)


def rebuild_food_index() -> dict[str, Any]:
    from agent.rag.documents import (
        load_food_documents_source,
        load_local_food_fallback_source,
        load_primary_food_documents_source,
    )

    load_food_documents_source.cache_clear()
    load_primary_food_documents_source.cache_clear()
    load_local_food_fallback_source.cache_clear()
    food_index.cache_clear()
    documents = build_food_documents()
    primary_documents = build_primary_food_documents()
    build_milvus_collection(primary_documents, collection_name=food_collection_name())
    return build_index(documents, FOOD_INDEX_PATH)


def rebuild_knowledge_index() -> dict[str, Any]:
    from agent.rag.documents import load_knowledge_documents_source, load_primary_knowledge_documents_source

    load_knowledge_documents_source.cache_clear()
    load_primary_knowledge_documents_source.cache_clear()
    knowledge_index.cache_clear()
    documents = build_knowledge_documents()
    primary_documents = build_primary_knowledge_documents()
    build_milvus_collection(primary_documents, collection_name=knowledge_collection_name())
    return build_index(documents, KNOWLEDGE_INDEX_PATH)


def retrieve_exercises(
    *,
    query: str,
    focus: str | None = None,
    level: str | None = None,
    exclude: Iterable[str] | None = None,
    excluded_conditions: Iterable[str] | None = None,
    source_exercise: dict[str, Any] | None = None,
    learned_preferences: dict[str, Any] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve and rerank exercise candidates."""

    excluded_names = _normalize_many(exclude)
    excluded_conditions_set = _normalize_many(excluded_conditions)
    requested_level = _normalize(level) if level else ""
    requested_focus = _normalize(focus) if focus else ""
    source_group = _normalize(str(source_exercise.get("replacement_group", ""))) if source_exercise else ""
    source_pattern = _normalize(str(source_exercise.get("movement_pattern") or source_exercise.get("movement_type") or "")) if source_exercise else ""
    source_muscles = _normalize_many((source_exercise or {}).get("primary_muscles") or (source_exercise or {}).get("target_muscle"))

    results = _search_exercise_documents(query, limit=max(limit * 12, 60))
    scored: list[tuple[int, float, int, dict[str, Any]]] = []
    for position, result in enumerate(results):
        document = result.get("document", {})
        raw = dict(document.get("raw") or {})
        metadata = dict(document.get("metadata") or {})
        raw = _normalize_candidate_source(raw or metadata)
        name = str(metadata.get("name") or raw.get("name") or "").strip()
        if not name or _normalize(name) in excluded_names:
            continue
        contraindications = _normalize_many(metadata.get("contraindications") or raw.get("contraindications"))
        if excluded_conditions_set and contraindications.intersection(excluded_conditions_set):
            continue

        score = float(result.get("score") or 0.0)
        score += _metadata_score(
            metadata=metadata,
            requested_focus=requested_focus,
            requested_level=requested_level,
            source_group=source_group,
            source_pattern=source_pattern,
            source_muscles=source_muscles,
        )
        score += score_exercise_preference(raw or metadata, learned_preferences)
        if score <= 0:
            continue
        scored.append((_source_tier(raw), score, position, raw))

    return _dedupe_exercise_candidates(_primary_first(scored, limit=max(limit * 3, limit)), limit=limit)


def retrieve_foods(
    *,
    query: str,
    category: str | None = None,
    diet_tags: Iterable[str] | None = None,
    excluded_allergens: Iterable[str] | None = None,
    min_protein_g: float | None = None,
    max_calories_per_100g: float | None = None,
    learned_preferences: dict[str, Any] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Retrieve and rerank food candidates for nutrition planning."""

    requested_category = _normalize(category) if category else ""
    requested_diet_tags = _normalize_many(diet_tags)
    excluded_allergen_set = _normalize_many(excluded_allergens)
    results = _search_food_documents(query, limit=max(limit * 12, 60))

    scored: list[tuple[int, float, int, dict[str, Any]]] = []
    for position, result in enumerate(results):
        document = result.get("document", {})
        raw = dict(document.get("raw") or {})
        metadata = dict(document.get("metadata") or {})
        raw = _normalize_candidate_source(raw or metadata)
        name = str(metadata.get("name") or raw.get("name") or "").strip()
        if not name:
            continue
        food_category = _normalize(str(metadata.get("category") or raw.get("category") or ""))
        food_diet_tags = _normalize_many(metadata.get("diet_tags") or raw.get("diet_tags"))
        food_allergens = _normalize_many(metadata.get("allergens") or raw.get("allergens"))

        if requested_category and food_category != requested_category:
            continue
        if requested_diet_tags and not requested_diet_tags.issubset(food_diet_tags):
            continue
        if excluded_allergen_set and food_allergens.intersection(excluded_allergen_set):
            continue
        if min_protein_g is not None and float(raw.get("protein_g", metadata.get("protein_g", 0.0)) or 0.0) < min_protein_g:
            continue
        if max_calories_per_100g is not None and float(raw.get("calories_per_100g", metadata.get("calories_per_100g", 0.0)) or 0.0) > max_calories_per_100g:
            continue
        if food_is_avoided(raw or metadata, learned_preferences):
            continue

        score = float(result.get("score") or 0.0)
        if requested_category and food_category == requested_category:
            score += 0.35
        if requested_diet_tags and requested_diet_tags.intersection(food_diet_tags):
            score += 0.2
        if str(raw.get("source") or metadata.get("source")) == "curated_rag":
            score += 0.05
        score += score_food_preference(raw or metadata, learned_preferences)
        scored.append((_source_tier(raw), score, position, raw))

    return _primary_first(scored, limit=limit)


def retrieve_knowledge(
    *,
    query: str,
    topic: str | None = None,
    goal: str | None = None,
    level: str | None = None,
    injury_areas: list[str] | None = None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Retrieve professional training/nutrition/recovery knowledge snippets."""

    requested_topic = _normalize(topic) if topic else ""
    requested_goal = _normalize(goal) if goal else ""
    requested_level = _normalize(level) if level else ""
    injury_set = {_normalize(a) for a in (injury_areas or []) if a}
    results = _search_knowledge_documents(query, limit=max(limit * 10, 40))
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for position, result in enumerate(results):
        document = result.get("document", {})
        raw = dict(document.get("raw") or {})
        metadata = dict(document.get("metadata") or {})
        title = str(document.get("title") or raw.get("title") or "").strip()
        text = str(raw.get("text") or document.get("text") or "").strip()
        if not title or not text:
            continue
        score = float(result.get("score") or 0.0)
        score += _knowledge_metadata_score(
            metadata=metadata,
            requested_topic=requested_topic,
            requested_goal=requested_goal,
            requested_level=requested_level,
            injury_set=injury_set,
        )
        scored.append(
            (
                score,
                position,
                {
                    "title": title,
                    "text": text,
                    "source": str(metadata.get("source") or raw.get("source") or "").strip(),
                    "source_url": str(metadata.get("source_url") or raw.get("source_url") or "").strip(),
                    "doc_type": str(metadata.get("doc_type") or raw.get("doc_type") or "").strip(),
                    "section": str(metadata.get("section") or raw.get("section") or "").strip(),
                    "topic": str(metadata.get("topic") or raw.get("topic") or "").strip(),
                    "evidence_type": str(metadata.get("evidence_type") or raw.get("evidence_type") or "").strip(),
                    "metadata": metadata,
                },
            )
        )

    scored.sort(key=lambda item: (-item[0], item[1]))
    return _dedupe_knowledge_results([item for _, _, item in scored], limit=limit)


def _metadata_score(
    *,
    metadata: dict[str, Any],
    requested_focus: str,
    requested_level: str,
    source_group: str,
    source_pattern: str,
    source_muscles: set[str],
) -> float:
    score = 0.0
    focus_tags = _normalize_many(metadata.get("focus_tags"))
    primary_muscles = _normalize_many(metadata.get("primary_muscles") or metadata.get("target_muscle"))
    secondary_muscles = _normalize_many(metadata.get("secondary_muscles"))
    group = _normalize(str(metadata.get("replacement_group", "")))
    pattern = _normalize(str(metadata.get("movement_pattern", "")))
    difficulty = _normalize(str(metadata.get("difficulty", "")))

    if requested_focus and requested_focus in focus_tags:
        score += 0.35
    if source_group and group == source_group:
        score += 0.65
    if source_pattern and pattern == source_pattern:
        score += 0.45
    if source_muscles and source_muscles.intersection(primary_muscles):
        score += 0.35
    if source_muscles and source_muscles.intersection(secondary_muscles):
        score += 0.15
    score += _level_score(requested_level, difficulty)
    return score


def _knowledge_metadata_score(
    *,
    metadata: dict[str, Any],
    requested_topic: str,
    requested_goal: str,
    requested_level: str,
    injury_set: set[str] | None = None,
) -> float:
    score = 0.0
    topics = _normalize_values(metadata.get("topic"))
    goals = _normalize_values(metadata.get("goal"))
    levels = _normalize_values(metadata.get("level"))
    evidence_type = _normalize(str(metadata.get("evidence_type") or ""))
    source = _normalize(str(metadata.get("source") or ""))

    if requested_topic and requested_topic in topics:
        score += 0.35
    if requested_goal and requested_goal in goals:
        score += 0.2
    if requested_level and requested_level in levels:
        score += 0.12
    if evidence_type in {"government_guideline", "meta_analysis", "systematic_review"}:
        score += 0.12
    elif evidence_type in {"clinical_guideline", "professional_reference"}:
        score += 0.08
    if source in {"exrx", "pmc", "pubmed", "dietary_guidelines_for_americans"}:
        score += 0.04

    if injury_set:
        muscle_groups = _normalize_values(metadata.get("muscle_group") or metadata.get("tags"))
        if muscle_groups & injury_set:
            score -= 0.3

    return score


def _search_exercise_documents(query: str, *, limit: int) -> list[dict[str, Any]]:
    if milvus_enabled():
        milvus_cache_key = _rag_search_cache_key(
            kind="exercise",
            backend=f"milvus:{exercise_collection_name()}",
            query=query,
            limit=limit,
        )
        cached_results = _load_cached_rag_search(milvus_cache_key)
        if cached_results is not None:
            return cached_results

        documents = build_primary_exercise_documents()
        ensure_milvus_collection(documents, collection_name=exercise_collection_name())
        milvus_results = search_milvus_collection(
            query,
            collection_name=exercise_collection_name(),
            limit=limit,
        )
        if milvus_results:
            _save_cached_rag_search(milvus_cache_key, milvus_results)
            return milvus_results

    local_cache_key = _rag_search_cache_key(kind="exercise", backend="local", query=query, limit=limit)
    cached_results = _load_cached_rag_search(local_cache_key)
    if cached_results is not None:
        return cached_results
    local_results = search_index(query, index=exercise_index(), limit=limit)
    _save_cached_rag_search(local_cache_key, local_results)
    return local_results


def _search_food_documents(query: str, *, limit: int) -> list[dict[str, Any]]:
    if milvus_enabled():
        milvus_cache_key = _rag_search_cache_key(
            kind="food",
            backend=f"milvus:{food_collection_name()}",
            query=query,
            limit=limit,
        )
        cached_results = _load_cached_rag_search(milvus_cache_key)
        if cached_results is not None:
            return cached_results

        documents = build_primary_food_documents()
        ensure_milvus_collection(documents, collection_name=food_collection_name())
        milvus_results = search_milvus_collection(
            query,
            collection_name=food_collection_name(),
            limit=limit,
        )
        if milvus_results:
            _save_cached_rag_search(milvus_cache_key, milvus_results)
            return milvus_results

    local_cache_key = _rag_search_cache_key(kind="food", backend="local", query=query, limit=limit)
    cached_results = _load_cached_rag_search(local_cache_key)
    if cached_results is not None:
        return cached_results
    local_results = search_index(query, index=food_index(), limit=limit)
    _save_cached_rag_search(local_cache_key, local_results)
    return local_results


def _search_knowledge_documents(query: str, *, limit: int) -> list[dict[str, Any]]:
    if milvus_enabled():
        milvus_cache_key = _rag_search_cache_key(
            kind="knowledge",
            backend=f"milvus:{knowledge_collection_name()}",
            query=query,
            limit=limit,
        )
        cached_results = _load_cached_rag_search(milvus_cache_key)
        if cached_results is not None:
            return cached_results

        documents = build_primary_knowledge_documents()
        ensure_milvus_collection(documents, collection_name=knowledge_collection_name())
        milvus_results = search_milvus_collection(
            query,
            collection_name=knowledge_collection_name(),
            limit=limit,
        )
        if milvus_results:
            _save_cached_rag_search(milvus_cache_key, milvus_results)
            return milvus_results

    local_cache_key = _rag_search_cache_key(kind="knowledge", backend="local", query=query, limit=limit)
    cached_results = _load_cached_rag_search(local_cache_key)
    if cached_results is not None:
        return cached_results
    local_results = search_index(query, index=knowledge_index(), limit=limit)
    _save_cached_rag_search(local_cache_key, local_results)
    return local_results


def _rag_search_cache_key(*, kind: str, backend: str, query: str, limit: int) -> str:
    payload = {
        "version": RAG_SEARCH_CACHE_VERSION,
        "index_version": INDEX_VERSION,
        "embedding": embedding_backend_name(),
        "dimensions": embedding_dimensions(),
        "kind": kind,
        "backend": backend,
        "query": " ".join(str(query or "").strip().lower().split()),
        "limit": int(limit),
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_cached_rag_search(cache_key: str) -> list[dict[str, Any]] | None:
    cached_payload = load_cache_item(RAG_SEARCH_CACHE_NAMESPACE, cache_key)
    if not isinstance(cached_payload, dict):
        return None
    results = cached_payload.get("results")
    return results if isinstance(results, list) else None


def _save_cached_rag_search(cache_key: str, results: list[dict[str, Any]]) -> None:
    if not results:
        return
    save_cache_item(
        RAG_SEARCH_CACHE_NAMESPACE,
        cache_key,
        {"results": results},
        ttl_seconds=redis_cache_ttl_seconds(),
    )


def _normalize_candidate_source(candidate: dict[str, Any]) -> dict[str, Any]:
    source = str(candidate.get("source") or "local").strip()
    if _normalize(source) in LOCAL_FALLBACK_SOURCES:
        return {**candidate, "source": "local_fallback"}
    return candidate


def _source_tier(candidate: dict[str, Any]) -> int:
    source = _normalize(str(candidate.get("source") or "local"))
    return 1 if source in LOCAL_FALLBACK_SOURCES else 0


def _primary_first(
    scored: list[tuple[int, float, int, dict[str, Any]]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    scored.sort(key=lambda item: (item[0], -item[1], item[2]))
    primary = [candidate for tier, _, _, candidate in scored if tier == 0]
    fallback = [candidate for tier, _, _, candidate in scored if tier > 0]
    if len(primary) >= limit:
        return primary[:limit]
    return [*primary, *fallback[: max(limit - len(primary), 0)]]


def _dedupe_exercise_candidates(candidates: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for candidate in candidates:
        name = str(candidate.get("name", "")).strip()
        key = _canonical_exercise_key(name)
        if not name or not key or key in seen_keys:
            if key:
                _replace_duplicate_with_cleaner_candidate(selected, candidate)
            continue
        duplicate = next(
            (item for item in selected if _looks_like_duplicate_exercise_name(name, str(item.get("name", "")))),
            None,
        )
        if duplicate:
            _replace_duplicate_with_cleaner_candidate(selected, candidate)
            continue
        seen_keys.add(key)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _dedupe_knowledge_results(candidates: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for candidate in candidates:
        key = "|".join(
            [
                _normalize(str(candidate.get("source") or "")),
                _normalize(str(candidate.get("title") or "")),
                _normalize(str(candidate.get("section") or "")),
            ]
        )
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _normalize_values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {_normalize(value)} if value.strip() else set()
    if isinstance(value, Iterable):
        return _normalize_many(str(item) for item in value)
    return set()


def _canonical_exercise_key(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    tokens = [
        token
        for token in normalized.split()
        if token not in {"exercise", "exercises", "workout", "movement", "demo", "tutorial"}
    ]
    return " ".join(tokens)


def _looks_like_duplicate_exercise_name(left: str, right: str) -> bool:
    left_key = _canonical_exercise_key(left)
    right_key = _canonical_exercise_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    return SequenceMatcher(None, left_key, right_key).ratio() >= 0.92


def _replace_duplicate_with_cleaner_candidate(selected: list[dict[str, Any]], candidate: dict[str, Any]) -> None:
    candidate_name = str(candidate.get("name", "")).strip()
    for index, existing in enumerate(selected):
        if not _looks_like_duplicate_exercise_name(candidate_name, str(existing.get("name", ""))):
            continue
        if _exercise_identity_quality(candidate) > _exercise_identity_quality(existing):
            selected[index] = candidate
        return


def _exercise_identity_quality(candidate: dict[str, Any]) -> float:
    name = str(candidate.get("name", "")).strip()
    normalized = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    tokens = normalized.split()
    source = _normalize(str(candidate.get("source") or ""))
    score = 0.0
    if source == "curated_rag":
        score += 0.4
    elif source == "wger":
        score += 0.25
    if not any(token in {"exercise", "exercises", "workout", "movement", "demo", "tutorial"} for token in tokens):
        score += 0.5
    score -= len(tokens) * 0.01
    return score


def _level_score(requested: str, candidate: str) -> float:
    if not requested or not candidate:
        return 0.0
    ranks = {"beginner": 0, "intermediate": 1, "advanced": 2}
    requested_rank = ranks.get(requested)
    candidate_rank = ranks.get(candidate)
    if requested_rank is None or candidate_rank is None:
        return 0.0
    if requested_rank == candidate_rank:
        return 0.25
    if requested == "beginner":
        return 0.08 if candidate == "intermediate" else -0.35
    if requested == "advanced":
        return 0.16 if candidate == "intermediate" else 0.05
    return 0.1 if abs(requested_rank - candidate_rank) == 1 else -0.2
