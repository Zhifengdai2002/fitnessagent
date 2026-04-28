"""Exercise retriever backed by the local vector index."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

from agent.rag.documents import build_exercise_documents
from agent.rag.vector_store import build_index, load_index, search_index


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_many(values: Iterable[str] | None) -> set[str]:
    if not values:
        return set()
    return {_normalize(str(value)) for value in values if str(value).strip()}


@lru_cache(maxsize=1)
def exercise_index() -> dict[str, Any]:
    index = load_index()
    if index.get("documents"):
        return index
    return build_index(build_exercise_documents())


def rebuild_exercise_index() -> dict[str, Any]:
    from agent.rag.documents import load_exercise_documents_source

    load_exercise_documents_source.cache_clear()
    exercise_index.cache_clear()
    return build_index(build_exercise_documents())


def retrieve_exercises(
    *,
    query: str,
    focus: str | None = None,
    level: str | None = None,
    exclude: Iterable[str] | None = None,
    excluded_conditions: Iterable[str] | None = None,
    source_exercise: dict[str, Any] | None = None,
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

    results = search_index(query, index=exercise_index(), limit=max(limit * 6, 20))
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for position, result in enumerate(results):
        document = result.get("document", {})
        raw = dict(document.get("raw") or {})
        metadata = dict(document.get("metadata") or {})
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
        if score <= 0:
            continue
        scored.append((score, position, raw or metadata))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [exercise for _, _, exercise in scored[:limit]]


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
