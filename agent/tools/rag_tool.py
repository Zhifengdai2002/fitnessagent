"""Structured retrieval helpers for local fitness knowledge."""

from __future__ import annotations

from typing import Any, Iterable

from agent.rag.retriever import retrieve_exercises
from agent.tools.exercise_tool import get_exercise_by_name, load_all_exercise_db


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_many(values: Iterable[str] | None) -> set[str]:
    if not values:
        return set()
    return {_normalize(value) for value in values if str(value).strip()}


def search_similar_exercises(
    *,
    exercise_name: str,
    focus: str | None = None,
    level: str | None = None,
    exclude: Iterable[str] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Find same-type replacement exercises from the local RAG index."""

    source = get_exercise_by_name(exercise_name)
    if not source:
        return []

    excluded_names = {_normalize(exercise_name), *_normalize_many(exclude)}
    query = build_exercise_retrieval_query(source, focus=focus, level=level)
    rag_matches = retrieve_exercises(
        query=query,
        focus=focus,
        level=level,
        exclude=excluded_names,
        source_exercise=source,
        limit=limit,
    )
    if rag_matches:
        return rag_matches[:limit]

    return _fallback_search_similar_exercises(
        source=source,
        exercise_name=exercise_name,
        focus=focus,
        level=level,
        exclude=exclude,
        limit=limit,
    )


def build_exercise_retrieval_query(
    source: dict[str, Any],
    *,
    focus: str | None = None,
    level: str | None = None,
) -> str:
    """Build a natural-language retrieval query for exercise substitution."""

    parts = [
        f"Find a substitute exercise for {source.get('name', '')}.",
        f"Focus: {focus}." if focus else "",
        f"Level: {level}." if level else "",
        f"Replacement group: {source.get('replacement_group', '')}.",
        f"Movement pattern: {source.get('movement_pattern') or source.get('movement_type') or ''}.",
        f"Primary muscles: {', '.join(source.get('primary_muscles') or source.get('target_muscle') or [])}.",
        f"Secondary muscles: {', '.join(source.get('secondary_muscles') or [])}.",
        f"Equipment: {', '.join(source.get('equipment') or [])}.",
        f"Notes: {source.get('notes', '')}.",
    ]
    return "\n".join(part for part in parts if part and not part.endswith(": ."))


def _fallback_search_similar_exercises(
    *,
    source: dict[str, Any],
    exercise_name: str,
    focus: str | None = None,
    level: str | None = None,
    exclude: Iterable[str] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Original metadata-only search kept as a fallback."""

    excluded_names = {_normalize(exercise_name), *_normalize_many(exclude)}
    source_focus = _normalize_many(source.get("focus_tags", []))
    requested_focus = {_normalize(focus)} if focus else set()
    source_muscles = _normalize_many(source.get("primary_muscles") or source.get("target_muscle", []))
    source_secondary = _normalize_many(source.get("secondary_muscles", []))
    source_group = _normalize(str(source.get("replacement_group", "")))
    source_pattern = _normalize(str(source.get("movement_pattern") or source.get("movement_type", "")))
    requested_level = _normalize(level) if level else ""

    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, candidate in enumerate(load_all_exercise_db()):
        candidate_name = str(candidate.get("name", ""))
        if _normalize(candidate_name) in excluded_names:
            continue

        candidate_focus = _normalize_many(candidate.get("focus_tags", []))
        if requested_focus and candidate_focus and not requested_focus.intersection(candidate_focus):
            continue

        score = 0
        candidate_group = _normalize(str(candidate.get("replacement_group", "")))
        candidate_pattern = _normalize(str(candidate.get("movement_pattern") or candidate.get("movement_type", "")))
        candidate_muscles = _normalize_many(candidate.get("primary_muscles") or candidate.get("target_muscle", []))
        candidate_secondary = _normalize_many(candidate.get("secondary_muscles", []))

        if source_group and candidate_group == source_group:
            score += 60
        if source_pattern and candidate_pattern == source_pattern:
            score += 35
        if source_focus and source_focus.intersection(candidate_focus):
            score += 25
        if source_muscles and source_muscles.intersection(candidate_muscles):
            score += 20
        if source_muscles and source_muscles.intersection(candidate_secondary):
            score += 8
        if source_secondary and source_secondary.intersection(candidate_muscles | candidate_secondary):
            score += 5
        score += _level_score(requested_level, _normalize(str(candidate.get("difficulty", ""))))

        if score <= 0:
            continue
        scored.append((score, index, candidate))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, _, candidate in scored[:limit]]


def _level_score(requested: str, candidate: str) -> int:
    if not requested or not candidate:
        return 0
    ranks = {"beginner": 0, "intermediate": 1, "advanced": 2}
    requested_rank = ranks.get(requested)
    candidate_rank = ranks.get(candidate)
    if requested_rank is None or candidate_rank is None:
        return 0
    if requested_rank == candidate_rank:
        return 18
    if requested == "beginner":
        return 6 if candidate == "intermediate" else -18
    if requested == "advanced":
        return 12 if candidate == "intermediate" else 3
    return 8 if abs(requested_rank - candidate_rank) == 1 else -8
