"""Structured retrieval helpers for local fitness knowledge."""

from __future__ import annotations

from typing import Any, Iterable

from agent.rag.retriever import retrieve_exercises
from agent.services.video_cache import get_cached_video_resource
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
    excluded_conditions: Iterable[str] | None = None,
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
        excluded_conditions=excluded_conditions,
        source_exercise=source,
        limit=max(limit * 4, 12),
    )
    rag_matches = rerank_replacement_candidates(
        candidates=rag_matches,
        source=source,
        focus=focus,
        level=level,
        excluded_conditions=excluded_conditions,
        limit=limit,
    )
    if rag_matches:
        return rag_matches

    return _fallback_search_similar_exercises(
        source=source,
        exercise_name=exercise_name,
        focus=focus,
        level=level,
        exclude=exclude,
        excluded_conditions=excluded_conditions,
        limit=limit,
    )


def rerank_replacement_candidates(
    *,
    candidates: list[dict[str, Any]],
    source: dict[str, Any],
    focus: str | None,
    level: str | None,
    excluded_conditions: Iterable[str] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Rerank replacement candidates with fitness-specific constraints."""

    excluded_condition_set = _normalize_many(excluded_conditions)
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, candidate in enumerate(candidates):
        score = replacement_candidate_score(
            candidate=candidate,
            source=source,
            focus=focus,
            level=level,
            excluded_conditions=excluded_condition_set,
        )
        if score <= 0:
            continue
        scored.append((score, index, candidate))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, _, candidate in scored[:limit]]


def replacement_candidate_score(
    *,
    candidate: dict[str, Any],
    source: dict[str, Any],
    focus: str | None,
    level: str | None,
    excluded_conditions: set[str] | None = None,
) -> int:
    """Score how suitable one exercise is as a substitute."""

    requested_focus = _normalize(focus) if focus else ""
    candidate_focus = _normalize_many(candidate.get("focus_tags", []))
    if requested_focus and requested_focus not in candidate_focus:
        return -100

    contraindications = _normalize_many(candidate.get("contraindications", []))
    if excluded_conditions and contraindications.intersection(excluded_conditions):
        return -100

    score = 0
    if requested_focus and requested_focus in candidate_focus:
        score += 50

    source_group = _canonical_replacement_group(source)
    candidate_group = _canonical_replacement_group(candidate)
    if source_group and candidate_group == source_group:
        score += 45

    source_pattern = _normalize(str(source.get("movement_pattern") or source.get("movement_type", "")))
    candidate_pattern = _normalize(str(candidate.get("movement_pattern") or candidate.get("movement_type", "")))
    if source_pattern and candidate_pattern == source_pattern:
        score += 25

    source_primary = _normalize_many(source.get("primary_muscles") or source.get("target_muscle", []))
    candidate_primary = _normalize_many(candidate.get("primary_muscles") or candidate.get("target_muscle", []))
    candidate_secondary = _normalize_many(candidate.get("secondary_muscles", []))
    if source_primary and source_primary.intersection(candidate_primary):
        score += 20
    elif source_primary and source_primary.intersection(candidate_secondary):
        score += 8

    score += _level_score(_normalize(level) if level else "", _normalize(str(candidate.get("difficulty", ""))))

    if _has_video_signal(candidate):
        score += 10
    return score


def _has_video_signal(candidate: dict[str, Any]) -> bool:
    """Reward known YouTube API video coverage without doing network lookup."""

    name = str(candidate.get("name", "")).strip()
    if not name:
        return False
    try:
        cached = get_cached_video_resource(name)
    except Exception:
        return False
    source = str((cached or {}).get("source") or "").strip().lower()
    provider = str((cached or {}).get("provider") or "").strip().lower()
    url = str((cached or {}).get("url") or "").strip().lower()
    return source == "youtube_api" and (not provider or provider == "youtube") and (
        "youtube.com/watch" in url or "youtu.be/" in url
    )


def _canonical_replacement_group(exercise: dict[str, Any]) -> str:
    group = _normalize(str(exercise.get("replacement_group", "")))
    pattern = _normalize(str(exercise.get("movement_pattern") or exercise.get("movement_type", "")))
    aliases = {
        "lat_pull": "vertical_pull",
        "pull_up": "vertical_pull",
        "pulldown": "vertical_pull",
        "row_pattern": "row",
        "horizontal_pull": "row",
        "shoulder_abduction": "shoulder_raise",
        "lateral_raise": "shoulder_raise",
        "rear_delt_upper_back": "rear_delt_upper_back",
        "squat": "squat_pattern",
        "hinge_pattern": "hinge_glute",
        "hip_hinge": "hinge_glute",
    }
    return aliases.get(group) or aliases.get(pattern) or group or pattern


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
    excluded_conditions: Iterable[str] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Original metadata-only search kept as a fallback."""

    excluded_names = {_normalize(exercise_name), *_normalize_many(exclude)}
    candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(load_all_exercise_db()):
        candidate_name = str(candidate.get("name", ""))
        if _normalize(candidate_name) in excluded_names:
            continue
        candidates.append(candidate)

    return rerank_replacement_candidates(
        candidates=candidates,
        source=source,
        focus=focus,
        level=level,
        excluded_conditions=excluded_conditions,
        limit=limit,
    )


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
