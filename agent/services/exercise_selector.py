"""Exercise candidate retrieval and focus-mapping utilities.

Used by the planner node to build exercise candidate pools for each
training session blueprint. Separated from planner.py to keep that
file focused on orchestration.
"""

from __future__ import annotations

import re
from typing import Any

from agent.rag.retriever import retrieve_exercises
from agent.services.planner_constants import FOCUS_ALIASES, FOCUS_LIBRARY
from agent.tools import find_exercises


# ---------------------------------------------------------------------------
# Focus helpers
# ---------------------------------------------------------------------------

def focus_key_from_value(value: str) -> str:
    normalized = value.strip().lower().replace("_", " ")
    if normalized in FOCUS_ALIASES:
        return FOCUS_ALIASES[normalized]
    for focus_key, config in FOCUS_LIBRARY.items():
        if normalized == focus_key.replace("_", " "):
            return focus_key
        if normalized == config["label"].lower():
            return focus_key
    return "functional_conditioning"


def focus_label(focus_key: str) -> str:
    return FOCUS_LIBRARY.get(focus_key, FOCUS_LIBRARY["functional_conditioning"])["label"]


def focus_to_targets(focus: str) -> tuple[list[str], str | None]:
    key = focus_key_from_value(focus)
    config = FOCUS_LIBRARY.get(key, FOCUS_LIBRARY["functional_conditioning"])
    return list(config["target_muscles"]), config["movement_type"]


def recommended_program_tags(
    equipment_access: list[str],
    excluded_conditions: list[str],
) -> list[str]:
    tags: list[str] = []
    if excluded_conditions:
        tags.append("low_impact_program")
    return tags


# ---------------------------------------------------------------------------
# Candidate retrieval
# ---------------------------------------------------------------------------

def retrieve_plan_exercise_candidates(
    *,
    focus_key: str,
    target_muscles: list[str],
    movement_type: str | None,
    fitness_level: str,
    training_goal: str,
    equipment_access: list[str],
    excluded_conditions: list[str],
    excluded_exercises: list[str],
    limit: int,
    learned_preferences: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    query = _build_query(
        focus_key=focus_key,
        target_muscles=target_muscles,
        movement_type=movement_type,
        fitness_level=fitness_level,
        training_goal=training_goal,
    )
    candidates = retrieve_exercises(
        query=query,
        focus=focus_key,
        level=fitness_level,
        exclude=excluded_exercises,
        excluded_conditions=excluded_conditions,
        learned_preferences=learned_preferences,
        limit=limit,
    )
    candidates = filter_candidates(candidates, excluded_exercises)
    if len(candidates) >= limit:
        return candidates[:limit]

    fallback = find_exercises(
        target_muscles=target_muscles,
        movement_type=movement_type,
        difficulty=fitness_level,
        available_equipment=equipment_access,
        training_goal=training_goal,
        excluded_conditions=excluded_conditions,
        recommended_for=recommended_program_tags(equipment_access, excluded_conditions),
        focus_tags=[focus_key],
        limit=max(limit * 2, limit),
    )
    return merge_candidates(candidates, fallback, excluded_exercises, limit)


def filter_candidates(
    candidates: list[dict[str, Any]],
    excluded_exercises: list[str],
) -> list[dict[str, Any]]:
    excluded = {normalize_exercise_key(v) for v in excluded_exercises}
    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ex in candidates:
        name_key = normalize_exercise_key(str(ex.get("name", "")))
        id_key = normalize_exercise_key(str(ex.get("id", "")))
        if not name_key or name_key in seen:
            continue
        if name_key in excluded or id_key in excluded:
            continue
        seen.add(name_key)
        filtered.append(ex)
    return filtered


def merge_candidates(
    primary: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    excluded_exercises: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    merged = filter_candidates(primary, excluded_exercises)
    seen = {normalize_exercise_key(str(ex.get("name", ""))) for ex in merged}
    excluded = {normalize_exercise_key(v) for v in excluded_exercises}
    for ex in fallback:
        if len(merged) >= limit:
            break
        name_key = normalize_exercise_key(str(ex.get("name", "")))
        id_key = normalize_exercise_key(str(ex.get("id", "")))
        if (
            not name_key
            or name_key in seen
            or name_key in excluded
            or id_key in excluded
            or any(_similar_exercise(name_key, s) for s in seen)
        ):
            continue
        seen.add(name_key)
        merged.append(ex)
    return merged[:limit]


def normalize_exercise_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    tokens = [
        t for t in normalized.split()
        if t not in {"exercise", "exercises", "workout", "movement", "demo", "tutorial"}
    ]
    return "_".join(tokens)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_query(
    *,
    focus_key: str,
    target_muscles: list[str],
    movement_type: str | None,
    fitness_level: str,
    training_goal: str,
) -> str:
    pieces = [
        f"Find exercises for a {focus_label(focus_key)} workout.",
        f"Focus tag: {focus_key}.",
        f"Fitness level: {fitness_level}.",
        f"Training goal: {training_goal}.",
        f"Target muscles: {', '.join(target_muscles)}.",
    ]
    if movement_type:
        pieces.append(f"Movement type: {movement_type}.")
    pieces.append("Prefer appropriate alternatives with matching muscles, movement pattern, and difficulty.")
    return "\n".join(pieces)


def _similar_exercise(left_key: str, right_key: str) -> bool:
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    left_tokens = set(left_key.split("_"))
    right_tokens = set(right_key.split("_"))
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))
    return overlap >= 0.8
