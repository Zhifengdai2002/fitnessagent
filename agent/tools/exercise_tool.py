"""Exercise lookup helpers backed by the local exercise database."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "exercise_db.json"


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_many(values: Iterable[str] | None) -> set[str]:
    if not values:
        return set()
    return {_normalize(value) for value in values}


@lru_cache(maxsize=1)
def load_exercise_db() -> list[dict[str, Any]]:
    """Load the local exercise database once per process."""

    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_exercise_by_id(exercise_id: str) -> dict[str, Any] | None:
    """Return one exercise by id."""

    target_id = _normalize(exercise_id)
    for exercise in load_exercise_db():
        if _normalize(exercise["id"]) == target_id:
            return exercise
    return None


def get_exercise_by_name(name: str) -> dict[str, Any] | None:
    """Return one exercise by display name."""

    target_name = _normalize(name)
    for exercise in load_exercise_db():
        if _normalize(exercise["name"]) == target_name:
            return exercise
    return None


def find_exercises(
    *,
    target_muscles: Iterable[str] | None = None,
    focus_tags: Iterable[str] | None = None,
    movement_type: str | None = None,
    difficulty: str | None = None,
    available_equipment: Iterable[str] | None = None,
    training_goal: str | None = None,
    excluded_conditions: Iterable[str] | None = None,
    recommended_for: Iterable[str] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Filter exercises against user goals, equipment, and safety constraints."""

    target_muscle_set = _normalize_many(target_muscles)
    focus_tag_set = _normalize_many(focus_tags)
    equipment_set = _normalize_many(available_equipment)
    excluded_condition_set = _normalize_many(excluded_conditions)
    recommended_for_set = _normalize_many(recommended_for)
    movement_type_normalized = _normalize(movement_type) if movement_type else None
    difficulty_normalized = _normalize(difficulty) if difficulty else None
    training_goal_normalized = _normalize(training_goal) if training_goal else None

    matches: list[dict[str, Any]] = []
    for exercise in load_exercise_db():
        exercise_muscles = _normalize_many(exercise.get("target_muscle", []))
        exercise_focus_tags = _normalize_many(exercise.get("focus_tags", []))
        exercise_equipment = _normalize_many(exercise.get("equipment", []))
        contraindications = _normalize_many(exercise.get("contraindications", []))
        recommendation_tags = _normalize_many(exercise.get("recommended_for", []))
        goal_tags = _normalize_many(exercise.get("training_goal_tags", []))

        if target_muscle_set and not target_muscle_set.intersection(exercise_muscles):
            continue
        if focus_tag_set and exercise_focus_tags and not focus_tag_set.intersection(exercise_focus_tags):
            continue
        if movement_type_normalized and _normalize(exercise.get("movement_type", "")) != movement_type_normalized:
            continue
        if difficulty_normalized and _normalize(exercise.get("difficulty", "")) != difficulty_normalized:
            continue
        if training_goal_normalized and training_goal_normalized not in goal_tags:
            continue
        if recommended_for_set and not recommended_for_set.intersection(recommendation_tags):
            continue
        if excluded_condition_set and excluded_condition_set.intersection(contraindications):
            continue

        # Allow bodyweight movements by default. For other movements, require
        # at least one equipment match when the caller provides an equipment list.
        if equipment_set and "bodyweight" not in exercise_equipment:
            if not equipment_set.intersection(exercise_equipment):
                continue

        matches.append(exercise)
        if len(matches) >= limit:
            break

    return matches


def build_video_resources(exercise_names: Iterable[str]) -> list[dict[str, str]]:
    """Return lightweight video resource objects for planned exercises."""

    resources: list[dict[str, str]] = []
    for name in exercise_names:
        exercise = get_exercise_by_name(name)
        if not exercise:
            continue
        resources.append(
            {
                "exercise_name": exercise["name"],
                "title": f"{exercise['name']} tutorial",
                "url": exercise["youtube_url"],
                "source": "youtube",
            }
        )
    return resources
