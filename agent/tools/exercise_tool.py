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
    """Return safe focus matches, ranking level and goal as preferences."""

    target_muscle_set = _normalize_many(target_muscles)
    focus_tag_set = _normalize_many(focus_tags)
    excluded_condition_set = _normalize_many(excluded_conditions)
    recommended_for_set = _normalize_many(recommended_for)
    movement_type_normalized = _normalize(movement_type) if movement_type else None
    difficulty_normalized = _normalize(difficulty) if difficulty else None
    training_goal_normalized = _normalize(training_goal) if training_goal else None

    scored_matches: list[tuple[int, int, dict[str, Any]]] = []
    for index, exercise in enumerate(load_exercise_db()):
        exercise_muscles = _normalize_many(exercise.get("target_muscle", []))
        exercise_focus_tags = _normalize_many(exercise.get("focus_tags", []))
        contraindications = _normalize_many(exercise.get("contraindications", []))
        recommendation_tags = _normalize_many(exercise.get("recommended_for", []))
        goal_tags = _normalize_many(exercise.get("training_goal_tags", []))

        if target_muscle_set and not target_muscle_set.intersection(exercise_muscles):
            continue
        if focus_tag_set and exercise_focus_tags and not focus_tag_set.intersection(exercise_focus_tags):
            continue
        if movement_type_normalized and _normalize(exercise.get("movement_type", "")) != movement_type_normalized:
            continue
        if excluded_condition_set and excluded_condition_set.intersection(contraindications):
            continue

        score = _preference_score(
            exercise_difficulty=_normalize(exercise.get("difficulty", "")),
            requested_difficulty=difficulty_normalized,
            goal_tags=goal_tags,
            requested_goal=training_goal_normalized,
            recommendation_tags=recommendation_tags,
            requested_recommendations=recommended_for_set,
        )
        scored_matches.append((score, index, exercise))

    scored_matches.sort(key=lambda item: (-item[0], item[1]))
    return [exercise for _, _, exercise in scored_matches[:limit]]


def _preference_score(
    *,
    exercise_difficulty: str,
    requested_difficulty: str | None,
    goal_tags: set[str],
    requested_goal: str | None,
    recommendation_tags: set[str],
    requested_recommendations: set[str],
) -> int:
    score = 0
    if requested_difficulty:
        score += _difficulty_preference_score(requested_difficulty, exercise_difficulty)
    if requested_goal and requested_goal in goal_tags:
        score += 20
    if requested_recommendations and requested_recommendations.intersection(recommendation_tags):
        score += 5
    return score


def _difficulty_preference_score(requested: str, exercise: str) -> int:
    if not exercise:
        return 0
    difficulty_rank = {"beginner": 0, "intermediate": 1, "advanced": 2}
    requested_rank = difficulty_rank.get(requested)
    exercise_rank = difficulty_rank.get(exercise)
    if requested_rank is None or exercise_rank is None:
        return 0
    if requested_rank == exercise_rank:
        return 30
    distance = abs(requested_rank - exercise_rank)
    if requested == "beginner":
        return 8 if exercise == "intermediate" else -20
    if requested == "advanced":
        return 20 if exercise == "intermediate" else 5
    return 15 if distance == 1 else -10


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
