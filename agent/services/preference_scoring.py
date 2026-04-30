"""User preference scoring for RAG reranking.

These helpers keep preference learning outside prompts. Retrieval can stay
source-agnostic while still respecting the user's repeated accepts, rejects,
food avoidances, and difficult exercises.
"""

from __future__ import annotations

from typing import Any, Iterable


def learned_preferences_from_context(memory_context: dict[str, Any] | None) -> dict[str, Any]:
    """Extract learned preferences from the four-layer memory context."""

    if not isinstance(memory_context, dict):
        return {}
    structured_profile = memory_context.get("structured_profile")
    if not isinstance(structured_profile, dict):
        return {}
    preferences = structured_profile.get("learned_preferences")
    return preferences if isinstance(preferences, dict) else {}


def score_exercise_preference(exercise: dict[str, Any], preferences: dict[str, Any] | None) -> float:
    """Return a small rerank score based on exercise history."""

    preferences = preferences or {}
    name = str(exercise.get("name", "")).strip()
    name_key = _normalize(name)
    focus_tags = _normalize_many(exercise.get("focus_tags"))
    primary_muscles = _normalize_many(exercise.get("primary_muscles") or exercise.get("target_muscle"))
    contraindications = _normalize_many(exercise.get("contraindications"))

    score = 0.0
    if _matches_any(name_key, preferences.get("liked_exercises")):
        score += 0.35
    if _matches_any(name_key, preferences.get("difficult_exercises")):
        score -= 0.6
    if _matches_any(name_key, preferences.get("avoided_exercises")):
        score -= 1.0

    preferred_focuses = _normalize_many(preferences.get("preferred_focuses"))
    if preferred_focuses and (focus_tags.intersection(preferred_focuses) or _focus_text_matches(focus_tags, preferred_focuses)):
        score += 0.12

    injury_areas = _normalize_many(preferences.get("active_injury_areas"))
    if injury_areas and (primary_muscles.intersection(injury_areas) or contraindications.intersection(injury_areas)):
        score -= 0.45
    return score


def score_food_preference(food: dict[str, Any], preferences: dict[str, Any] | None) -> float:
    """Return a rerank score for food preferences and avoidances."""

    preferences = preferences or {}
    name = str(food.get("name", "")).strip()
    name_key = _normalize(name)
    category = _normalize(str(food.get("category", "")))
    diet_tags = _normalize_many(food.get("diet_tags"))

    if _matches_any(name_key, preferences.get("avoided_foods")):
        return -2.0

    score = 0.0
    if _matches_any(name_key, preferences.get("preferred_foods")):
        score += 0.35
    preferred_food_terms = _normalize_many(preferences.get("preferred_foods"))
    if category and category in preferred_food_terms:
        score += 0.08
    if preferred_food_terms and diet_tags.intersection(preferred_food_terms):
        score += 0.08
    return score


def food_is_avoided(food: dict[str, Any], preferences: dict[str, Any] | None) -> bool:
    """Whether a food should be removed from candidate sets."""

    preferences = preferences or {}
    name_key = _normalize(str(food.get("name", "")))
    return _matches_any(name_key, preferences.get("avoided_foods"))


def _focus_text_matches(focus_tags: set[str], preferred_focuses: set[str]) -> bool:
    for focus in focus_tags:
        for preferred in preferred_focuses:
            if focus and preferred and (focus in preferred or preferred in focus):
                return True
    return False


def _matches_any(name_key: str, values: Any) -> bool:
    if not name_key:
        return False
    for value in _normalize_many(values):
        if value and (value == name_key or value in name_key or name_key in value):
            return True
    return False


def _normalize_many(values: Iterable[Any] | None) -> set[str]:
    if not values:
        return set()
    if isinstance(values, str):
        values = [values]
    return {_normalize(str(value)) for value in values if str(value).strip()}


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")
