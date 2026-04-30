"""Build retrievable documents from local fitness knowledge sources."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

EXERCISE_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "exercise_db.json"
FOOD_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "food_db.json"
EXERCISE_RAG_SEED_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge" / "exercise_rag_seed.json"
FOOD_RAG_SEED_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge" / "food_rag_seed.json"
WGER_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "external" / "wger_exercises.json"
LOCAL_EXERCISE_FALLBACK_LIMIT = 24
LOCAL_FOOD_FALLBACK_LIMIT = 8


def build_exercise_documents() -> list[dict[str, Any]]:
    """Return normalized exercise documents for local retrieval."""

    documents: list[dict[str, Any]] = []
    for exercise in load_exercise_documents_source():
        name = str(exercise.get("name", "")).strip()
        if not name:
            continue
        primary = _list_text(exercise.get("primary_muscles") or exercise.get("target_muscle"))
        secondary = _list_text(exercise.get("secondary_muscles"))
        focus = _list_text(exercise.get("focus_tags"))
        equipment = _list_text(exercise.get("equipment"))
        difficulty = str(exercise.get("difficulty", "")).strip()
        movement_pattern = str(exercise.get("movement_pattern") or exercise.get("movement_type") or "").strip()
        replacement_group = str(exercise.get("replacement_group", "")).strip()
        notes = str(exercise.get("notes", "")).strip()
        contraindications = _list_text(exercise.get("contraindications"))
        goals = _list_text(exercise.get("training_goal_tags"))

        text = "\n".join(
            piece
            for piece in [
                f"Exercise: {name}",
                f"Difficulty: {difficulty}" if difficulty else "",
                f"Movement pattern: {movement_pattern}" if movement_pattern else "",
                f"Replacement group: {replacement_group}" if replacement_group else "",
                f"Primary muscles: {primary}" if primary else "",
                f"Secondary muscles: {secondary}" if secondary else "",
                f"Focus tags: {focus}" if focus else "",
                f"Equipment: {equipment}" if equipment else "",
                f"Training goals: {goals}" if goals else "",
                f"Contraindications: {contraindications}" if contraindications else "",
                f"Coaching notes: {notes}" if notes else "",
            ]
            if piece
        )
        documents.append(
            {
                "id": f"exercise:{exercise.get('id') or _slug(name)}",
                "type": "exercise",
                "title": name,
                "text": text,
                "metadata": {
                    "source": exercise.get("source") or "local",
                    "exercise_id": exercise.get("id", ""),
                    "name": name,
                    "difficulty": difficulty,
                    "movement_pattern": movement_pattern,
                    "replacement_group": replacement_group,
                    "primary_muscles": exercise.get("primary_muscles") or exercise.get("target_muscle") or [],
                    "secondary_muscles": exercise.get("secondary_muscles") or [],
                    "target_muscle": exercise.get("target_muscle") or [],
                    "focus_tags": exercise.get("focus_tags") or [],
                    "equipment": exercise.get("equipment") or [],
                    "contraindications": exercise.get("contraindications") or [],
                    "training_goal_tags": exercise.get("training_goal_tags") or [],
                    "youtube_url": exercise.get("youtube_url", ""),
                    "notes": notes,
                },
                "raw": exercise,
            }
        )
    return documents


def build_food_documents() -> list[dict[str, Any]]:
    """Return normalized food documents for nutrition retrieval."""

    documents: list[dict[str, Any]] = []
    for food in load_food_documents_source():
        name = str(food.get("name", "")).strip()
        if not name:
            continue
        category = str(food.get("category", "")).strip()
        diet_tags = _list_text(food.get("diet_tags"))
        allergens = _list_text(food.get("allergens"))
        notes = str(food.get("notes", "")).strip()
        macro_line = (
            f"Calories {food.get('calories_per_100g', 0)} kcal, "
            f"protein {food.get('protein_g', 0)}g, carbs {food.get('carbs_g', 0)}g, "
            f"fat {food.get('fat_g', 0)}g, fiber {food.get('fiber_g', 0)}g per 100g"
        )

        text = "\n".join(
            piece
            for piece in [
                f"Food: {name}",
                f"Category: {category}" if category else "",
                macro_line,
                f"Diet tags: {diet_tags}" if diet_tags else "",
                f"Allergens: {allergens}" if allergens else "",
                f"Nutrition notes: {notes}" if notes else "",
            ]
            if piece
        )
        documents.append(
            {
                "id": f"food:{food.get('id') or _slug(name)}",
                "type": "food",
                "title": name,
                "text": text,
                "metadata": {
                    "source": food.get("source") or "local",
                    "food_id": food.get("id", ""),
                    "name": name,
                    "category": category,
                    "diet_tags": food.get("diet_tags") or [],
                    "allergens": food.get("allergens") or [],
                    "calories_per_100g": food.get("calories_per_100g", 0),
                    "protein_g": food.get("protein_g", 0),
                    "carbs_g": food.get("carbs_g", 0),
                    "fat_g": food.get("fat_g", 0),
                    "fiber_g": food.get("fiber_g", 0),
                    "notes": notes,
                },
                "raw": food,
            }
        )
    return documents


@lru_cache(maxsize=1)
def load_exercise_documents_source() -> list[dict[str, Any]]:
    exercises = load_primary_exercise_documents_source()
    fallback = load_local_exercise_fallback_source()
    exercises.extend(fallback)
    return _dedupe_exercises(exercises)


@lru_cache(maxsize=1)
def load_food_documents_source() -> list[dict[str, Any]]:
    foods = load_primary_food_documents_source()
    foods.extend(load_local_food_fallback_source())
    return _dedupe_by_name(foods)


@lru_cache(maxsize=1)
def load_primary_exercise_documents_source() -> list[dict[str, Any]]:
    """Professional/RAG exercise sources used before local fallback."""

    exercises: list[dict[str, Any]] = []
    if EXERCISE_RAG_SEED_PATH.exists():
        exercises.extend(_load_json_list(EXERCISE_RAG_SEED_PATH))
    if WGER_CACHE_PATH.exists():
        exercises.extend(_load_json_list(WGER_CACHE_PATH))
    return _dedupe_exercises(exercises)


@lru_cache(maxsize=1)
def load_primary_food_documents_source() -> list[dict[str, Any]]:
    """Professional/RAG food sources used before local fallback."""

    if FOOD_RAG_SEED_PATH.exists():
        return _dedupe_by_name(_load_json_list(FOOD_RAG_SEED_PATH))
    return []


@lru_cache(maxsize=1)
def load_local_exercise_fallback_source() -> list[dict[str, Any]]:
    """Small local safety net kept only for missing RAG/API coverage."""

    fallback = _load_json_list(EXERCISE_DB_PATH)
    fallback = sorted(fallback, key=_local_exercise_fallback_priority)
    return [_mark_local_fallback(item) for item in fallback[:LOCAL_EXERCISE_FALLBACK_LIMIT]]


@lru_cache(maxsize=1)
def load_full_legacy_exercise_source() -> list[dict[str, Any]]:
    """Full legacy local DB for exact lookups on old persisted plans only."""

    return _load_json_list(EXERCISE_DB_PATH)


@lru_cache(maxsize=1)
def load_local_food_fallback_source() -> list[dict[str, Any]]:
    """Small local nutrition safety net kept only for missing RAG coverage."""

    fallback = _load_json_list(FOOD_DB_PATH)
    fallback = sorted(fallback, key=_local_food_fallback_priority)
    return [_mark_local_fallback(item) for item in fallback[:LOCAL_FOOD_FALLBACK_LIMIT]]


@lru_cache(maxsize=1)
def load_full_legacy_food_source() -> list[dict[str, Any]]:
    """Full legacy local DB for exact lookups on old persisted meals only."""

    return _load_json_list(FOOD_DB_PATH)


def _dedupe_exercises(exercises: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _dedupe_by_name(exercises)


def _dedupe_by_name(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for item in items:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        name_key = _slug(name)
        if name_key in seen_names:
            continue
        seen_names.add(name_key)
        deduped.append(item)
    return deduped


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _mark_local_fallback(item: dict[str, Any]) -> dict[str, Any]:
    return {**item, "source": "local_fallback"}


def _local_exercise_fallback_priority(exercise: dict[str, Any]) -> tuple[int, str]:
    name = _slug(str(exercise.get("name", "")))
    preferred_names = [
        "incline_push_up",
        "push_up",
        "machine_chest_press",
        "lat_pulldown",
        "seated_cable_row",
        "goblet_squat",
        "glute_bridge",
        "step_up",
        "dumbbell_lateral_raise",
        "face_pull",
        "plank",
        "dead_bug",
        "mountain_climber",
        "jump_rope",
    ]
    preferred_rank = {preferred_name: index for index, preferred_name in enumerate(preferred_names)}
    recommended = _list_text(exercise.get("recommended_for"))
    rank = preferred_rank.get(name, len(preferred_names) + 1)
    if "beginner_program" in recommended:
        rank -= 1
    return (rank, name)


def _local_food_fallback_priority(food: dict[str, Any]) -> tuple[int, str]:
    name = _slug(str(food.get("name", "")))
    preferred_names = [
        "chicken_breast",
        "egg",
        "firm_tofu",
        "brown_rice,_cooked",
        "white_rice,_cooked",
        "sweet_potato",
        "broccoli",
        "apple",
    ]
    preferred_rank = {preferred_name: index for index, preferred_name in enumerate(preferred_names)}
    return (preferred_rank.get(name, len(preferred_names) + 1), name)


def _list_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return ", ".join(str(item).strip() for item in value if str(item).strip())


def _slug(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")
