"""Food lookup and macro helpers backed by the local food database."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "food_db.json"


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_many(values: Iterable[str] | None) -> set[str]:
    if not values:
        return set()
    return {_normalize(value) for value in values}


@lru_cache(maxsize=1)
def load_food_db() -> list[dict[str, Any]]:
    """Load the local food database once per process."""

    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_food_by_id(food_id: str) -> dict[str, Any] | None:
    """Return one food item by id."""

    target_id = _normalize(food_id)
    for food in load_food_db():
        if _normalize(food["id"]) == target_id:
            return food
    return None


def get_food_by_name(name: str) -> dict[str, Any] | None:
    """Return one food item by display name."""

    target_name = _normalize(name)
    for food in load_food_db():
        if _normalize(food["name"]) == target_name:
            return food
    return None


def find_foods(
    *,
    category: str | None = None,
    diet_tags: Iterable[str] | None = None,
    excluded_allergens: Iterable[str] | None = None,
    min_protein_g: float | None = None,
    max_calories_per_100g: float | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Filter foods for planning around dietary style and macro targets."""

    diet_tag_set = _normalize_many(diet_tags)
    allergen_set = _normalize_many(excluded_allergens)
    category_normalized = _normalize(category) if category else None

    matches: list[dict[str, Any]] = []
    for food in load_food_db():
        food_category = _normalize(food.get("category", ""))
        food_diet_tags = _normalize_many(food.get("diet_tags", []))
        food_allergens = _normalize_many(food.get("allergens", []))

        if category_normalized and food_category != category_normalized:
            continue
        if diet_tag_set and not diet_tag_set.issubset(food_diet_tags):
            continue
        if allergen_set and allergen_set.intersection(food_allergens):
            continue
        if min_protein_g is not None and float(food.get("protein_g", 0.0)) < min_protein_g:
            continue
        if max_calories_per_100g is not None and float(food.get("calories_per_100g", 0.0)) > max_calories_per_100g:
            continue

        matches.append(food)
        if len(matches) >= limit:
            break

    return matches


def calculate_food_macros(food_id: str, grams: float) -> dict[str, float]:
    """Scale per-100g nutrition values to a requested serving size."""

    food = get_food_by_id(food_id)
    if not food:
        raise ValueError(f"Unknown food id: {food_id}")

    multiplier = grams / 100.0
    return {
        "food_id": food["id"],
        "food_name": food["name"],
        "grams": grams,
        "calories": round(float(food["calories_per_100g"]) * multiplier, 1),
        "protein_g": round(float(food["protein_g"]) * multiplier, 1),
        "carbs_g": round(float(food["carbs_g"]) * multiplier, 1),
        "fat_g": round(float(food["fat_g"]) * multiplier, 1),
        "fiber_g": round(float(food["fiber_g"]) * multiplier, 1),
    }


def calculate_total_macros(items: Iterable[dict[str, float | str]]) -> dict[str, float]:
    """Aggregate nutrition totals for multiple selected foods.

    Expected item format:
    {"food_id": "chicken_breast", "grams": 150}
    """

    totals = {
        "calories": 0.0,
        "protein_g": 0.0,
        "carbs_g": 0.0,
        "fat_g": 0.0,
        "fiber_g": 0.0,
    }

    for item in items:
        food_id = str(item["food_id"])
        grams = float(item["grams"])
        macros = calculate_food_macros(food_id, grams)
        for key in totals:
            totals[key] += float(macros[key])

    return {key: round(value, 1) for key, value in totals.items()}
