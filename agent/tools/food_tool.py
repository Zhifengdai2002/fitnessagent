"""Food lookup and macro helpers backed by the local food database."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

from agent.rag.documents import (
    load_full_legacy_food_source,
    load_local_food_fallback_source,
    load_primary_food_documents_source,
)

LOCAL_FALLBACK_SOURCES = {"", "local", "local_fallback"}


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_many(values: Iterable[str] | None) -> set[str]:
    if not values:
        return set()
    return {_normalize(value) for value in values}


@lru_cache(maxsize=1)
def load_food_db() -> list[dict[str, Any]]:
    """Load planning food sources with local JSON kept as a tiny fallback."""

    foods: list[dict[str, Any]] = []
    foods.extend(load_primary_food_documents_source())
    foods.extend(load_local_food_fallback_source())
    return _dedupe_foods(foods)


def get_food_by_id(food_id: str) -> dict[str, Any] | None:
    """Return one food item by id."""

    target_id = _normalize(food_id)
    for food in load_food_db():
        if _normalize(food["id"]) == target_id:
            return food
    for food in load_full_legacy_food_source():
        if _normalize(str(food.get("id", ""))) == target_id:
            return {**food, "source": food.get("source") or "local_fallback"}
    return None


def get_food_by_name(name: str) -> dict[str, Any] | None:
    """Return one food item by display name."""

    target_name = _normalize(name)
    for food in load_food_db():
        if _normalize(food["name"]) == target_name:
            return food
    for food in load_full_legacy_food_source():
        if _normalize(str(food.get("name", ""))) == target_name:
            return {**food, "source": food.get("source") or "local_fallback"}
    return None


def find_foods(
    *,
    category: str | None = None,
    diet_tags: Iterable[str] | None = None,
    excluded_allergens: Iterable[str] | None = None,
    min_protein_g: float | None = None,
    max_calories_per_100g: float | None = None,
    learned_preferences: dict[str, Any] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Filter foods for planning around dietary style and macro targets."""

    rag_matches = _retrieve_rag_foods(
        category=category,
        diet_tags=diet_tags,
        excluded_allergens=excluded_allergens,
        min_protein_g=min_protein_g,
        max_calories_per_100g=max_calories_per_100g,
        learned_preferences=learned_preferences,
        limit=limit,
    )
    if rag_matches:
        return rag_matches

    diet_tag_set = _normalize_many(diet_tags)
    allergen_set = _normalize_many(excluded_allergens)
    category_normalized = _normalize(category) if category else None

    primary_matches: list[dict[str, Any]] = []
    fallback_matches: list[dict[str, Any]] = []
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

        if _source_tier(food) == 0:
            primary_matches.append(food)
        else:
            fallback_matches.append(food)
        if len(primary_matches) >= limit:
            break

    if len(primary_matches) >= limit:
        return primary_matches[:limit]
    return [*primary_matches, *fallback_matches[: max(0, limit - len(primary_matches))]]


def _retrieve_rag_foods(
    *,
    category: str | None,
    diet_tags: Iterable[str] | None,
    excluded_allergens: Iterable[str] | None,
    min_protein_g: float | None,
    max_calories_per_100g: float | None,
    learned_preferences: dict[str, Any] | None,
    limit: int,
) -> list[dict[str, Any]]:
    try:
        from agent.rag.retriever import retrieve_foods
    except Exception:
        return []

    query_parts = [
        category or "food",
        " ".join(diet_tags or []),
        "high protein" if min_protein_g is not None else "",
        "low calorie" if max_calories_per_100g is not None else "",
    ]
    try:
        return retrieve_foods(
            query=" ".join(part for part in query_parts if part),
            category=category,
            diet_tags=diet_tags,
            excluded_allergens=excluded_allergens,
            min_protein_g=min_protein_g,
            max_calories_per_100g=max_calories_per_100g,
            learned_preferences=learned_preferences,
            limit=limit,
        )
    except Exception:
        return []


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


def _dedupe_foods(foods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for food in foods:
        name = str(food.get("name", "")).strip()
        if not name:
            continue
        name_key = _normalize(name)
        if name_key in seen_names:
            continue
        seen_names.add(name_key)
        deduped.append(food)
    return deduped


def _source_tier(food: dict[str, Any]) -> int:
    """Return 0 for RAG/imported foods and 1 for local fallback foods."""

    source = _normalize(str(food.get("source", "")))
    return 1 if source in LOCAL_FALLBACK_SOURCES else 0
