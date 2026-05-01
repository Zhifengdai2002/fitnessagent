"""Nutrition target calculation and meal suggestion building.

Extracted from planner.py to isolate food-related logic.
"""

from __future__ import annotations

from typing import Any

from agent.services.planner_constants import GOAL_LABELS, map_goal_tag
from agent.state import MealSuggestion
from agent.tools import calculate_food_macros, find_foods


def build_nutrition_targets(
    user_profile: dict[str, Any],
    goals: dict[str, Any],
    current_state: dict[str, Any],
) -> dict[str, int | float]:
    current_weight = float(user_profile.get("weight_kg") or 70.0)
    primary_goal = map_goal_tag(str(goals.get("primary_goal", "weight_loss")))

    if primary_goal == "weight_loss":
        calories = int(current_weight * 28)
    elif primary_goal == "strength":
        calories = int(current_weight * 33)
    else:
        calories = int(current_weight * 29)

    protein_g = int(round(current_weight * 1.8))
    fat_g = int(round(current_weight * 0.8))
    carbs_g = max(100, int(round((calories - protein_g * 4 - fat_g * 9) / 4)))

    return {
        "daily_calories": calories,
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g,
        "hydration_liters": round(max(2.0, current_weight * 0.035), 1),
    }


def build_meal_suggestions(
    *,
    goals: dict[str, Any],
    constraints: dict[str, Any],
    nutrition_targets: dict[str, Any],
    learned_preferences: dict[str, Any] | None = None,
) -> list[MealSuggestion]:
    primary_goal = map_goal_tag(str(goals.get("primary_goal", "weight_loss")))
    dietary_preferences = constraints.get("dietary_preferences", [])
    food_allergies = constraints.get("food_allergies", [])

    protein_pool = find_foods(
        category="protein",
        diet_tags=dietary_preferences or None,
        excluded_allergens=food_allergies,
        min_protein_g=10,
        learned_preferences=learned_preferences,
        limit=3,
    )
    carb_pool = find_foods(
        category="carb",
        diet_tags=[t for t in dietary_preferences if t in {"vegan", "gluten_free"}] or None,
        excluded_allergens=food_allergies,
        learned_preferences=learned_preferences,
        limit=3,
    )
    fruit_pool = find_foods(
        category="fruit",
        excluded_allergens=food_allergies,
        learned_preferences=learned_preferences,
        limit=2,
    )

    serving_plan = [
        ("breakfast", protein_pool[:1] + carb_pool[:1] + fruit_pool[:1], 100),
        ("lunch", protein_pool[1:2] + carb_pool[1:2], 150),
        ("dinner", protein_pool[2:3] + carb_pool[2:3], 180),
    ]
    if primary_goal != "weight_loss":
        serving_plan.append(("snack", protein_pool[:1] + fruit_pool[1:2], 80))

    suggestions: list[MealSuggestion] = []
    for meal_slot, foods, grams in serving_plan:
        for food in foods:
            macro = calculate_food_macros(food["id"], grams)
            suggestions.append({
                "food_name": food["name"],
                "serving_size": f"{grams}g",
                "calories": int(round(macro["calories"])),
                "protein_g": macro["protein_g"],
                "carbs_g": macro["carbs_g"],
                "fat_g": macro["fat_g"],
                "meal_slot": meal_slot,
            })

    if not suggestions:
        suggestions.append({
            "food_name": "Balanced whole-food meal",
            "serving_size": "1 plate",
            "calories": int(nutrition_targets["daily_calories"] // 3),
            "protein_g": round(float(nutrition_targets["protein_g"]) / 3, 1),
            "carbs_g": round(float(nutrition_targets["carbs_g"]) / 3, 1),
            "fat_g": round(float(nutrition_targets["fat_g"]) / 3, 1),
            "meal_slot": "lunch",
        })
    return suggestions


def build_food_candidates(
    constraints: dict[str, Any],
    *,
    learned_preferences: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    dietary_preferences = constraints.get("dietary_preferences", [])
    food_allergies = constraints.get("food_allergies", [])
    candidates: list[dict[str, Any]] = []
    for category in ["protein", "carb", "fruit", "vegetable", "fat"]:
        for food in find_foods(
            category=category,
            diet_tags=dietary_preferences or None,
            excluded_allergens=food_allergies,
            learned_preferences=learned_preferences,
            limit=4,
        ):
            candidates.append({
                "name": food["name"],
                "category": food["category"],
                "protein_g": food["protein_g"],
                "carbs_g": food["carbs_g"],
                "fat_g": food["fat_g"],
                "calories_per_100g": food["calories_per_100g"],
            })
    return candidates


def extract_grams(serving_size: str, default: float) -> float:
    digits = "".join(ch for ch in serving_size if ch.isdigit() or ch == ".")
    return float(digits) if digits else default
