"""Shared constants and pure lookups for the planner and its sub-modules.

Keeping these here avoids circular imports between planner.py and the
services that planner.py delegates to.
"""

from __future__ import annotations

DEFAULT_DAYS = ["Monday", "Tuesday", "Thursday", "Saturday"]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_INDEX = {day: index for index, day in enumerate(WEEKDAY_ORDER)}

GOAL_LABELS = {
    "weight_loss": "减重",
    "strength": "力量",
    "sculpting": "塑形",
}

FOCUS_LIBRARY: dict[str, dict] = {
    "upper_chest_arms": {
        "label": "Upper Body (Chest + Arms)",
        "target_muscles": ["chest", "triceps", "biceps"],
        "movement_type": None,
    },
    "upper_shoulders": {
        "label": "Upper Body (Shoulders)",
        "target_muscles": ["shoulders", "side delts", "rear delts"],
        "movement_type": None,
    },
    "back_training": {
        "label": "Back Training",
        "target_muscles": ["lats", "upper back", "mid back"],
        "movement_type": "pull",
    },
    "lower_legs_glutes": {
        "label": "Lower Body (Legs + Glutes)",
        "target_muscles": ["quads", "glutes", "hamstrings"],
        "movement_type": None,
    },
    "functional_core": {
        "label": "Functional (Core + Abs)",
        "target_muscles": ["core", "abs", "obliques"],
        "movement_type": "core",
    },
    "functional_power": {
        "label": "Functional (Power)",
        "target_muscles": ["glutes", "quads", "core", "shoulders"],
        "movement_type": "power",
    },
    "functional_conditioning": {
        "label": "Functional (Conditioning)",
        "target_muscles": ["core", "glutes", "shoulders"],
        "movement_type": "conditioning",
    },
}

GOAL_FOCUS_TEMPLATES: dict[str, list[str]] = {
    "weight_loss": [
        "upper_chest_arms",
        "lower_legs_glutes",
        "functional_conditioning",
        "back_training",
        "functional_core",
    ],
    "strength": [
        "upper_chest_arms",
        "back_training",
        "lower_legs_glutes",
        "upper_shoulders",
        "functional_power",
    ],
    "sculpting": [
        "upper_chest_arms",
        "upper_shoulders",
        "back_training",
        "lower_legs_glutes",
        "functional_core",
    ],
}

FOCUS_ALIASES: dict[str, str] = {
    "legs": "lower_legs_glutes",
    "practice legs": "lower_legs_glutes",
    "train legs": "lower_legs_glutes",
    "leg training": "lower_legs_glutes",
    "leg day": "lower_legs_glutes",
    "leg focus": "lower_legs_glutes",
    "lower body": "lower_legs_glutes",
    "lower body (legs + glutes)": "lower_legs_glutes",
    "upper body": "upper_chest_arms",
    "upper body (chest + arms)": "upper_chest_arms",
    "shoulders": "upper_shoulders",
    "upper body (shoulders)": "upper_shoulders",
    "back": "back_training",
    "back training": "back_training",
    "core": "functional_core",
    "functional (core + abs)": "functional_core",
    "power": "functional_power",
    "functional (power)": "functional_power",
    "conditioning": "functional_conditioning",
    "conditioning strength": "functional_conditioning",
    "functional (conditioning)": "functional_conditioning",
}


def map_goal_tag(primary_goal: str) -> str:
    """Normalize any goal string to one of: weight_loss | strength | sculpting."""
    normalized = primary_goal.strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "减重": "weight_loss",
        "fat_loss": "weight_loss",
        "weight_loss": "weight_loss",
        "力量": "strength",
        "strength": "strength",
        "塑形": "sculpting",
        "sculpting": "sculpting",
        "body_recomposition": "sculpting",
    }
    return mapping.get(normalized, "weight_loss")
