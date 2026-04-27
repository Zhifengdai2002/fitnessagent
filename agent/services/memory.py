"""Structured memory helpers for planning and AI Coach updates."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from agent.services.persistence import safe_iso_date

MEMORY_COLLECTIONS = [
    "injury_events",
    "food_preferences",
    "training_preferences",
    "plan_modification_logs",
    "body_metrics",
    "daily_feedback_records",
]


def default_memory_store() -> dict[str, list[dict[str, Any]]]:
    return {collection: [] for collection in MEMORY_COLLECTIONS}


def normalize_memory_store(memory_store: Any) -> dict[str, list[dict[str, Any]]]:
    normalized = default_memory_store()
    if not isinstance(memory_store, dict):
        return normalized
    for key in normalized:
        value = memory_store.get(key, [])
        normalized[key] = value if isinstance(value, list) else []
    return normalized


def append_memory_item(
    memory_store: dict[str, list[dict[str, Any]]],
    collection: str,
    item: dict[str, Any],
    *,
    unique_key: str | None = None,
    limit: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    store = normalize_memory_store(memory_store)
    if not item or collection not in store:
        return store

    items = list(store.get(collection, []))
    if unique_key and item.get(unique_key):
        for index, existing in enumerate(items):
            if existing.get(unique_key) == item.get(unique_key):
                items[index] = item
                store[collection] = items
                return store

    items.append(item)
    store[collection] = items[-limit:]
    return store


def memory_context_for_planning(
    memory_store: dict[str, list[dict[str, Any]]],
    target_date: str,
) -> dict[str, Any]:
    store = normalize_memory_store(memory_store)
    safe_target = safe_iso_date(target_date) or date.today().isoformat()
    active_injuries = [
        injury
        for injury in store.get("injury_events", [])
        if memory_injury_is_active(injury, safe_target)
    ]
    return {
        "active_injuries": active_injuries,
        "recent_food_preferences": store.get("food_preferences", [])[-10:],
        "recent_training_preferences": store.get("training_preferences", [])[-10:],
        "recent_plan_modifications": store.get("plan_modification_logs", [])[-12:],
        "recent_daily_feedback": store.get("daily_feedback_records", [])[-14:],
        "recent_body_metrics": store.get("body_metrics", [])[-14:],
    }


def memory_injury_is_active(injury: dict[str, Any], target_date: str) -> bool:
    if str(injury.get("status", "active")).lower() != "active":
        return False
    injury_date = safe_iso_date(injury.get("date"))
    if not injury_date:
        return False
    try:
        days = (datetime.fromisoformat(target_date).date() - datetime.fromisoformat(injury_date).date()).days
    except ValueError:
        return False
    expires_after = _clamp_int(injury.get("expires_after_days"), 1, 60, 7)
    return 0 <= days <= expires_after


def _clamp_int(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = fallback
    return max(minimum, min(maximum, number))

