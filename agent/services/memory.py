"""Structured memory helpers for planning and AI Coach updates.

The runtime uses four memory layers:

1. session_metadata: current user/session/date/plan identifiers.
2. structured_profile: stable profile plus learned preferences.
3. conversation_summary: compact medium-term chat memory.
4. sliding_window: the last few raw chat turns.

Legacy recent_* collections are still exposed so existing planner prompts and
tooling keep working while newer code can consume the cleaner layers.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from agent.services.persistence import safe_iso_date

CHAT_WINDOW_LIMIT = 12
CONVERSATION_SUMMARY_CHAR_LIMIT = 1400

MEMORY_COLLECTIONS = [
    "injury_events",
    "food_preferences",
    "training_preferences",
    "plan_modification_logs",
    "body_metrics",
    "daily_feedback_records",
    "exercise_feedback_records",
    "plan_decisions",
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
    *,
    profile_inputs: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    session_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    store = normalize_memory_store(memory_store)
    safe_target = safe_iso_date(target_date) or date.today().isoformat()
    active_injuries = [
        injury
        for injury in store.get("injury_events", [])
        if memory_injury_is_active(injury, safe_target)
    ]
    legacy_context = {
        "active_injuries": active_injuries,
        "recent_food_preferences": store.get("food_preferences", [])[-10:],
        "recent_training_preferences": store.get("training_preferences", [])[-10:],
        "recent_plan_modifications": store.get("plan_modification_logs", [])[-12:],
        "recent_daily_feedback": store.get("daily_feedback_records", [])[-14:],
        "recent_exercise_feedback": store.get("exercise_feedback_records", [])[-30:],
        "recent_body_metrics": store.get("body_metrics", [])[-14:],
    }
    session_state = session_state or {}
    result = result or {}
    return {
        **legacy_context,
        "session_metadata": build_session_metadata(
            target_date=safe_target,
            result=result,
            session_state=session_state,
        ),
        "structured_profile": build_structured_profile(
            profile_inputs=profile_inputs or {},
            result=result,
            memory_store=store,
            active_injuries=active_injuries,
        ),
        "conversation_summary": str(session_state.get("conversation_summary") or "").strip(),
        "sliding_window": conversation_sliding_window(
            session_state.get("assistant_chat_messages", []),
            limit=CHAT_WINDOW_LIMIT,
        ),
    }


def build_session_metadata(
    *,
    target_date: str,
    result: dict[str, Any],
    session_state: dict[str, Any],
) -> dict[str, Any]:
    current_plan = _as_dict(result.get("current_plan"))
    return {
        "user_id": _first_text(result.get("user_profile", {}).get("user_id"), "demo-user"),
        "thread_id": _first_text(session_state.get("thread_id"), result.get("thread_id"), ""),
        "active_date": target_date,
        "current_cycle": current_plan.get("cycle_number"),
        "current_plan_id": current_plan.get("plan_id", ""),
        "cycle_start_date": current_plan.get("cycle_start_date", ""),
        "cycle_end_date": current_plan.get("cycle_end_date", ""),
        "last_action_message": str(session_state.get("last_action_message") or ""),
    }


def build_structured_profile(
    *,
    profile_inputs: dict[str, Any],
    result: dict[str, Any],
    memory_store: dict[str, list[dict[str, Any]]],
    active_injuries: list[dict[str, Any]],
) -> dict[str, Any]:
    user_profile = {**profile_inputs, **_as_dict(result.get("user_profile"))}
    constraints = _as_dict(result.get("constraints"))
    goals = _as_dict(result.get("goals"))
    latest_body_metric = _latest_dict(memory_store.get("body_metrics", []))
    learned_preferences = build_learned_preferences(memory_store, active_injuries)
    return {
        "static_profile": {
            "user_id": _first_text(user_profile.get("user_id"), "demo-user"),
            "age": user_profile.get("age"),
            "sex": user_profile.get("sex"),
            "height_cm": user_profile.get("height_cm"),
            "activity_level": user_profile.get("activity_level"),
        },
        "body_metrics": {
            "weight_kg": latest_body_metric.get("weight_kg", user_profile.get("weight_kg")),
            "body_fat_pct": latest_body_metric.get("body_fat_pct", user_profile.get("body_fat_pct")),
            "last_recorded_date": latest_body_metric.get("date") or latest_body_metric.get("record_date", ""),
        },
        "training_profile": {
            "fitness_level": _first_text(user_profile.get("fitness_level"), "beginner"),
            "sessions_per_week": constraints.get("sessions_per_week") or profile_inputs.get("sessions_per_week"),
            "minutes_per_session": constraints.get("minutes_per_session") or profile_inputs.get("minutes_per_session"),
            "available_days": constraints.get("available_days") or profile_inputs.get("available_days", []),
        },
        "goals": {
            "primary_goal": goals.get("primary_goal") or profile_inputs.get("primary_goal", ""),
            "timeline_weeks": goals.get("timeline_weeks") or profile_inputs.get("timeline_weeks"),
            "target_weight_kg": goals.get("target_weight_kg") or profile_inputs.get("target_weight_kg"),
            "target_body_fat_pct": goals.get("target_body_fat_pct") or profile_inputs.get("target_body_fat_pct"),
        },
        "learned_preferences": learned_preferences,
    }


def build_learned_preferences(
    memory_store: dict[str, list[dict[str, Any]]],
    active_injuries: list[dict[str, Any]],
) -> dict[str, Any]:
    exercise_feedback = memory_store.get("exercise_feedback_records", [])
    plan_logs = memory_store.get("plan_modification_logs", [])
    liked_exercises: list[str] = []
    difficult_exercises: list[str] = []
    preferred_focuses: list[str] = []
    for item in exercise_feedback[-60:]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("exercise_name") or item.get("name") or "").strip()
        focus = str(item.get("focus") or "").strip()
        emoji = str(item.get("feeling_emoji") or item.get("emoji") or "")
        feeling = str(item.get("workout_feeling") or item.get("feeling") or "").lower()
        status = str(item.get("status") or item.get("workout_status") or "").lower()
        if focus:
            preferred_focuses.append(focus)
        if name and emoji in {"😊", "🙂"} and status != "cancelled":
            liked_exercises.append(name)
        if name and (emoji in {"😫", "😖"} or "pain" in feeling or "hard" in feeling or status == "cancelled"):
            difficult_exercises.append(name)

    avoided_foods: list[str] = []
    preferred_foods: list[str] = []
    for item in memory_store.get("food_preferences", [])[-40:]:
        if not isinstance(item, dict):
            continue
        food = str(item.get("food") or item.get("name") or item.get("preference") or "").strip()
        scope = str(item.get("scope") or item.get("type") or "").lower()
        if not food:
            continue
        if "avoid" in scope or item.get("avoid") is True:
            avoided_foods.append(food)
        else:
            preferred_foods.append(food)

    for item in plan_logs[-30:]:
        if isinstance(item, dict):
            summary = str(item.get("summary") or "").strip()
            if summary:
                preferred_focuses.append(summary)

    return {
        "liked_exercises": _dedupe_keep_order(liked_exercises)[-12:],
        "difficult_exercises": _dedupe_keep_order(difficult_exercises)[-12:],
        "preferred_focuses": _dedupe_keep_order(preferred_focuses)[-10:],
        "avoided_foods": _dedupe_keep_order(avoided_foods)[-12:],
        "preferred_foods": _dedupe_keep_order(preferred_foods)[-12:],
        "active_injury_areas": _dedupe_keep_order(
            str(injury.get("area") or injury.get("injury_area") or "").strip()
            for injury in active_injuries
            if isinstance(injury, dict)
        ),
    }


def compact_conversation_memory(
    messages: list[dict[str, Any]],
    existing_summary: str = "",
    *,
    limit: int = CHAT_WINDOW_LIMIT,
) -> tuple[str, list[dict[str, Any]]]:
    clean_messages = [
        {"role": str(message.get("role") or ""), "content": str(message.get("content") or "").strip()}
        for message in messages
        if isinstance(message, dict) and str(message.get("content") or "").strip()
    ]
    if len(clean_messages) <= limit:
        return existing_summary.strip(), clean_messages

    archived = clean_messages[:-limit]
    window = clean_messages[-limit:]
    archived_summary = summarize_chat_messages(archived)
    summary = " | ".join(part for part in [existing_summary.strip(), archived_summary] if part)
    if len(summary) > CONVERSATION_SUMMARY_CHAR_LIMIT:
        summary = summary[-CONVERSATION_SUMMARY_CHAR_LIMIT:].lstrip(" |")
    return summary, window


def summarize_chat_messages(messages: list[dict[str, Any]], *, max_items: int = 8) -> str:
    pairs: list[str] = []
    for message in messages[-max_items:]:
        role = "User" if message.get("role") == "user" else "Coach"
        content = str(message.get("content") or "").strip()
        if content:
            pairs.append(f"{role}: {_truncate(content, 120)}")
    return "Earlier chat summary: " + " / ".join(pairs) if pairs else ""


def conversation_sliding_window(messages: Any, *, limit: int = CHAT_WINDOW_LIMIT) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    return [
        {"role": str(message.get("role") or ""), "content": str(message.get("content") or "").strip()}
        for message in messages[-limit:]
        if isinstance(message, dict) and str(message.get("content") or "").strip()
    ]


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


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _latest_dict(items: Any) -> dict[str, Any]:
    if not isinstance(items, list):
        return {}
    for item in reversed(items):
        if isinstance(item, dict):
            return item
    return {}


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _dedupe_keep_order(values: Any) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            output.append(text)
    return output


def _truncate(value: str, limit: int) -> str:
    text = str(value).strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
