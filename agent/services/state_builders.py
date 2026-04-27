"""Pure state builders used by UI and API layers."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from agent.services.memory import memory_context_for_planning
from agent.services.persistence import safe_iso_date
from agent.state import FitnessAgentState


def build_initial_state(
    *,
    profile_inputs: dict[str, Any],
    thread_id: str,
    active_date: str,
    memory_store: dict[str, list[dict[str, Any]]],
) -> FitnessAgentState:
    target_date = display_reference_date(active_date or profile_inputs.get("start_date", date.today().isoformat()))
    return {
        "thread_id": thread_id,
        "current_date": target_date,
        "profile_notes": profile_inputs.get("profile_notes", ""),
        "plan_change_request": "",
        "normalized_change_request": {},
        "user_profile": {
            "user_id": "demo-user",
            "age": profile_inputs["age"],
            "sex": profile_inputs["sex"],
            "height_cm": profile_inputs["height_cm"],
            "weight_kg": profile_inputs["weight_kg"],
            "body_fat_pct": profile_inputs["body_fat_pct"],
            "fitness_level": profile_inputs["fitness_level"],
            "activity_level": profile_inputs.get("activity_level", "lightly_active"),
        },
        "constraints": {
            "sessions_per_week": profile_inputs["sessions_per_week"],
            "minutes_per_session": profile_inputs.get("minutes_per_session", 60),
            "available_days": profile_inputs["available_days"],
            "program_start_date": profile_inputs["start_date"],
            "injuries": split_csv(profile_inputs.get("injuries_text", "")),
            "pain_sensitive_areas": [],
            "food_allergies": split_csv(profile_inputs.get("allergies_text", "")),
            "dietary_preferences": profile_inputs["dietary_preferences"],
            "equipment_access": profile_inputs["equipment_access"],
        },
        "goals": {
            "primary_goal": profile_inputs["primary_goal"],
            "timeline_weeks": profile_inputs["timeline_weeks"],
            "target_weight_kg": profile_inputs["target_weight_kg"],
            "target_body_fat_pct": profile_inputs["target_body_fat_pct"],
        },
        "current_state": {
            "date": target_date,
            "weight_kg": profile_inputs["weight_kg"],
            "body_fat_pct": profile_inputs["body_fat_pct"],
            "sleep_hours": 7.0,
            "recovery_score": 0.75,
            "notes": profile_inputs.get("profile_notes", ""),
        },
        "latest_feedback": {},
        "daily_history": [],
        "memory_context": memory_context_for_planning(memory_store, target_date),
    }


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def display_reference_date(value: str) -> str:
    iso_value = safe_iso_date(value)
    if not iso_value:
        return date.today().isoformat()
    try:
        return datetime.fromisoformat(iso_value).date().isoformat()
    except ValueError:
        return date.today().isoformat()

