"""Shared planning/date helpers independent from UI frameworks."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from agent.services.persistence import safe_iso_date

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_INDEX = {day: index for index, day in enumerate(WEEKDAY_ORDER)}


def default_equipment_access() -> list[str]:
    return ["bodyweight", "dumbbell", "barbell", "bench", "rack", "cable_machine", "kettlebell", "box"]


def sort_days(days: list[str]) -> list[str]:
    normalized_days: list[str] = []
    for day in days:
        cleaned = str(day).strip()
        if cleaned in WEEKDAY_INDEX and cleaned not in normalized_days:
            normalized_days.append(cleaned)
    return sorted(normalized_days, key=lambda item: WEEKDAY_INDEX[item])


def sort_workout_sessions(workout_sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        workout_sessions,
        key=lambda session: (
            str(session.get("scheduled_date", "")) or "9999-12-31",
            WEEKDAY_INDEX.get(str(session.get("day", "")), 99),
        ),
    )


def iso_to_date(value: str) -> date:
    try:
        return datetime.fromisoformat(safe_iso_date(value)).date()
    except ValueError:
        return date.today()


def next_calendar_date(value: str) -> str:
    return (iso_to_date(value) + timedelta(days=1)).isoformat()


def target_starts_new_cycle(current_plan: dict[str, Any], target_date: str) -> bool:
    cycle_end_date = safe_iso_date(current_plan.get("cycle_end_date"))
    target_iso = safe_iso_date(target_date)
    if not cycle_end_date or not target_iso:
        return False
    try:
        return datetime.fromisoformat(target_iso).date() > datetime.fromisoformat(cycle_end_date).date()
    except ValueError:
        return False


def current_interaction_date(
    *,
    active_date: str | None,
    result: dict[str, Any] | None = None,
) -> str:
    result = result or {}
    return safe_iso_date(active_date or result.get("current_date") or date.today().isoformat()) or date.today().isoformat()


def session_for_history_date(
    result: dict[str, Any],
    feedback_date: str,
    current_session: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if current_session and same_iso_date(current_session.get("scheduled_date"), feedback_date):
        return current_session
    sessions = sort_workout_sessions(result.get("current_plan", {}).get("workout_sessions", []))
    for session in sessions:
        if same_iso_date(session.get("scheduled_date"), feedback_date):
            return session
    return {}


def same_iso_date(left: object, right: object) -> bool:
    left_date = safe_iso_date(left)
    right_date = safe_iso_date(right)
    return bool(left_date and right_date and left_date == right_date)

