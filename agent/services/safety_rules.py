"""Hard safety rules evaluated in code — not by the LLM.

These rules run at the API entry layer before any LLM call. They cannot be
overridden by prompt engineering or model output.

Rules:
  1. sleep_hours < 5.0  → cancel today's workout
  2. Injury language in feedback text → cancel today's workout
"""

from __future__ import annotations

from typing import Any

SLEEP_CANCEL_THRESHOLD = 5.0

_NEGATED_TERMS = [
    "no injury", "not injured", "no pain", "pain free", "pain-free",
    "doesn't hurt", "does not hurt", "feels fine", "recovered",
    "恢复了", "不疼", "没有受伤",
]

_INJURY_TERMS = [
    "injured", "injury", "hurt", "hurts", "pain", "painful", "ache",
    "strain", "strained", "sprain", "sprained", "pulled",
    "受伤", "疼", "痛", "拉伤", "扭伤",
]


def contains_injury_language(text: str) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in _NEGATED_TERMS):
        return False
    return any(term in lowered for term in _INJURY_TERMS)


def check_hard_rules(
    *,
    sleep_hours: float | None = None,
    feedback_text: str = "",
) -> dict[str, Any] | None:
    """Return a cancellation dict if a hard rule fires, else None.

    The caller should cancel today's workout and skip the planner when
    this returns a non-None value.
    """
    reasons: list[str] = []

    if sleep_hours is not None and sleep_hours < SLEEP_CANCEL_THRESHOLD:
        reasons.append(
            f"Sleep was only {sleep_hours:.1f} hours, below the "
            f"{SLEEP_CANCEL_THRESHOLD}-hour safety threshold."
        )

    if feedback_text and contains_injury_language(feedback_text):
        reasons.append("Injury or pain was reported in today's feedback.")

    if not reasons:
        return None

    return {
        "cancel": True,
        "reasons": reasons,
        "sleep_hard_stop": sleep_hours is not None and sleep_hours < SLEEP_CANCEL_THRESHOLD,
        "injury_reported": bool(feedback_text and contains_injury_language(feedback_text)),
    }


def cancel_today_session_in_plan(plan: dict[str, Any], today: str, reasons: list[str]) -> None:
    """Mark today's workout session as cancelled in-place."""
    for session in plan.get("workout_sessions", []):
        if session.get("scheduled_date") == today or session.get("is_today"):
            session["is_cancelled"] = True
            existing = list(session.get("safety_notes", []))
            session["safety_notes"] = existing + reasons
            break
