"""Display-time enrichment for existing plan payloads."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.tools import build_exercise_plan_payload, build_video_resources, get_exercise_by_name


def hydrate_agent_result_for_display(result: dict[str, Any]) -> dict[str, Any]:
    """Backfill exercise teaching fields and videos for persisted plans."""

    if not isinstance(result, dict) or not result.get("current_plan"):
        return result

    hydrated = deepcopy(result)
    current_plan = hydrated.get("current_plan") or {}
    sessions = current_plan.get("workout_sessions") or []
    if not isinstance(sessions, list):
        return hydrated

    exercise_names: list[str] = []
    for session in sessions:
        if not isinstance(session, dict):
            continue
        focus = str(session.get("focus", "")).strip()
        exercises = session.get("exercises") or []
        if not isinstance(exercises, list):
            continue
        enriched_exercises = []
        for exercise in exercises:
            if not isinstance(exercise, dict):
                continue
            enriched = _hydrate_exercise(exercise, focus=focus)
            if str(enriched.get("name", "")).strip():
                exercise_names.append(str(enriched["name"]).strip())
            enriched_exercises.append(enriched)
        session["exercises"] = enriched_exercises

    if exercise_names:
        hydrated["youtube_resources"] = build_video_resources(exercise_names)
    return hydrated


def _hydrate_exercise(exercise: dict[str, Any], *, focus: str) -> dict[str, Any]:
    if _has_teaching_fields(exercise):
        return dict(exercise)

    exercise_name = str(exercise.get("name", "")).strip()
    source_exercise = get_exercise_by_name(exercise_name)
    if not source_exercise:
        return dict(exercise)

    hydrated = build_exercise_plan_payload(
        source_exercise,
        sets=_coerce_int(exercise.get("sets"), default=4),
        reps=str(exercise.get("reps") or "").strip() or "10-15",
        notes=str(exercise.get("notes") or "").strip(),
        focus=focus,
    )

    for key, value in exercise.items():
        if value not in (None, "", []):
            hydrated[key] = value
    return hydrated


def _has_teaching_fields(exercise: dict[str, Any]) -> bool:
    return bool(
        exercise.get("why_this_exercise")
        and exercise.get("coaching_cue")
        and exercise.get("common_mistake")
    )


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
