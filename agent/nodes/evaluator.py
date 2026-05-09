"""Evaluator node — rule-based plan safety and quality checks.

Runs after the Planner and validates the generated cycle plan against:
1. Active injury conflicts — exercises targeting injured muscle groups
2. Schedule overload — planned sessions exceed user's available days
3. Duration mismatch — sessions significantly over the user's stated limit
4. Duplicate focus — same focus scheduled on consecutive days without rest

Warnings are written to evaluation_result for logging and frontend display.
needs_revision stays False (no re-generation triggered) — warnings are
informational so the coach can surface them to the user.
"""

from __future__ import annotations

from agent.state import FitnessAgentState


def feedback_evaluation_node(state: FitnessAgentState) -> FitnessAgentState:
    warnings: list[str] = []

    plan = state.get("current_plan") or {}
    sessions = [s for s in (plan.get("workout_sessions") or []) if isinstance(s, dict)]
    constraints = state.get("constraints") or {}
    memory_context = state.get("memory_context") or {}

    active_injuries = [
        inj for inj in (memory_context.get("active_injuries") or [])
        if isinstance(inj, dict)
    ]
    injury_areas = {
        _norm(str(inj.get("area") or inj.get("injury_area") or ""))
        for inj in active_injuries
        if inj.get("area") or inj.get("injury_area")
    }

    available_days = int(constraints.get("sessions_per_week") or 7)
    max_duration = int(constraints.get("minutes_per_session") or 90)

    active_sessions = [s for s in sessions if not s.get("is_cancelled") and not s.get("is_rest")]

    # 1. Injury conflict check
    if injury_areas:
        for session in active_sessions:
            for exercise in (session.get("exercises") or []):
                if not isinstance(exercise, dict):
                    continue
                muscles = {
                    _norm(m)
                    for m in (
                        (exercise.get("primary_muscles") or [])
                        + [str(exercise.get("target_muscle") or "")]
                    )
                    if m
                }
                contraindications = {
                    _norm(c) for c in (exercise.get("contraindications") or []) if c
                }
                conflict = injury_areas & (muscles | contraindications)
                if conflict:
                    name = str(exercise.get("name") or "Unknown exercise")
                    warnings.append(
                        f"'{name}' targets {', '.join(conflict)} which overlaps an active injury."
                    )

    # 2. Schedule overload check
    if len(active_sessions) > available_days:
        warnings.append(
            f"Plan has {len(active_sessions)} active sessions but user constraint is "
            f"{available_days} sessions/week."
        )

    # 3. Duration mismatch check
    for session in active_sessions:
        duration = int(session.get("duration_minutes") or 0)
        if duration > max_duration + 15:
            date_label = session.get("scheduled_date") or session.get("day") or "a session"
            warnings.append(
                f"Session on {date_label} is {duration} min, exceeding the {max_duration} min limit."
            )

    # 4. Consecutive same-focus check
    dated = [
        s for s in active_sessions
        if s.get("scheduled_date") and s.get("focus")
    ]
    dated.sort(key=lambda s: str(s.get("scheduled_date") or ""))
    for i in range(1, len(dated)):
        if dated[i].get("focus") == dated[i - 1].get("focus"):
            focus = str(dated[i].get("focus"))
            d1 = str(dated[i - 1].get("scheduled_date"))
            d2 = str(dated[i].get("scheduled_date"))
            warnings.append(
                f"Same focus '{focus}' scheduled on consecutive days {d1} and {d2}."
            )

    return {
        "evaluation_result": {"warnings": warnings, "warning_count": len(warnings)},
        "needs_revision": False,
        "revision_reason": "",
        "feedback_history": list(state.get("feedback_history", [])),
        "state_history": list(state.get("state_history", [])),
    }


def _norm(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")
