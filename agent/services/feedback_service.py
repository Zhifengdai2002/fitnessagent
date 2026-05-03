"""Daily feedback and cycle rollover services."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from agent.graph import run_agent
from agent.services.memory import (
    append_memory_item,
    memory_context_for_planning,
    normalize_memory_store,
)
from agent.services.persistence import safe_iso_date
from agent.services.planning_helpers import session_for_history_date, sort_workout_sessions
from agent.services.state_builders import build_initial_state
from agent.state import FitnessAgentState

FEELING_EMOJI_LABELS = {
    "😊": "Good",
    "😐": "Okay",
    "😫": "Hard",
}


def record_daily_feedback_and_advance(
    *,
    previous_result: FitnessAgentState,
    current_session: dict[str, Any],
    feedback_date: str,
    target_date: str,
    current_weight_kg: float,
    current_body_fat_pct: float,
    workout_feeling: str,
    feeling_emoji: str,
    adherence_score: float = 1.0,
    memory_store: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[FitnessAgentState, dict[str, list[dict[str, Any]]]]:
    store = normalize_memory_store(memory_store)
    updated_result: FitnessAgentState = dict(previous_result)
    current_session = session_for_history_date(previous_result, feedback_date, current_session)
    completed_actions = completed_actions_from_session(current_session)
    feedback_notes = workout_feeling.strip()
    latest_feedback = build_daily_feedback(
        feedback_date=feedback_date,
        completed_actions=completed_actions,
        workout_feeling=feedback_notes,
        feeling_emoji=feeling_emoji,
        current_weight_kg=current_weight_kg,
        current_body_fat_pct=current_body_fat_pct,
        adherence_score=adherence_score,
    )
    today_state = {
        **dict(previous_result.get("current_state", {})),
        "date": feedback_date,
        "weight_kg": current_weight_kg,
        "body_fat_pct": current_body_fat_pct,
        "notes": daily_feedback_summary(workout_feeling=feedback_notes, feeling_emoji=feeling_emoji),
    }
    tomorrow_state = {
        **dict(previous_result.get("current_state", {})),
        "date": target_date,
        "weight_kg": current_weight_kg,
        "body_fat_pct": current_body_fat_pct,
        "notes": "",
    }
    daily_entry = {
        "date": feedback_date,
        "cycle_number": cycle_number_for_feedback_date(previous_result, current_session, feedback_date),
        "plan_focus": current_session.get("focus", "No scheduled workout") if current_session else "No scheduled workout",
        "status": daily_history_status(current_session),
        "weight_kg": current_weight_kg,
        "body_fat_pct": current_body_fat_pct,
        "completed_actions": completed_actions,
        "completed_plan": deepcopy(current_session) if current_session else {},
        "feedback": {
            "workout_feeling": feedback_notes,
            "emoji": feeling_emoji,
            "emoji_label": FEELING_EMOJI_LABELS.get(feeling_emoji, ""),
            "injury_areas": history_injury_areas(current_session, latest_feedback),
        },
    }

    updated_result["current_date"] = target_date
    updated_result["current_state"] = tomorrow_state
    updated_result["latest_feedback"] = latest_feedback
    updated_result["state_history"] = append_unique_history_item(
        previous_result.get("state_history", []),
        today_state,
        "date",
    )
    updated_result["feedback_history"] = append_unique_history_item(
        previous_result.get("feedback_history", []),
        latest_feedback,
        "date",
    )
    updated_result["daily_history"] = append_unique_history_item(
        previous_result.get("daily_history", []),
        daily_entry,
        "date",
    )
    store = record_memory_daily_feedback(
        memory_store=store,
        feedback_date=feedback_date,
        daily_entry=daily_entry,
        latest_feedback=latest_feedback,
    )
    return updated_result, store


def generate_next_cycle_after_feedback(
    *,
    profile_inputs: dict[str, Any],
    previous_result: FitnessAgentState,
    feedback_date: str,
    target_date: str,
    current_weight_kg: float,
    current_body_fat_pct: float,
    workout_feeling: str,
    feeling_emoji: str,
    adherence_score: float = 1.0,
    thread_id: str,
    memory_store: dict[str, list[dict[str, Any]]] | None = None,
    session_state: dict[str, Any] | None = None,
) -> FitnessAgentState:
    store = normalize_memory_store(memory_store)
    latest_feedback = dict(previous_result.get("latest_feedback", {}))
    feedback_summary = daily_feedback_summary(workout_feeling=workout_feeling, feeling_emoji=feeling_emoji)
    latest_feedback["performance_notes"] = feedback_summary
    latest_feedback.setdefault("date", feedback_date)
    latest_feedback.setdefault("completed_actions", [])
    latest_feedback.setdefault("completed_workouts", [])
    latest_feedback.setdefault("pain_points", pain_points_from_text(workout_feeling))
    latest_feedback.setdefault("soreness_areas", soreness_areas_from_text(workout_feeling))
    fatigue_level, motivation_level, recovery_score = emoji_training_signals(feeling_emoji)
    latest_feedback.setdefault("fatigue_level", fatigue_level)
    latest_feedback.setdefault("motivation_level", motivation_level)
    latest_feedback.setdefault("recovery_score", recovery_score)
    latest_feedback.setdefault("pain_level", 5 if latest_feedback.get("pain_points") else 0)
    latest_feedback["adherence_score"] = adherence_score
    latest_feedback["manual_log"] = {
        "date": feedback_date,
        "weight_kg": current_weight_kg,
        "body_fat_pct": current_body_fat_pct,
        "notes": workout_feeling,
        "feeling_emoji": feeling_emoji,
    }

    base_state = build_initial_state(
        profile_inputs=profile_inputs,
        thread_id=thread_id,
        active_date=target_date,
        memory_store=store,
        previous_result=previous_result,
    )
    rollover_state: FitnessAgentState = {
        **base_state,
        "thread_id": thread_id,
        "current_date": target_date,
        "plan_change_request": "Generate the next cycle plan after the previous cycle ended.",
        "normalized_change_request": {},
        "current_state": {
            **dict(previous_result.get("current_state", {})),
            "date": target_date,
            "weight_kg": current_weight_kg,
            "body_fat_pct": current_body_fat_pct,
            "notes": "Generate the next cycle plan after incorporating the latest daily feedback.",
        },
        "latest_feedback": latest_feedback,
        "current_plan": previous_result.get("current_plan", {}),
        "plan_history": previous_result.get("plan_history", []),
        "daily_history": previous_result.get("daily_history", []),
        "feedback_history": previous_result.get("feedback_history", []),
        "state_history": previous_result.get("state_history", []),
        "memory_context": memory_context_for_planning(
            store,
            target_date,
            profile_inputs=profile_inputs,
            result=previous_result,
            session_state={"thread_id": thread_id, "active_date": target_date},
        ),
    }
    # RL Phase 1a: backfill reward for the cycle that just ended
    store = _backfill_last_plan_reward(store, adherence_score, feedback_date)

    next_result = run_agent(rollover_state)
    next_result["daily_history"] = list(previous_result.get("daily_history", []))

    # RL Phase 1b: record (state, action) snapshot for this new cycle
    store = _record_plan_decision(
        result=next_result,
        adherence_last_cycle=adherence_score,
        memory_store=store,
    )

    # Level upgrade: check if user has earned a promotion
    if session_state is not None:
        _maybe_upgrade_fitness_level(
            session_state=session_state,
            profile_inputs=profile_inputs,
            daily_history=list(next_result.get("daily_history", [])),
        )

    return next_result


def daily_history_status(current_session: dict[str, Any]) -> str:
    if not current_session:
        return "no_scheduled"
    if current_session.get("is_cancelled"):
        return "cancelled"
    return "completed"


def cycle_number_for_feedback_date(
    result: FitnessAgentState | dict[str, Any],
    current_session: dict[str, Any],
    feedback_date: str,
) -> int:
    if current_session:
        return history_item_cycle_number({"completed_plan": current_session})
    current_plan = result.get("current_plan", {})
    feedback_iso = safe_iso_date(feedback_date)
    cycle_start = safe_iso_date(current_plan.get("cycle_start_date"))
    cycle_end = safe_iso_date(current_plan.get("cycle_end_date"))
    try:
        feedback_day = datetime.fromisoformat(feedback_iso).date()
        start_day = datetime.fromisoformat(cycle_start).date()
        end_day = datetime.fromisoformat(cycle_end).date()
    except ValueError:
        return history_item_cycle_number({"completed_plan": current_plan})
    if start_day <= feedback_day <= end_day:
        return history_item_cycle_number({"completed_plan": current_plan})
    return history_item_cycle_number({"completed_plan": current_session})


def history_injury_areas(current_session: dict[str, Any], latest_feedback: dict[str, Any]) -> list[str]:
    session_areas = coerce_string_list(current_session.get("injury_areas"), [])
    if session_areas:
        return session_areas
    if current_session.get("injury_reported"):
        return ["reported injury area"]
    return list(latest_feedback.get("pain_points", []))


def history_item_cycle_number(item: dict[str, Any]) -> int:
    completed_plan = item.get("completed_plan", {})
    try:
        return int(item.get("cycle_number") or completed_plan.get("cycle_number") or 1)
    except (TypeError, ValueError):
        return 1


def build_daily_feedback(
    *,
    feedback_date: str,
    completed_actions: list[str],
    workout_feeling: str,
    feeling_emoji: str,
    current_weight_kg: float,
    current_body_fat_pct: float,
    adherence_score: float = 1.0,
) -> dict[str, Any]:
    fatigue_level, motivation_level, recovery_score = emoji_training_signals(feeling_emoji)
    pain_points = pain_points_from_text(workout_feeling)
    pain_level = 5 if pain_points else 0
    summary = daily_feedback_summary(workout_feeling=workout_feeling, feeling_emoji=feeling_emoji)
    return {
        "date": feedback_date,
        "completed_workouts": completed_actions,
        "completed_actions": completed_actions,
        "feeling_emoji": feeling_emoji,
        "adherence_score": adherence_score,
        "fatigue_level": fatigue_level,
        "pain_level": pain_level,
        "pain_points": pain_points,
        "soreness_areas": soreness_areas_from_text(workout_feeling),
        "motivation_level": motivation_level,
        "performance_notes": summary,
        "manual_log": {
            "date": feedback_date,
            "weight_kg": current_weight_kg,
            "body_fat_pct": current_body_fat_pct,
            "notes": workout_feeling,
            "feeling_emoji": feeling_emoji,
        },
    }


def completed_actions_from_session(session: dict[str, Any]) -> list[str]:
    if not session or session.get("is_cancelled"):
        return []
    return [
        str(exercise.get("name", "")).strip()
        for exercise in session.get("exercises", [])
        if str(exercise.get("name", "")).strip()
    ]


def emoji_training_signals(feeling_emoji: str) -> tuple[int, int, float]:
    if feeling_emoji == "😊":
        return 2, 9, 0.85
    if feeling_emoji == "😫":
        return 8, 3, 0.45
    return 5, 6, 0.7


def daily_feedback_summary(*, workout_feeling: str, feeling_emoji: str) -> str:
    label = FEELING_EMOJI_LABELS.get(feeling_emoji, "Logged")
    feeling = workout_feeling.strip()
    if feeling:
        return f"{feeling_emoji} {label}: {feeling}"
    return f"{feeling_emoji} {label}"


def pain_points_from_text(text: str) -> list[str]:
    normalized = text.lower()
    mappings = {
        "knee": ["knee", "knees", "膝盖"],
        "back": ["back", "lower back", "腰", "背"],
        "shoulder": ["shoulder", "shoulders", "肩"],
        "ankle": ["ankle", "ankles", "脚踝"],
        "hip": ["hip", "hips", "髋"],
        "wrist": ["wrist", "wrists", "手腕"],
    }
    injury_terms = ["pain", "hurt", "ache", "injury", "疼", "痛", "受伤", "拉伤", "扭伤"]
    if not any(term in normalized for term in injury_terms):
        return []
    return [
        body_part
        for body_part, aliases in mappings.items()
        if any(alias in normalized for alias in aliases)
    ] or ["reported pain"]


def soreness_areas_from_text(text: str) -> list[str]:
    normalized = text.lower()
    mappings = {
        "legs": ["legs", "quads", "hamstrings", "glutes", "腿", "臀"],
        "chest": ["chest", "胸"],
        "back": ["back", "背"],
        "arms": ["arms", "biceps", "triceps", "手臂"],
        "shoulders": ["shoulders", "肩"],
        "core": ["core", "abs", "腹", "核心"],
    }
    soreness_terms = ["sore", "soreness", "酸", "酸痛"]
    if not any(term in normalized for term in soreness_terms):
        return []
    return [
        area
        for area, aliases in mappings.items()
        if any(alias in normalized for alias in aliases)
    ]


def append_unique_history_item(history: list[dict], item: dict, date_key: str) -> list[dict]:
    if not item:
        return list(history)
    updated_history = list(history)
    item_date = item.get(date_key)
    if item_date:
        for index, existing in enumerate(updated_history):
            if same_iso_date(existing.get(date_key), item_date) or existing.get(date_key) == item_date:
                updated_history[index] = item
                return updated_history
    updated_history.append(item)
    return updated_history


def record_memory_daily_feedback(
    *,
    memory_store: dict[str, list[dict[str, Any]]],
    feedback_date: str,
    daily_entry: dict[str, Any],
    latest_feedback: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    store = append_memory_item(
        memory_store,
        "daily_feedback_records",
        {
            "date": feedback_date,
            "status": daily_entry.get("status", ""),
            "plan_focus": daily_entry.get("plan_focus", ""),
            "completed_actions": daily_entry.get("completed_actions", []),
            "feeling": daily_entry.get("feedback", {}).get("workout_feeling", ""),
            "emoji": daily_entry.get("feedback", {}).get("emoji", ""),
            "injury_areas": daily_entry.get("feedback", {}).get("injury_areas", []),
        },
        unique_key="date",
    )
    store = append_memory_item(
        store,
        "body_metrics",
        {
            "date": feedback_date,
            "weight_kg": daily_entry.get("weight_kg"),
            "body_fat_pct": daily_entry.get("body_fat_pct"),
        },
        unique_key="date",
    )
    for index, exercise_feedback in enumerate(
        exercise_feedback_items_from_daily_entry(feedback_date, daily_entry)
    ):
        store = append_memory_item(
            store,
            "exercise_feedback_records",
            exercise_feedback,
            unique_key="id",
            limit=500,
        )
    injury_areas = coerce_string_list(daily_entry.get("feedback", {}).get("injury_areas"), [])
    if injury_areas:
        for area in injury_areas:
            normalized_area = "".join(ch for ch in area.lower() if ch.isalnum()) or "reportedinjuryarea"
            store = append_memory_item(
                store,
                "injury_events",
                {
                    "id": f"{feedback_date}-{normalized_area}",
                    "date": feedback_date,
                    "area": area,
                    "risk_level": "medium",
                    "source": "daily_feedback",
                    "summary": str(latest_feedback.get("performance_notes") or "Injury or pain was reported in daily feedback."),
                    "status": "active",
                    "expires_after_days": 7,
                    "recorded_at": datetime.now().isoformat(timespec="seconds"),
                },
                unique_key="id",
            )
    return store


def exercise_feedback_items_from_daily_entry(
    feedback_date: str,
    daily_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    completed_plan = daily_entry.get("completed_plan", {})
    if not isinstance(completed_plan, dict):
        completed_plan = {}
    exercises = completed_plan.get("exercises", [])
    if not isinstance(exercises, list):
        exercises = []
    completed_names = coerce_string_list(daily_entry.get("completed_actions"), [])
    if not exercises and completed_names:
        exercises = [{"name": name} for name in completed_names]

    feedback = daily_entry.get("feedback", {})
    if not isinstance(feedback, dict):
        feedback = {}
    status = str(daily_entry.get("status") or daily_history_status(completed_plan))
    cycle_number = history_item_cycle_number(daily_entry)
    focus = str(daily_entry.get("plan_focus") or completed_plan.get("focus") or "")
    emoji = str(feedback.get("emoji") or "")
    emoji_label = str(feedback.get("emoji_label") or FEELING_EMOJI_LABELS.get(emoji, ""))
    workout_feeling = str(feedback.get("workout_feeling") or "")
    injury_areas = coerce_string_list(feedback.get("injury_areas"), [])

    rows: list[dict[str, Any]] = []
    for index, exercise in enumerate(exercises):
        if not isinstance(exercise, dict):
            continue
        name = str(exercise.get("name") or "").strip()
        if not name:
            continue
        rows.append(
            {
                "id": f"{feedback_date}-{index}-{normalize_memory_key(name)}",
                "date": feedback_date,
                "cycle_number": cycle_number,
                "exercise_name": name,
                "focus": focus,
                "sets": exercise.get("sets"),
                "reps": exercise.get("reps"),
                "status": "cancelled" if status == "cancelled" else "completed",
                "feeling_emoji": emoji,
                "emoji_label": emoji_label,
                "workout_feeling": workout_feeling,
                "injury_areas": injury_areas,
                "source": "daily_feedback",
            }
        )
    if not rows and status in {"cancelled", "no_scheduled"}:
        rows.append(
            {
                "id": f"{feedback_date}-0-{status}",
                "date": feedback_date,
                "cycle_number": cycle_number,
                "exercise_name": "",
                "focus": focus,
                "sets": None,
                "reps": "",
                "status": status,
                "feeling_emoji": emoji,
                "emoji_label": emoji_label,
                "workout_feeling": workout_feeling,
                "injury_areas": injury_areas,
                "source": "daily_feedback",
            }
        )
    return rows


def same_iso_date(left: object, right: object) -> bool:
    left_date = safe_iso_date(left)
    right_date = safe_iso_date(right)
    return bool(left_date and right_date and left_date == right_date)


def coerce_string_list(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return fallback
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    return cleaned or fallback


def normalize_memory_key(value: str) -> str:
    normalized = "".join(char.lower() for char in value if char.isalnum())
    return normalized or "item"


# ---------------------------------------------------------------------------
# RL Phase 1: plan decision snapshot + level upgrade
# ---------------------------------------------------------------------------

_LEVEL_ORDER = ["beginner", "intermediate", "advanced"]
_UPGRADE_CYCLES = {"beginner": 3, "intermediate": 5}
_DOWNGRADE_CYCLES = {"intermediate": 2, "advanced": 2}


def _backfill_last_plan_reward(
    memory_store: dict[str, list[dict[str, Any]]],
    reward: float,
    cycle_end_date: str,
) -> dict[str, list[dict[str, Any]]]:
    decisions = memory_store.get("plan_decisions", [])
    for entry in reversed(decisions):
        if entry.get("reward") is None:
            entry["reward"] = reward
            entry["cycle_end_date"] = cycle_end_date
            break
    return memory_store


def _record_plan_decision(
    result: FitnessAgentState | dict[str, Any],
    adherence_last_cycle: float,
    memory_store: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    current_plan = result.get("current_plan", {})
    user_profile = result.get("user_profile", {})
    goals = result.get("goals", {})
    current_state = result.get("current_state", {})
    sessions = current_plan.get("workout_sessions", [])

    focus_counts: dict[str, int] = {}
    for session in sessions:
        focus = str(session.get("focus", "")).strip()
        if focus and not session.get("is_cancelled"):
            focus_counts[focus] = focus_counts.get(focus, 0) + 1
    total = max(len(sessions), 1)

    entry = {
        "decision_id": str(uuid4()),
        "timestamp": str(datetime.today().date()),
        "state": {
            "weight_kg": current_state.get("weight_kg"),
            "body_fat_pct": current_state.get("body_fat_pct"),
            "adherence_last_cycle": adherence_last_cycle,
            "goal": goals.get("primary_goal"),
            "fitness_level": user_profile.get("fitness_level"),
            "active_injuries": [
                str(inj.get("area") or inj.get("areas") or "")
                for inj in memory_store.get("injury_events", [])
                if isinstance(inj, dict) and str(inj.get("status", "active")) == "active"
            ],
        },
        "action": {
            "sessions_per_week": len(sessions),
            "avg_duration_minutes": int(
                sum(s.get("duration_minutes", 60) for s in sessions) / total
            ),
            "focus_distribution": {
                k: round(v / total, 2) for k, v in focus_counts.items()
            },
            "avg_sets_per_session": round(
                sum(
                    sum(e.get("sets", 4) for e in s.get("exercises", []))
                    for s in sessions
                ) / max(sum(len(s.get("exercises", [])) for s in sessions), 1),
                1,
            ),
            "daily_calories": current_plan.get("nutrition_targets", {}).get("daily_calories"),
        },
        "reward": None,
        "cycle_end_date": None,
    }
    return append_memory_item(memory_store, "plan_decisions", entry)


def _cycle_completion_results(
    daily_history: list[dict[str, Any]],
    planned_per_cycle: int,
) -> list[bool]:
    """Per-cycle completion bool: True if completed sessions >= planned_per_cycle.

    Groups daily_history by cycle_number, counts status=="completed" entries,
    then drops the last (possibly in-progress) cycle before returning.
    """
    cycle_completed: dict[int, int] = {}
    for entry in daily_history:
        if not isinstance(entry, dict):
            continue
        cycle = entry.get("cycle_number")
        if cycle is None:
            continue
        if str(entry.get("status", "")) == "completed":
            cycle_completed[int(cycle)] = cycle_completed.get(int(cycle), 0) + 1

    if not cycle_completed:
        return []
    sorted_cycles = sorted(cycle_completed.keys())[:-1]  # drop current/incomplete cycle
    return [cycle_completed[c] >= planned_per_cycle for c in sorted_cycles]


def _maybe_upgrade_fitness_level(
    session_state: dict[str, Any],
    profile_inputs: dict[str, Any],
    daily_history: list[dict[str, Any]],
) -> None:
    current_level = str(profile_inputs.get("fitness_level", "beginner")).lower()
    if current_level not in _LEVEL_ORDER:
        return

    planned = int(profile_inputs.get("sessions_per_week", 3))
    results = _cycle_completion_results(daily_history, planned)
    if not results:
        return

    # Upgrade: last N cycles all met target
    if current_level in _UPGRADE_CYCLES:
        n = _UPGRADE_CYCLES[current_level]
        if len(results) >= n and all(results[-n:]):
            idx = _LEVEL_ORDER.index(current_level)
            new_level = _LEVEL_ORDER[idx + 1]
            profile_inputs["fitness_level"] = new_level
            session_state["profile_inputs"] = profile_inputs
            session_state["last_action_message"] = (
                f"Great progress! Your training level has been upgraded to {new_level}."
            )
            return

    # Downgrade: last N cycles all failed
    if current_level in _DOWNGRADE_CYCLES:
        n = _DOWNGRADE_CYCLES[current_level]
        if len(results) >= n and not any(results[-n:]):
            idx = _LEVEL_ORDER.index(current_level)
            new_level = _LEVEL_ORDER[idx - 1]
            profile_inputs["fitness_level"] = new_level
            session_state["profile_inputs"] = profile_inputs
            session_state["last_action_message"] = (
                f"Training level adjusted to {new_level} — "
                "let's build consistency before increasing intensity."
            )
