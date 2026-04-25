"""Planner node for generating a first-pass fitness plan."""

from __future__ import annotations

from copy import deepcopy
import json
from datetime import date, datetime, timedelta
from uuid import uuid4

from agent.llm import call_model_json, load_prompt
from agent.state import FitnessAgentState, FitnessPlan, MealSuggestion, WorkoutSession
from agent.tools import (
    build_video_resources,
    calculate_food_macros,
    find_exercises,
    find_foods,
    get_exercise_by_name,
    get_food_by_name,
)


DEFAULT_DAYS = ["Monday", "Tuesday", "Thursday", "Saturday"]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_INDEX = {day: index for index, day in enumerate(WEEKDAY_ORDER)}
GOAL_LABELS = {
    "weight_loss": "减重",
    "strength": "力量",
    "sculpting": "塑形",
}
FOCUS_LIBRARY = {
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
GOAL_FOCUS_TEMPLATES = {
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
FOCUS_ALIASES = {
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


def plan_generation_node(state: FitnessAgentState) -> FitnessAgentState:
    """Build a structured workout and nutrition plan from the current state."""

    user_profile = state.get("user_profile", {})
    constraints = state.get("constraints", {})
    goals = state.get("goals", {})
    current_state = state.get("current_state", {})
    latest_feedback = state.get("latest_feedback", {})

    sessions_per_week = max(3, min(int(constraints.get("sessions_per_week", 3)), 5))
    cycle_slots = _resolve_cycle_slots(
        constraints=constraints,
        current_date=state.get("current_date", ""),
        sessions_per_week=sessions_per_week,
    )
    fitness_level = str(user_profile.get("fitness_level", "beginner"))
    training_goal = _map_goal_tag(str(goals.get("primary_goal", "weight_loss")))
    excluded_conditions = _collect_excluded_conditions(constraints, latest_feedback)
    equipment_access = constraints.get("equipment_access", ["bodyweight"])
    user_notes = _build_context_notes(state)
    session_focuses = _select_focuses(sessions_per_week, training_goal)
    hard_stop_context = _build_hard_stop_context(state)
    cancel_today = bool(hard_stop_context.get("cancel"))
    requested_focus = None if cancel_today else _requested_focus_from_state(state)
    intensity_adjustment = "" if cancel_today else _intensity_adjustment_from_state(state)
    target_date = _safe_date_string(state.get("current_date", ""))
    session_focuses = _apply_change_request_focus_override(
        session_focuses=session_focuses,
        cycle_slots=cycle_slots,
        current_date=str(state.get("current_date", "")),
        requested_focus=requested_focus,
    )

    cycle_blueprints = [
        _build_session_blueprint(
            day=slot["day"],
            scheduled_date=slot["scheduled_date"],
            cycle_number=slot["cycle_number"],
            cycle_session_index=slot["cycle_session_index"],
            focus=focus,
            duration_minutes=int(constraints.get("minutes_per_session", 60)),
            fitness_level=fitness_level,
            training_goal=training_goal,
            equipment_access=equipment_access,
            excluded_conditions=excluded_conditions,
            excluded_exercises=constraints.get("excluded_exercises", []),
            exercise_count_delta=_exercise_count_delta_for_intensity(
                fitness_level,
                intensity_adjustment if slot["scheduled_date"] == target_date else "",
            ),
            intensity_adjustment=intensity_adjustment if slot["scheduled_date"] == target_date else "",
        )
        for slot, focus in zip(cycle_slots, session_focuses)
    ]
    session_blueprints = _with_ad_hoc_today_blueprint(
        cycle_blueprints=cycle_blueprints,
        current_date=str(state.get("current_date", "")),
        requested_focus=requested_focus,
        constraints=constraints,
        fitness_level=fitness_level,
        training_goal=training_goal,
        equipment_access=equipment_access,
        excluded_conditions=excluded_conditions,
        exercise_count_delta=_exercise_count_delta_for_intensity(fitness_level, intensity_adjustment),
        intensity_adjustment=intensity_adjustment,
    )

    nutrition_candidate_pool = _build_food_candidates(constraints)
    fallback_nutrition_targets = _build_nutrition_targets(user_profile, goals, current_state)
    fallback_meal_suggestions = _build_meal_suggestions(
        goals=goals,
        constraints=constraints,
        nutrition_targets=fallback_nutrition_targets,
    )
    fallback_coaching_focus = _build_coaching_focus(latest_feedback, training_goal)
    fallback_recovery_actions = _build_recovery_actions(current_state, latest_feedback, excluded_conditions)
    planning_current_state = _strip_body_metrics(current_state)
    planning_latest_feedback = _strip_body_metrics_from_feedback(latest_feedback)

    model_plan = call_model_json(
        system_prompt=load_prompt("planner_prompt.txt"),
        user_prompt=json.dumps(
            {
                "user_profile": user_profile,
                "constraints": constraints,
                "goals": goals,
                "profile_notes": state.get("profile_notes", ""),
                "plan_change_request": state.get("plan_change_request", ""),
                "normalized_change_request": state.get("normalized_change_request", {}),
                "current_day_name": _resolve_current_day_name(state.get("current_date", "")),
                "current_state": planning_current_state,
                "latest_feedback": planning_latest_feedback,
                "existing_weekly_plan": state.get("current_plan", {}).get("workout_sessions", []),
                "cycle_context": {
                    "cycle_number": cycle_slots[0]["cycle_number"] if cycle_slots else 1,
                    "cycle_start_date": cycle_slots[0]["cycle_start_date"] if cycle_slots else state.get("current_date", ""),
                    "cycle_end_date": cycle_slots[0]["cycle_end_date"] if cycle_slots else state.get("current_date", ""),
                },
                "session_blueprints": session_blueprints,
                "food_candidates": nutrition_candidate_pool,
                "fallback_nutrition_targets": fallback_nutrition_targets,
            },
            ensure_ascii=True,
            indent=2,
        ),
        temperature=0.3,
        max_tokens=4000,
    )

    workout_sessions = [
        _finalize_workout_session(
            blueprint=blueprint,
            session_payload=payload,
            fitness_level=fitness_level,
            training_goal=training_goal,
        )
        for blueprint, payload in zip_longest_with_last(
            session_blueprints,
            model_plan.get("workout_sessions", []),
        )
    ]
    if cancel_today:
        workout_sessions = _preserve_other_sessions_for_today_cancellation(
            workout_sessions=workout_sessions,
            previous_plan=state.get("current_plan", {}),
            current_date=str(state.get("current_date", "")),
        )
        workout_sessions = _apply_today_injury_cancellation(
            workout_sessions=workout_sessions,
            current_date=str(state.get("current_date", "")),
            cancellation_context=hard_stop_context,
        )
    else:
        workout_sessions = _preserve_other_sessions_for_targeted_update(
            workout_sessions=workout_sessions,
            previous_plan=state.get("current_plan", {}),
            target_date=target_date,
            has_targeted_update=bool(state.get("plan_change_request") or state.get("normalized_change_request")),
        )
        workout_sessions = _resolve_same_cycle_focus_conflict(
            workout_sessions=workout_sessions,
            previous_plan=state.get("current_plan", {}),
            target_date=target_date,
            requested_focus=requested_focus,
            state=state,
        )
    workout_sessions = _sort_workout_sessions(workout_sessions)
    nutrition_targets = _finalize_nutrition_targets(
        model_plan.get("nutrition_targets", {}),
        fallback_nutrition_targets,
    )
    meal_suggestions = _finalize_meal_suggestions(
        model_plan.get("meal_suggestions", []),
        nutrition_candidate_pool,
        fallback_meal_suggestions,
    )
    coaching_focus = _coerce_string_list(
        model_plan.get("coaching_focus", []),
        fallback_coaching_focus,
    )
    recovery_actions = _coerce_string_list(
        model_plan.get("recovery_actions", []),
        fallback_recovery_actions,
    )

    current_plan: FitnessPlan = {
        "plan_id": str(uuid4()),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "cycle_number": cycle_slots[0]["cycle_number"] if cycle_slots else 1,
        "cycle_start_date": cycle_slots[0]["cycle_start_date"] if cycle_slots else state.get("current_date", ""),
        "cycle_end_date": cycle_slots[0]["cycle_end_date"] if cycle_slots else state.get("current_date", ""),
        "summary": str(model_plan.get("summary") or _build_summary(sessions_per_week, goals, constraints)),
        "objective_alignment": str(
            model_plan.get("objective_alignment") or _build_objective_alignment(goals, user_notes)
        ),
        "workout_sessions": workout_sessions,
        "nutrition_targets": nutrition_targets,
        "meal_suggestions": meal_suggestions,
        "recovery_actions": recovery_actions,
        "coaching_focus": coaching_focus,
    }

    exercise_names = [
        exercise["name"]
        for session in workout_sessions
        for exercise in session.get("exercises", [])
    ]
    youtube_resources = build_video_resources(exercise_names)

    updated_plan_history = list(state.get("plan_history", []))
    previous_plan = state.get("current_plan")
    if previous_plan and previous_plan.get("plan_id"):
        updated_plan_history.append(previous_plan)

    return {
        "current_plan": current_plan,
        "youtube_resources": youtube_resources,
        "coaching_message": _build_hard_stop_message(hard_stop_context)
        if cancel_today
        else str(
            model_plan.get("coaching_message")
            or _build_coaching_message(goals, coaching_focus, recovery_actions)
        ),
        "needs_revision": False,
        "revision_reason": "",
        "plan_history": updated_plan_history,
    }


def _resolve_session_days(constraints: dict, sessions_per_week: int) -> list[str]:
    available_days = _sort_days(constraints.get("available_days") or DEFAULT_DAYS)
    if len(available_days) < sessions_per_week:
        for day in DEFAULT_DAYS:
            if day not in available_days:
                available_days.append(day)
            if len(available_days) == sessions_per_week:
                break
    return _sort_days(available_days)


def _resolve_cycle_slots(*, constraints: dict, current_date: str, sessions_per_week: int) -> list[dict]:
    available_days = _resolve_session_days(constraints, sessions_per_week)
    start_date = _safe_date(constraints.get("program_start_date")) or _safe_date(current_date) or date.today()
    reference_date = _safe_date(current_date) or start_date
    if reference_date < start_date:
        cycle_index = 0
    else:
        cycle_index = (reference_date - start_date).days // 7
    cycle_start_date = start_date + timedelta(days=cycle_index * 7)
    cycle_end_date = cycle_start_date + timedelta(days=6)
    target_weekdays = {WEEKDAY_INDEX[day] for day in available_days}
    cycle_dates = [
        cycle_start_date + timedelta(days=offset)
        for offset in range(7)
        if (cycle_start_date + timedelta(days=offset)).weekday() in target_weekdays
    ][:sessions_per_week]

    return [
        {
            "day": cycle_date.strftime("%A"),
            "scheduled_date": cycle_date.isoformat(),
            "cycle_number": cycle_index + 1,
            "cycle_session_index": index + 1,
            "cycle_start_date": cycle_start_date.isoformat(),
            "cycle_end_date": cycle_end_date.isoformat(),
        }
        for index, cycle_date in enumerate(cycle_dates)
    ]


def _generate_training_dates(*, start_date: date, available_days: list[str], count: int) -> list[date]:
    target_weekdays = {WEEKDAY_INDEX[day] for day in available_days}
    training_dates: list[date] = []
    cursor = start_date
    max_days = 365
    while len(training_dates) < count and max_days > 0:
        if cursor.weekday() in target_weekdays:
            training_dates.append(cursor)
        cursor += timedelta(days=1)
        max_days -= 1
    return training_dates


def _safe_date(value: object) -> date | None:
    try:
        return datetime.fromisoformat(str(value)).date()
    except (TypeError, ValueError):
        return None


def _select_focuses(sessions_per_week: int, training_goal: str) -> list[str]:
    template = GOAL_FOCUS_TEMPLATES.get(training_goal, GOAL_FOCUS_TEMPLATES["weight_loss"])
    return template[:sessions_per_week]


def _apply_change_request_focus_override(
    *,
    session_focuses: list[str],
    cycle_slots: list[dict],
    current_date: str,
    requested_focus: str | None,
) -> list[str]:
    if not requested_focus or not session_focuses or not cycle_slots:
        return session_focuses

    current_date_iso = _safe_date_string(current_date)
    current_index = next(
        (index for index, slot in enumerate(cycle_slots) if slot.get("scheduled_date") == current_date_iso),
        None,
    )
    if current_index is None or current_index >= len(session_focuses):
        return session_focuses

    updated_focuses = list(session_focuses)
    updated_focuses[current_index] = requested_focus
    return updated_focuses


def _first_unused_focus(used_focuses: set[str]) -> str:
    for focus_key in FOCUS_LIBRARY:
        if focus_key not in used_focuses:
            return focus_key
    return "functional_conditioning"


def _requested_focus_from_state(state: FitnessAgentState) -> str | None:
    normalized_focus = _requested_focus_from_normalized_change_request(state.get("normalized_change_request", {}))
    if normalized_focus:
        return normalized_focus
    return _requested_focus_from_change_request(str(state.get("plan_change_request", "")))


def _intensity_adjustment_from_state(state: FitnessAgentState) -> str:
    normalized_change_request = state.get("normalized_change_request", {})
    normalized_intensity = str(normalized_change_request.get("intensity_adjustment", "")).strip().lower()
    if normalized_intensity in {"higher", "lower"}:
        return normalized_intensity

    text = " ".join(
        str(part)
        for part in [
            state.get("plan_change_request", ""),
            state.get("latest_feedback", {}).get("performance_notes", ""),
            state.get("latest_feedback", {}).get("manual_log", {}).get("notes", ""),
            state.get("current_state", {}).get("notes", ""),
        ]
        if part
    ).lower()
    return _intensity_adjustment_from_text(text)


def _intensity_adjustment_from_text(text: str) -> str:
    higher_terms = [
        "higher",
        "increase",
        "harder",
        "more intense",
        "more intensity",
        "add intensity",
        "add exercise",
        "add one exercise",
        "more exercises",
        "challenge",
        "push harder",
        "excited",
        "feel good",
        "feeling good",
        "great",
        "strong",
        "energetic",
        "too easy",
        "ready for more",
        "加强",
        "加大强度",
        "加量",
        "加一个动作",
        "状态很好",
        "感觉很好",
        "很兴奋",
        "精力很好",
        "太简单",
    ]
    lower_terms = [
        "lower",
        "reduce",
        "decrease",
        "easier",
        "lighter",
        "less intense",
        "less intensity",
        "shorter",
        "too hard",
        "too tired",
        "not good",
        "uncomfortable",
        "tired",
        "fatigued",
        "exhausted",
        "low energy",
        "drained",
        "struggling",
        "不舒服",
        "降低强度",
        "减量",
        "轻一点",
        "少一点",
        "状态不好",
        "不太好",
        "太累",
    ]
    if any(term in text for term in lower_terms):
        return "lower"
    if any(term in text for term in higher_terms):
        return "higher"
    return ""


def _exercise_count_delta_for_intensity(fitness_level: str, intensity_adjustment: str) -> int:
    if intensity_adjustment == "higher":
        return 1
    if intensity_adjustment == "lower" and fitness_level != "beginner":
        return -1
    return 0


def _build_hard_stop_context(state: FitnessAgentState) -> dict:
    current_state = state.get("current_state", {})
    latest_feedback = state.get("latest_feedback", {})
    normalized_change_request = state.get("normalized_change_request", {})

    sleep_hours = _safe_float(current_state.get("sleep_hours"), 7.0)
    injury_areas = _coerce_string_list(normalized_change_request.get("injury_areas"), [])
    feedback_pain_points = _coerce_string_list(latest_feedback.get("pain_points"), [])
    if not injury_areas:
        injury_areas = feedback_pain_points

    notes_blob = " ".join(
        str(part)
        for part in [
            state.get("plan_change_request", ""),
            current_state.get("notes", ""),
            latest_feedback.get("performance_notes", ""),
            latest_feedback.get("manual_log", {}).get("notes", ""),
            " ".join(_coerce_string_list(latest_feedback.get("completed_workouts"), [])),
            " ".join(feedback_pain_points),
        ]
        if part
    ).lower()
    injury_reported = bool(
        normalized_change_request.get("injury_reported")
        or _contains_injury_language(notes_blob)
    )
    sleep_hard_stop = sleep_hours < 5.0

    reasons = []
    if sleep_hard_stop:
        reasons.append(f"Sleep was only {sleep_hours:.1f} hours, below the 5-hour safety threshold.")
    if injury_reported:
        area_text = ", ".join(injury_areas) if injury_areas else "the reported injury area"
        reasons.append(f"Injury was reported around {area_text}.")

    return {
        "cancel": bool(sleep_hard_stop or injury_reported),
        "sleep_hours": sleep_hours,
        "sleep_hard_stop": sleep_hard_stop,
        "injury_reported": injury_reported,
        "injury_areas": injury_areas,
        "reasons": reasons,
    }


def _contains_injury_language(text: str) -> bool:
    negated_terms = [
        "no injury",
        "not injured",
        "no pain",
        "pain free",
        "pain-free",
        "doesn't hurt",
        "does not hurt",
        "feels fine",
        "recovered",
        "恢复了",
        "不疼",
        "没有受伤",
    ]
    if any(term in text for term in negated_terms):
        return False

    injury_terms = [
        "injured",
        "injury",
        "hurt",
        "hurts",
        "pain",
        "painful",
        "ache",
        "strain",
        "strained",
        "sprain",
        "sprained",
        "pulled",
        "受伤",
        "疼",
        "痛",
        "拉伤",
        "扭伤",
    ]
    return any(term in text for term in injury_terms)


def _safe_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _strip_body_metrics(current_state: dict) -> dict:
    planning_state = dict(current_state)
    planning_state.pop("weight_kg", None)
    planning_state.pop("body_fat_pct", None)
    return planning_state


def _strip_body_metrics_from_feedback(latest_feedback: dict) -> dict:
    planning_feedback = deepcopy(latest_feedback)
    manual_log = dict(planning_feedback.get("manual_log", {}))
    manual_log.pop("weight_kg", None)
    manual_log.pop("body_fat_pct", None)
    if manual_log:
        planning_feedback["manual_log"] = manual_log
    else:
        planning_feedback.pop("manual_log", None)
    return planning_feedback


def _should_cancel_today_for_injury(state: FitnessAgentState) -> bool:
    normalized_change_request = state.get("normalized_change_request", {})
    if normalized_change_request.get("cancel_today") or normalized_change_request.get("injury_reported"):
        return True

    # Safety fallback for cases where model normalization fails or is unavailable.
    request_text = str(state.get("plan_change_request", "")).lower()
    injury_terms = [
        "injured",
        "injury",
        "hurt",
        "hurts",
        "pain",
        "strain",
        "strained",
        "sprain",
        "sprained",
        "pulled",
    ]
    return any(term in request_text for term in injury_terms)


def _requested_focus_from_normalized_change_request(normalized_change_request: dict) -> str | None:
    focus_category = str(normalized_change_request.get("focus_category", "")).strip()
    if focus_category in FOCUS_LIBRARY:
        return focus_category
    return None


def _with_ad_hoc_today_blueprint(
    *,
    cycle_blueprints: list[dict],
    current_date: str,
    requested_focus: str | None,
    constraints: dict,
    fitness_level: str,
    training_goal: str,
    equipment_access: list[str],
    excluded_conditions: list[str],
    exercise_count_delta: int,
    intensity_adjustment: str = "",
) -> list[dict]:
    current_date_iso = _safe_date_string(current_date)
    if not requested_focus or not current_date_iso:
        return cycle_blueprints
    if any(blueprint.get("scheduled_date") == current_date_iso for blueprint in cycle_blueprints):
        return cycle_blueprints

    parsed_date = _safe_date(current_date_iso)
    if parsed_date is None:
        return cycle_blueprints

    cycle_number = int(cycle_blueprints[0].get("cycle_number", 1)) if cycle_blueprints else 1
    ad_hoc_blueprint = _build_session_blueprint(
        day=parsed_date.strftime("%A"),
        scheduled_date=current_date_iso,
        cycle_number=cycle_number,
        cycle_session_index=0,
        focus=requested_focus,
        duration_minutes=int(constraints.get("minutes_per_session", 60)),
        fitness_level=fitness_level,
        training_goal=training_goal,
        equipment_access=equipment_access,
        excluded_conditions=excluded_conditions,
        excluded_exercises=constraints.get("excluded_exercises", []),
        exercise_count_delta=exercise_count_delta,
        intensity_adjustment=intensity_adjustment,
    )
    ad_hoc_blueprint["is_ad_hoc"] = True
    return [ad_hoc_blueprint, *cycle_blueprints]


def _apply_today_injury_cancellation(
    *,
    workout_sessions: list[WorkoutSession],
    current_date: str,
    cancellation_context: dict,
) -> list[WorkoutSession]:
    current_date_iso = _safe_date_string(current_date)
    if not current_date_iso:
        return workout_sessions

    cancelled_session = _build_cancelled_today_session(
        current_date=current_date_iso,
        cancellation_context=cancellation_context,
    )
    updated_sessions: list[WorkoutSession] = []
    replaced_existing_session = False
    for session in workout_sessions:
        if session.get("scheduled_date") == current_date_iso:
            replacement = dict(cancelled_session)
            replacement["day"] = session.get("day", replacement["day"])
            replacement["cycle_number"] = session.get("cycle_number", replacement["cycle_number"])
            replacement["cycle_session_index"] = session.get("cycle_session_index", replacement["cycle_session_index"])
            replacement["is_ad_hoc"] = bool(session.get("is_ad_hoc", False))
            updated_sessions.append(replacement)
            replaced_existing_session = True
        else:
            updated_sessions.append(session)

    if not replaced_existing_session:
        updated_sessions.insert(0, cancelled_session)
    return updated_sessions


def _preserve_other_sessions_for_today_cancellation(
    *,
    workout_sessions: list[WorkoutSession],
    previous_plan: dict,
    current_date: str,
) -> list[WorkoutSession]:
    previous_sessions = previous_plan.get("workout_sessions", [])
    if not previous_sessions:
        return workout_sessions

    current_date_iso = _safe_date_string(current_date)
    previous_by_date = {
        session.get("scheduled_date", ""): session
        for session in previous_sessions
        if session.get("scheduled_date")
    }
    preserved_sessions: list[WorkoutSession] = []
    for session in workout_sessions:
        scheduled_date = session.get("scheduled_date", "")
        if scheduled_date == current_date_iso:
            preserved_sessions.append(session)
            continue
        previous_session = previous_by_date.get(scheduled_date)
        preserved_sessions.append(deepcopy(previous_session) if previous_session else session)
    return preserved_sessions


def _preserve_other_sessions_for_targeted_update(
    *,
    workout_sessions: list[WorkoutSession],
    previous_plan: dict,
    target_date: str,
    has_targeted_update: bool,
) -> list[WorkoutSession]:
    if not has_targeted_update or not previous_plan or not target_date:
        return workout_sessions

    previous_by_date = {
        session.get("scheduled_date", ""): session
        for session in previous_plan.get("workout_sessions", [])
        if session.get("scheduled_date")
    }
    if not previous_by_date:
        return workout_sessions

    preserved_sessions: list[WorkoutSession] = []
    for session in workout_sessions:
        scheduled_date = str(session.get("scheduled_date", ""))
        if scheduled_date == target_date:
            preserved_sessions.append(session)
            continue
        previous_session = previous_by_date.get(scheduled_date)
        preserved_sessions.append(deepcopy(previous_session) if previous_session else session)
    return preserved_sessions


def _resolve_same_cycle_focus_conflict(
    *,
    workout_sessions: list[WorkoutSession],
    previous_plan: dict,
    target_date: str,
    requested_focus: str | None,
    state: FitnessAgentState,
) -> list[WorkoutSession]:
    if not requested_focus or not previous_plan or not target_date:
        return workout_sessions

    change_mode = _change_request_mode(state)
    if change_mode not in {"add", "replace"}:
        return workout_sessions

    target_focus_label = _focus_label(_focus_key_from_value(requested_focus))
    duplicate_session = _find_same_cycle_duplicate_focus_session(
        sessions=workout_sessions,
        target_date=target_date,
        target_focus_label=target_focus_label,
    )
    if not duplicate_session:
        return workout_sessions

    if change_mode == "add":
        return _replace_session_on_date(
            workout_sessions,
            str(duplicate_session.get("scheduled_date", "")),
            _build_rest_session_from_session(
                duplicate_session,
                reason=f"{target_focus_label} was added temporarily on {target_date}, so this duplicate session becomes recovery.",
            ),
        )

    previous_target_session = _session_for_date(
        previous_plan.get("workout_sessions", []),
        target_date,
    )
    if not previous_target_session:
        return _replace_session_on_date(
            workout_sessions,
            str(duplicate_session.get("scheduled_date", "")),
            _build_rest_session_from_session(
                duplicate_session,
                reason=f"{target_focus_label} moved to {target_date}, so this duplicate session becomes recovery.",
            ),
        )

    swapped_session = _copy_session_onto_slot(previous_target_session, duplicate_session)
    return _replace_session_on_date(
        workout_sessions,
        str(duplicate_session.get("scheduled_date", "")),
        swapped_session,
    )


def _find_same_cycle_duplicate_focus_session(
    *,
    sessions: list[WorkoutSession],
    target_date: str,
    target_focus_label: str,
) -> WorkoutSession | None:
    target_session = _session_for_date(sessions, target_date)
    if not target_session:
        return None

    target_cycle = target_session.get("cycle_number")
    for session in sessions:
        scheduled_date = str(session.get("scheduled_date", ""))
        if scheduled_date == target_date:
            continue
        if session.get("is_ad_hoc") or session.get("is_cancelled"):
            continue
        if target_cycle and session.get("cycle_number") != target_cycle:
            continue
        if str(session.get("focus", "")) == target_focus_label:
            return session
    return None


def _change_request_mode(state: FitnessAgentState) -> str:
    normalized_change_request = state.get("normalized_change_request", {})
    if normalized_change_request.get("cancel_today") or normalized_change_request.get("injury_reported"):
        return "cancel"

    request_text = str(state.get("plan_change_request", "")).lower()
    temporary_add_terms = [
        "temporarily add",
        "temporary add",
        "add",
        "extra",
        "also do",
        "can i do",
        "can i add",
        "could i do",
        "could i add",
        "加练",
        "加一个",
        "加一节",
        "临时加",
        "加",
    ]
    replace_terms = [
        "replace",
        "switch",
        "swap",
        "change today's",
        "change today",
        "instead",
        "更换",
        "替换",
        "换成",
        "改成",
        "换",
        "改",
    ]
    if any(term in request_text for term in replace_terms):
        return "replace"
    if any(term in request_text for term in temporary_add_terms):
        return "add"
    if normalized_change_request.get("focus_category"):
        return "replace"
    return ""


def _session_for_date(sessions: list[dict], target_date: str) -> dict | None:
    for session in sessions:
        if str(session.get("scheduled_date", "")) == target_date:
            return session
    return None


def _replace_session_on_date(
    sessions: list[WorkoutSession],
    target_date: str,
    replacement: WorkoutSession,
) -> list[WorkoutSession]:
    return [
        deepcopy(replacement) if str(session.get("scheduled_date", "")) == target_date else session
        for session in sessions
    ]


def _copy_session_onto_slot(source_session: dict, slot_session: dict) -> WorkoutSession:
    copied = deepcopy(source_session)
    copied["day"] = slot_session.get("day", copied.get("day", ""))
    copied["scheduled_date"] = slot_session.get("scheduled_date", copied.get("scheduled_date", ""))
    copied["cycle_number"] = slot_session.get("cycle_number", copied.get("cycle_number", 1))
    copied["cycle_session_index"] = slot_session.get("cycle_session_index", copied.get("cycle_session_index", 1))
    copied["is_ad_hoc"] = bool(slot_session.get("is_ad_hoc", False))
    return copied


def _build_rest_session_from_session(session: dict, *, reason: str) -> WorkoutSession:
    return {
        "day": str(session.get("day", "")),
        "scheduled_date": str(session.get("scheduled_date", "")),
        "cycle_number": int(session.get("cycle_number", 1)),
        "cycle_session_index": int(session.get("cycle_session_index", 1)),
        "is_ad_hoc": bool(session.get("is_ad_hoc", False)),
        "is_cancelled": True,
        "focus": "Recovery / Rest",
        "duration_minutes": 0,
        "warmup": [],
        "exercises": [],
        "cooldown": [],
        "safety_notes": [reason],
    }


def _build_cancelled_today_session(*, current_date: str, cancellation_context: dict) -> WorkoutSession:
    parsed_date = _safe_date(current_date) or date.today()
    reasons = _coerce_string_list(cancellation_context.get("reasons"), [])
    if not reasons:
        reasons = ["Training is cancelled by a hard safety rule."]
    return {
        "day": parsed_date.strftime("%A"),
        "scheduled_date": current_date,
        "cycle_number": 1,
        "cycle_session_index": 0,
        "is_ad_hoc": True,
        "is_cancelled": True,
        "focus": "Workout Cancelled",
        "duration_minutes": 0,
        "warmup": [],
        "exercises": [],
        "cooldown": [],
        "safety_notes": [
            "Today's workout is cancelled.",
            *reasons,
            "Rest, avoid training through pain or heavy fatigue, and update your status before resuming training.",
        ],
    }


def _build_hard_stop_message(cancellation_context: dict) -> str:
    reasons = _coerce_string_list(cancellation_context.get("reasons"), [])
    reason_text = " ".join(reasons) if reasons else "A hard safety rule was triggered."
    return (
        f"Today's workout is cancelled. {reason_text} "
        "Rest today and update your sleep, pain, or injury status before resuming training."
    )


def _requested_focus_from_change_request(plan_change_request: str) -> str | None:
    normalized_request = plan_change_request.strip().lower().replace("_", " ")
    if not normalized_request:
        return None

    for alias in sorted(FOCUS_ALIASES, key=len, reverse=True):
        if alias in normalized_request:
            return FOCUS_ALIASES[alias]
    return None


def _map_goal_tag(primary_goal: str) -> str:
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


def _collect_excluded_conditions(constraints: dict, latest_feedback: dict) -> list[str]:
    injuries = list(constraints.get("injuries", []))
    pain_sensitive_areas = list(constraints.get("pain_sensitive_areas", []))
    pain_points = list(latest_feedback.get("pain_points", []))
    return injuries + pain_sensitive_areas + pain_points


def _build_workout_session(
    *,
    day: str,
    scheduled_date: str,
    cycle_number: int,
    cycle_session_index: int,
    focus: str,
    duration_minutes: int,
    fitness_level: str,
    training_goal: str,
    equipment_access: list[str],
    excluded_conditions: list[str],
    excluded_exercises: list[str],
    exercise_count_delta: int = 0,
) -> WorkoutSession:
    """Legacy heuristic session builder retained as a fallback template."""

    focus_key = _focus_key_from_value(focus)
    target_muscles, movement_type = _focus_to_targets(focus_key)
    candidate_exercises = find_exercises(
        target_muscles=target_muscles,
        movement_type=movement_type,
        difficulty=fitness_level,
        available_equipment=equipment_access,
        training_goal=training_goal,
        excluded_conditions=excluded_conditions,
        recommended_for=_recommended_program_tags(equipment_access, excluded_conditions),
        focus_tags=[focus_key],
        limit=_exercise_count_for_level(fitness_level, exercise_count_delta),
    )
    filtered_exercises = [
        exercise
        for exercise in candidate_exercises
        if exercise["id"] not in excluded_exercises and exercise["name"] not in excluded_exercises
    ]
    if not filtered_exercises:
        fallback_exercises = find_exercises(
            target_muscles=target_muscles,
            difficulty=fitness_level,
            available_equipment=equipment_access,
            excluded_conditions=excluded_conditions,
            limit=_exercise_count_for_level(fitness_level, exercise_count_delta),
        )
        filtered_exercises = [
            exercise
            for exercise in fallback_exercises
            if exercise["id"] not in excluded_exercises and exercise["name"] not in excluded_exercises
        ]

    return {
        "day": day,
        "scheduled_date": scheduled_date,
        "cycle_number": cycle_number,
        "cycle_session_index": cycle_session_index,
        "focus": _focus_label(focus_key),
        "duration_minutes": duration_minutes,
        "warmup": _build_warmup(focus_key),
        "exercises": [
            {
                "name": exercise["name"],
                "target_muscle": ", ".join(exercise.get("target_muscle", [])),
                "sets": _sets_for_level(fitness_level),
                "reps": _rep_range_for_goal(training_goal, fitness_level),
                "equipment": ", ".join(exercise.get("equipment", [])),
                "notes": exercise.get("notes", ""),
            }
            for exercise in filtered_exercises
        ],
        "cooldown": _build_cooldown(focus_key),
        "safety_notes": _build_safety_notes(excluded_conditions, filtered_exercises),
    }


def _build_session_blueprint(
    *,
    day: str,
    scheduled_date: str,
    cycle_number: int,
    cycle_session_index: int,
    focus: str,
    duration_minutes: int,
    fitness_level: str,
    training_goal: str,
    equipment_access: list[str],
    excluded_conditions: list[str],
    excluded_exercises: list[str],
    exercise_count_delta: int = 0,
    intensity_adjustment: str = "",
) -> dict:
    focus_key = _focus_key_from_value(focus)
    target_count = _exercise_count_for_level(fitness_level, exercise_count_delta)
    candidate_limit = _candidate_pool_limit(target_count)
    target_muscles, movement_type = _focus_to_targets(focus_key)
    candidates = find_exercises(
        target_muscles=target_muscles,
        movement_type=movement_type,
        difficulty=fitness_level,
        available_equipment=equipment_access,
        training_goal=training_goal,
        excluded_conditions=excluded_conditions,
        recommended_for=_recommended_program_tags(equipment_access, excluded_conditions),
        focus_tags=[focus_key],
        limit=candidate_limit,
    )
    filtered_candidates = [
        {
            "name": exercise["name"],
            "target_muscle": exercise.get("target_muscle", []),
            "difficulty": exercise.get("difficulty", ""),
            "training_goal_tags": exercise.get("training_goal_tags", []),
            "movement_pattern": exercise.get("movement_pattern", ""),
            "replacement_group": exercise.get("replacement_group", ""),
            "equipment": exercise.get("equipment", []),
            "notes": exercise.get("notes", ""),
        }
        for exercise in candidates
        if exercise["id"] not in excluded_exercises and exercise["name"] not in excluded_exercises
    ]
    if len(filtered_candidates) < target_count:
        relaxed_candidates = find_exercises(
            target_muscles=target_muscles,
            movement_type=movement_type,
            available_equipment=equipment_access,
            excluded_conditions=excluded_conditions,
            focus_tags=[focus_key],
            limit=candidate_limit,
        )
        seen_names = {candidate["name"] for candidate in filtered_candidates}
        for exercise in relaxed_candidates:
            if len(filtered_candidates) >= target_count:
                break
            if exercise["name"] in seen_names:
                continue
            if exercise["id"] in excluded_exercises or exercise["name"] in excluded_exercises:
                continue
            seen_names.add(exercise["name"])
            filtered_candidates.append(
                {
                    "name": exercise["name"],
                    "target_muscle": exercise.get("target_muscle", []),
                    "difficulty": exercise.get("difficulty", ""),
                    "training_goal_tags": exercise.get("training_goal_tags", []),
                    "movement_pattern": exercise.get("movement_pattern", ""),
                    "replacement_group": exercise.get("replacement_group", ""),
                    "equipment": exercise.get("equipment", []),
                    "notes": exercise.get("notes", ""),
                }
            )
    if not filtered_candidates:
        heuristic_session = _build_workout_session(
            day=day,
            scheduled_date=scheduled_date,
            cycle_number=cycle_number,
            cycle_session_index=cycle_session_index,
            focus=focus,
            duration_minutes=duration_minutes,
            fitness_level=fitness_level,
            training_goal=training_goal,
            equipment_access=equipment_access,
            excluded_conditions=excluded_conditions,
            excluded_exercises=excluded_exercises,
            exercise_count_delta=exercise_count_delta,
        )
        filtered_candidates = [
            {
                "name": exercise["name"],
                "target_muscle": exercise["target_muscle"].split(", "),
                "difficulty": fitness_level,
                "training_goal_tags": [training_goal],
                "movement_pattern": "",
                "replacement_group": "",
                "equipment": exercise["equipment"].split(", "),
                "notes": exercise["notes"],
            }
            for exercise in heuristic_session["exercises"]
        ]
    return {
        "day": day,
        "scheduled_date": scheduled_date,
        "cycle_number": cycle_number,
        "cycle_session_index": cycle_session_index,
        "focus": _focus_label(focus_key),
        "focus_key": focus_key,
        "duration_minutes": duration_minutes,
        "target_exercise_count": target_count,
        "intensity_adjustment": intensity_adjustment,
        "candidate_exercises": filtered_candidates,
        "warmup_hint": _build_warmup(focus_key),
        "cooldown_hint": _build_cooldown(focus_key),
    }


def _build_food_candidates(constraints: dict) -> list[dict]:
    dietary_preferences = constraints.get("dietary_preferences", [])
    food_allergies = constraints.get("food_allergies", [])
    categories = ["protein", "carb", "fruit", "vegetable", "fat"]
    candidates: list[dict] = []
    for category in categories:
        foods = find_foods(
            category=category,
            diet_tags=dietary_preferences or None,
            excluded_allergens=food_allergies,
            limit=4,
        )
        for food in foods:
            candidates.append(
                {
                    "name": food["name"],
                    "category": food["category"],
                    "protein_g": food["protein_g"],
                    "carbs_g": food["carbs_g"],
                    "fat_g": food["fat_g"],
                    "calories_per_100g": food["calories_per_100g"],
                }
            )
    return candidates


def _finalize_workout_session(
    *,
    blueprint: dict,
    session_payload: dict | None,
    fitness_level: str,
    training_goal: str,
) -> WorkoutSession:
    session_payload = session_payload or {}
    # The model may describe a focus differently; the blueprint is the source of truth for slots.
    focus_label = _focus_label(_focus_key_from_value(str(blueprint.get("focus_key") or blueprint.get("focus", ""))))
    requested_exercises = session_payload.get("exercises", [])
    finalized_exercises = _finalize_exercises(
        requested_exercises=requested_exercises,
        candidate_exercises=blueprint.get("candidate_exercises", []),
        fitness_level=fitness_level,
        training_goal=training_goal,
        target_count=int(blueprint.get("target_exercise_count") or _exercise_count_for_level(fitness_level)),
        intensity_adjustment=str(blueprint.get("intensity_adjustment", "")),
    )
    return {
        # Keep the blueprint day fixed so the weekly schedule does not drift.
        "day": str(blueprint.get("day", "")),
        "scheduled_date": str(blueprint.get("scheduled_date", "")),
        "cycle_number": int(blueprint.get("cycle_number", 1)),
        "cycle_session_index": int(blueprint.get("cycle_session_index", 1)),
        "is_ad_hoc": bool(blueprint.get("is_ad_hoc", False)),
        "focus": focus_label,
        "duration_minutes": int(blueprint.get("duration_minutes", 60)),
        "warmup": _coerce_string_list(session_payload.get("warmup", []), blueprint.get("warmup_hint", [])),
        "exercises": finalized_exercises,
        "cooldown": _coerce_string_list(session_payload.get("cooldown", []), blueprint.get("cooldown_hint", [])),
        "safety_notes": _coerce_string_list(
            session_payload.get("safety_notes", []),
            ["Prioritize controlled tempo and stop any set that causes sharp pain."],
        ),
    }


def _finalize_exercises(
    *,
    requested_exercises: list,
    candidate_exercises: list[dict],
    fitness_level: str,
    training_goal: str,
    target_count: int | None = None,
    intensity_adjustment: str = "",
) -> list[dict]:
    target_count = target_count or _exercise_count_for_level(fitness_level)
    finalized: list[dict] = []
    candidate_names = {candidate["name"]: candidate for candidate in candidate_exercises}
    used_names: set[str] = set()

    for item in requested_exercises:
        if len(finalized) >= target_count:
            break
        if isinstance(item, str):
            name = item
            payload = {}
        else:
            name = str(item.get("name", ""))
            payload = item
        if name not in candidate_names:
            continue
        if name in used_names:
            continue
        matched = get_exercise_by_name(name)
        if not matched:
            continue
        used_names.add(name)
        finalized.append(
            {
                "name": matched["name"],
                "target_muscle": ", ".join(matched.get("target_muscle", [])),
                "sets": _sets_for_level(fitness_level),
                "reps": _rep_range_for_goal(training_goal, fitness_level, intensity_adjustment),
                "equipment": ", ".join(matched.get("equipment", [])),
                "notes": _apply_intensity_note(
                    str(payload.get("notes") or matched.get("notes", "")),
                    intensity_adjustment,
                ),
            }
        )

    for candidate in candidate_exercises:
        if len(finalized) >= target_count:
            break
        if candidate["name"] in used_names:
            continue
        matched = get_exercise_by_name(candidate["name"])
        if not matched:
            continue
        used_names.add(candidate["name"])
        finalized.append(
            {
                "name": matched["name"],
                "target_muscle": ", ".join(matched.get("target_muscle", [])),
                "sets": _sets_for_level(fitness_level),
                "reps": _rep_range_for_goal(training_goal, fitness_level, intensity_adjustment),
                "equipment": ", ".join(matched.get("equipment", [])),
                "notes": _apply_intensity_note(matched.get("notes", ""), intensity_adjustment),
            }
        )
    return finalized


def _finalize_nutrition_targets(model_targets: dict, fallback_targets: dict[str, int | float]) -> dict[str, int | float]:
    return {
        "daily_calories": int(model_targets.get("daily_calories") or fallback_targets["daily_calories"]),
        "protein_g": int(model_targets.get("protein_g") or fallback_targets["protein_g"]),
        "carbs_g": int(model_targets.get("carbs_g") or fallback_targets["carbs_g"]),
        "fat_g": int(model_targets.get("fat_g") or fallback_targets["fat_g"]),
        "hydration_liters": float(model_targets.get("hydration_liters") or fallback_targets["hydration_liters"]),
    }


def _finalize_meal_suggestions(
    model_meals: list,
    food_candidates: list[dict],
    fallback_meals: list[MealSuggestion],
) -> list[MealSuggestion]:
    finalized: list[MealSuggestion] = []
    valid_food_names = {candidate["name"] for candidate in food_candidates}
    for item in model_meals:
        if not isinstance(item, dict):
            continue
        food_name = str(item.get("food_name", ""))
        if food_name not in valid_food_names:
            continue
        matched = get_food_by_name(food_name)
        if not matched:
            continue
        serving_size = str(item.get("serving_size", "100g"))
        grams = _extract_grams(serving_size, default=100.0)
        macro = calculate_food_macros(matched["id"], grams)
        finalized.append(
            {
                "food_name": matched["name"],
                "serving_size": f"{int(grams)}g",
                "calories": int(round(macro["calories"])),
                "protein_g": macro["protein_g"],
                "carbs_g": macro["carbs_g"],
                "fat_g": macro["fat_g"],
                "meal_slot": str(item.get("meal_slot", "meal")),
            }
        )
    return finalized or fallback_meals


def _extract_grams(serving_size: str, default: float) -> float:
    digits = "".join(char for char in serving_size if char.isdigit() or char == ".")
    return float(digits) if digits else default


def _coerce_string_list(value: list | None, fallback: list[str]) -> list[str]:
    if not value:
        return fallback
    return [str(item) for item in value if str(item).strip()]


def zip_longest_with_last(left: list[dict], right: list) -> list[tuple[dict, dict | None]]:
    pairs: list[tuple[dict, dict | None]] = []
    for index, blueprint in enumerate(left):
        payload = right[index] if index < len(right) and isinstance(right[index], dict) else None
        pairs.append((blueprint, payload))
    return pairs


def _resolve_current_day_name(current_date: str) -> str:
    try:
        return datetime.fromisoformat(current_date).strftime("%A")
    except ValueError:
        return datetime.today().strftime("%A")


def _safe_date_string(value: object) -> str:
    try:
        return datetime.fromisoformat(str(value)).date().isoformat()
    except (TypeError, ValueError):
        return ""


def _sort_days(days: list[str]) -> list[str]:
    unique_days: list[str] = []
    for day in days:
        cleaned = str(day).strip()
        if cleaned in WEEKDAY_INDEX and cleaned not in unique_days:
            unique_days.append(cleaned)
    return sorted(unique_days, key=lambda item: WEEKDAY_INDEX[item])


def _sort_workout_sessions(workout_sessions: list[WorkoutSession]) -> list[WorkoutSession]:
    return sorted(
        workout_sessions,
        key=lambda session: (
            str(session.get("scheduled_date", "")) or "9999-12-31",
            WEEKDAY_INDEX.get(str(session.get("day", "")), 99),
        ),
    )


def _focus_key_from_value(value: str) -> str:
    normalized = value.strip().lower().replace("_", " ")
    if normalized in FOCUS_ALIASES:
        return FOCUS_ALIASES[normalized]
    for focus_key, config in FOCUS_LIBRARY.items():
        if normalized == focus_key.replace("_", " "):
            return focus_key
        if normalized == config["label"].lower():
            return focus_key
    return "functional_conditioning"


def _focus_label(focus_key: str) -> str:
    return FOCUS_LIBRARY.get(focus_key, FOCUS_LIBRARY["functional_conditioning"])["label"]


def _focus_to_targets(focus: str) -> tuple[list[str], str | None]:
    focus_key = _focus_key_from_value(focus)
    config = FOCUS_LIBRARY.get(focus_key, FOCUS_LIBRARY["functional_conditioning"])
    return list(config["target_muscles"]), config["movement_type"]


def _recommended_program_tags(equipment_access: list[str], excluded_conditions: list[str]) -> list[str]:
    tags = []
    if excluded_conditions:
        tags.append("low_impact_program")
    return tags


def _build_warmup(focus: str) -> list[str]:
    focus_key = _focus_key_from_value(focus)
    if focus_key == "lower_legs_glutes":
        return ["5 minutes brisk walk", "bodyweight squats", "hip mobility drill"]
    if focus_key in {"upper_chest_arms", "upper_shoulders", "back_training"}:
        return ["band pull-aparts", "arm circles", "light push-up regression"]
    if focus_key == "functional_power":
        return ["dynamic skips", "ankle pogo hops", "glute activation"]
    return ["5 minutes light cardio", "world's greatest stretch", "dead bug activation"]


def _build_cooldown(focus: str) -> list[str]:
    focus_key = _focus_key_from_value(focus)
    if focus_key == "lower_legs_glutes":
        return ["hamstring stretch", "quad stretch", "slow nasal breathing"]
    if focus_key in {"upper_chest_arms", "upper_shoulders", "back_training"}:
        return ["doorway chest stretch", "lat stretch", "slow nasal breathing"]
    return ["child's pose", "hip flexor stretch", "2 minutes easy breathing"]


def _build_safety_notes(excluded_conditions: list[str], exercises: list[dict]) -> list[str]:
    notes = ["Prioritize controlled tempo and stop any set that causes sharp pain."]
    if excluded_conditions:
        notes.append(f"Watch symptoms around: {', '.join(excluded_conditions)}.")
    if any("knee" in ",".join(exercise.get("target_muscle", [])).lower() for exercise in exercises):
        notes.append("Keep knee tracking controlled and reduce range of motion if discomfort rises.")
    return notes


def _sets_for_level(fitness_level: str) -> int:
    return 4


def _exercise_count_for_level(fitness_level: str, delta: int = 0) -> int:
    base_count = {"beginner": 2, "intermediate": 3, "advanced": 4}.get(fitness_level, 3)
    return max(2, base_count + delta)


def _candidate_pool_limit(target_count: int) -> int:
    return max(target_count * 3, 8)


def _rep_range_for_goal(training_goal: str, fitness_level: str = "", intensity_adjustment: str = "") -> str:
    if fitness_level == "beginner":
        if intensity_adjustment == "higher":
            return "8-10"
        if intensity_adjustment == "lower":
            return "6-8"
        return "6-10"
    if intensity_adjustment == "higher":
        return "12-15"
    if intensity_adjustment == "lower":
        return "10-12"
    return "10-15"


def _apply_intensity_note(base_note: str, intensity_adjustment: str) -> str:
    note = base_note.strip()
    if intensity_adjustment == "higher":
        cue = "Higher intensity: use full range of motion, controlled tempo, and a brief pause while keeping clean form."
    elif intensity_adjustment == "lower":
        cue = "Lower intensity: use conservative load, controlled range of motion, and stop well before form breaks."
    else:
        return note
    return f"{note} {cue}".strip()


def _build_nutrition_targets(user_profile: dict, goals: dict, current_state: dict) -> dict[str, int | float]:
    # Daily weight/body-fat check-ins are records, not same-day planning controls.
    current_weight = float(user_profile.get("weight_kg") or 70.0)
    primary_goal = _map_goal_tag(str(goals.get("primary_goal", "weight_loss")))

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


def _build_meal_suggestions(*, goals: dict, constraints: dict, nutrition_targets: dict) -> list[MealSuggestion]:
    primary_goal = _map_goal_tag(str(goals.get("primary_goal", "weight_loss")))
    dietary_preferences = constraints.get("dietary_preferences", [])
    food_allergies = constraints.get("food_allergies", [])

    protein_pool = find_foods(
        category="protein",
        diet_tags=dietary_preferences or None,
        excluded_allergens=food_allergies,
        min_protein_g=10,
        limit=3,
    )
    carb_pool = find_foods(
        category="carb",
        diet_tags=[tag for tag in dietary_preferences if tag in {"vegan", "gluten_free"}] or None,
        excluded_allergens=food_allergies,
        limit=3,
    )
    fruit_pool = find_foods(
        category="fruit",
        excluded_allergens=food_allergies,
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
            suggestions.append(
                {
                    "food_name": food["name"],
                    "serving_size": f"{grams}g",
                    "calories": int(round(macro["calories"])),
                    "protein_g": macro["protein_g"],
                    "carbs_g": macro["carbs_g"],
                    "fat_g": macro["fat_g"],
                    "meal_slot": meal_slot,
                }
            )

    if not suggestions:
        suggestions.append(
            {
                "food_name": "Balanced whole-food meal",
                "serving_size": "1 plate",
                "calories": int(nutrition_targets["daily_calories"] // 3),
                "protein_g": round(float(nutrition_targets["protein_g"]) / 3, 1),
                "carbs_g": round(float(nutrition_targets["carbs_g"]) / 3, 1),
                "fat_g": round(float(nutrition_targets["fat_g"]) / 3, 1),
                "meal_slot": "lunch",
            }
        )
    return suggestions


def _build_coaching_focus(latest_feedback: dict, training_goal: str) -> list[str]:
    focus = []
    if latest_feedback.get("fatigue_level", 0) and int(latest_feedback["fatigue_level"]) >= 7:
        focus.append("Manage fatigue with conservative effort and longer recovery.")
    if latest_feedback.get("adherence_score", 1.0) and float(latest_feedback.get("adherence_score", 1.0)) < 0.7:
        focus.append("Keep sessions simple and repeatable to rebuild consistency.")
    if training_goal == "weight_loss":
        focus.append("Favor consistency, daily movement, and sustainable nutrition adherence.")
    elif training_goal == "strength":
        focus.append("Prioritize progressive overload and protein intake.")
    elif training_goal == "sculpting":
        focus.append("Prioritize technique, balanced volume, and body-composition consistency.")
    else:
        focus.append("Focus on clean movement quality and manageable training volume.")
    return focus


def _build_recovery_actions(current_state: dict, latest_feedback: dict, excluded_conditions: list[str]) -> list[str]:
    actions = ["Take 5-10 minutes for warm-up and cooldown on every session."]
    if float(current_state.get("sleep_hours", 7.0)) < 6.0:
        actions.append("Aim for an earlier bedtime to restore recovery capacity.")
    if int(latest_feedback.get("fatigue_level", 0)) >= 7:
        actions.append("Reduce accessory volume by one set if fatigue remains high.")
    if excluded_conditions:
        actions.append("Avoid pushing through painful ranges of motion; substitute when symptoms spike.")
    return actions


def _build_summary(sessions_per_week: int, goals: dict, constraints: dict) -> str:
    primary_goal = GOAL_LABELS.get(_map_goal_tag(str(goals.get("primary_goal", "weight_loss"))), "减重")
    return (
        f"{sessions_per_week}-session cycle for {primary_goal} "
        "with 60-minute sessions and structured nutrition support."
    )


def _build_objective_alignment(goals: dict, user_notes: list[str]) -> str:
    goal_text = GOAL_LABELS.get(_map_goal_tag(str(goals.get("primary_goal", "weight_loss"))), "减重")
    if user_notes:
        return f"Built to support {goal_text} while respecting: {'; '.join(user_notes)}."
    return f"Built to support {goal_text} with manageable progression and safety constraints."


def _build_context_notes(state: FitnessAgentState) -> list[str]:
    constraints = state.get("constraints", {})
    notes = []
    if constraints.get("injuries"):
        notes.append(f"injuries={', '.join(constraints['injuries'])}")
    if constraints.get("food_allergies"):
        notes.append(f"allergies={', '.join(constraints['food_allergies'])}")
    return notes


def _build_coaching_message(goals: dict, coaching_focus: list[str], recovery_actions: list[str]) -> str:
    goal = GOAL_LABELS.get(_map_goal_tag(str(goals.get("primary_goal", "weight_loss"))), "减重")
    message_parts = [f"This plan is centered on {goal}."]
    if coaching_focus:
        message_parts.append(coaching_focus[0])
    if recovery_actions:
        message_parts.append(recovery_actions[0])
    return " ".join(message_parts)
