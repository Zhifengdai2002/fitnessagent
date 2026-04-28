"""AI Coach chat service independent from Streamlit UI."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime
from typing import Any

from agent.graph import run_agent
from agent.llm import call_model_json, call_model_text, call_model_tool, load_prompt
from agent.services.memory import append_memory_item, default_memory_store, normalize_memory_store
from agent.services.persistence import PERSISTED_SESSION_KEYS, safe_iso_date
from agent.services.planning_helpers import (
    current_interaction_date,
    iso_to_date,
    same_iso_date,
    sort_workout_sessions,
)
from agent.services.state_builders import build_initial_state
from agent.state import FitnessAgentState
from agent.tools import (
    build_video_resources,
    calculate_food_macros,
    find_foods,
    get_food_by_name,
    search_similar_exercises,
)

ALLOWED_CHANGE_REQUEST_TYPES = {"workout_change", "nutrition_change", "mixed_change", "recovery_change", "none", "unclear"}
ALLOWED_CHANGE_REQUEST_SCOPES = {"today_only", "current_cycle", "future_default", "permanent", "unclear"}
ALLOWED_FOCUS_CATEGORIES = {
    "",
    "upper_chest_arms",
    "upper_shoulders",
    "back_training",
    "lower_legs_glutes",
    "functional_core",
    "functional_power",
    "functional_conditioning",
}

ALLOWED_COACH_TOOLS = {
    "no_action",
    "cancel_workout",
    "adjust_sets",
    "adjust_intensity",
    "replace_exercise",
    "replace_food",
    "update_today_plan",
}

COACH_TOOL_SCHEMAS = [
    {
        "tool_name": "cancel_workout",
        "description": "Cancel today's workout because the user requested cancellation or reported injury/pain.",
        "arguments": {"injury_reported": "bool", "injury_areas": "list[str]", "reason": "string"},
    },
    {
        "tool_name": "adjust_sets",
        "description": "Adjust today's set count only. Does not add or remove exercises.",
        "arguments": {"set_adjustment": "increase|decrease|target", "set_target": "int, optional"},
    },
    {
        "tool_name": "adjust_intensity",
        "description": "Adjust reps, notes, and possibly exercise count for vague higher/lower intensity requests.",
        "arguments": {"intensity_adjustment": "higher|lower"},
    },
    {
        "tool_name": "replace_exercise",
        "description": "Replace one or more exercises with same-focus alternatives.",
        "arguments": {"target": "string, optional", "replacement_preference": "string, optional"},
    },
    {
        "tool_name": "replace_food",
        "description": "Replace today's food items while preserving the workout plan.",
        "arguments": {"temporary_food_avoidances": "list[str]"},
    },
    {
        "tool_name": "update_today_plan",
        "description": "Use the planner workflow to update today's workout focus or broader same-day plan.",
        "arguments": {"focus_category": "string, optional", "summary": "string, optional"},
    },
    {
        "tool_name": "no_action",
        "description": "No app state change is needed.",
        "arguments": {},
    },
]

COACH_NATIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "cancel_workout",
            "description": "Cancel today's workout because the user requested cancellation or reported injury/pain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "injury_reported": {"type": "boolean"},
                    "injury_areas": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_sets",
            "description": "Adjust today's set count only. Do not add or remove exercises.",
            "parameters": {
                "type": "object",
                "properties": {
                    "set_adjustment": {"type": "string", "enum": ["increase", "decrease", "target"]},
                    "set_target": {"type": "integer", "minimum": 0, "maximum": 12},
                    "summary": {"type": "string"},
                },
                "required": ["set_adjustment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_intensity",
            "description": "Adjust reps, notes, and possibly exercise count for vague higher/lower intensity requests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intensity_adjustment": {"type": "string", "enum": ["higher", "lower"]},
                    "summary": {"type": "string"},
                },
                "required": ["intensity_adjustment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_exercise",
            "description": "Replace one or more exercises with same-focus alternatives.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "replacement_preference": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_food",
            "description": "Replace today's food items while preserving the workout plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "temporary_food_avoidances": {"type": "array", "items": {"type": "string"}},
                    "summary": {"type": "string"},
                },
                "required": ["temporary_food_avoidances"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_today_plan",
            "description": "Use the planner workflow to update today's workout focus or broader same-day plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus_category": {
                        "type": "string",
                        "enum": [
                            "",
                            "upper_chest_arms",
                            "upper_shoulders",
                            "back_training",
                            "lower_legs_glutes",
                            "functional_core",
                            "functional_power",
                            "functional_conditioning",
                        ],
                    },
                    "summary": {"type": "string"},
                },
                "required": [],
            },
        },
    },
]


def call_chat_assistant(user_message: str, session_state: dict[str, Any]) -> str:
    update_summary = maybe_update_today_from_chat(user_message, session_state)
    result = session_state.get("agent_result", {})
    context = build_chat_context(result, session_state)
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in session_state.get("assistant_chat_messages", [])[-8:]
        if message.get("role") in {"user", "assistant"} and message.get("content")
    ]
    system_prompt = (
        "You are FitnessAgent's floating chat coach. Use the supplied app context, "
        "base every workout answer on FitnessAgent's hard planning rules: baseline beginner plans "
        "use 2 exercises, intermediate plans use 3 exercises, advanced plans use 4 exercises; default "
        "normal exercise volume is 4 sets. If the user explicitly asks to add/reduce sets, change sets "
        "only within 3-5 sets and do not add exercises. Once changed, today's set target persists across "
        "later same-day workout edits until the user changes sets again. For vague higher intensity, use higher reps "
        "(beginner 8-10, intermediate/advanced 12-15), challenge notes, and only add an exercise when "
        "the user asks for more work generally. For lower intensity, "
        "never go below 2 exercises, keep beginner at 2, reduce intermediate to 2, reduce advanced "
        "to 3, use lower reps (beginner 6-8, intermediate/advanced 10-12), and add conservative notes. "
        "Baseline beginner reps are 6-10; baseline intermediate and advanced reps are 10-15. "
        "Daily weight and body-fat check-ins are record-only. Answer concisely, and stay within general "
        "fitness coaching. Do not diagnose medical issues. If the user reports injury, sharp pain, "
        "chest pain, dizziness, or other red flags, advise stopping training and consulting a qualified "
        "professional. If the app context says today's plan was updated, say that it has been updated "
        "and summarize the current Today's Plan. Current app context:\n"
        f"{context}"
    )
    prompt = user_message
    if update_summary:
        prompt = f"{user_message}\n\nApp action already completed: {update_summary}"
    try:
        reply = call_model_text(
            system_prompt=system_prompt,
            user_prompt=prompt,
            history=history,
            temperature=0.4,
            max_tokens=700,
        )
    except Exception as exc:
        return f"Chat is unavailable right now: {exc}"
    return reply or "I could not generate a response just now."


def maybe_update_today_from_chat(user_message: str, session_state: dict[str, Any]) -> str:
    profile_inputs = session_state.get("profile_inputs")
    previous_result = session_state.get("agent_result")
    if not profile_inputs or not previous_result:
        return ""

    decision = route_coach_message(user_message, previous_result, session_state)
    tool_call = select_coach_tool_call(user_message, previous_result, session_state, decision)
    if tool_call.get("tool_name") == "no_action":
        return ""

    summary = execute_coach_tool_call(
        tool_call,
        {
            "user_message": user_message,
            "profile_inputs": profile_inputs,
            "previous_result": previous_result,
            "session_state": session_state,
            "decision": decision,
        },
    )

    if summary:
        commit_memory(session_state, action_message_for_tool_call(tool_call, decision))
    return summary


def select_coach_tool_call(
    user_message: str,
    result: FitnessAgentState | dict[str, Any],
    session_state: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    fallback = fallback_tool_call_from_decision(decision)
    if fallback.get("tool_name") == "no_action":
        return fallback
    user_payload = json.dumps(
        {
            "user_message": user_message,
            "app_context": build_chat_context(result, session_state),
            "coordinator_decision": decision,
            "fallback_tool_call": fallback,
        },
        ensure_ascii=True,
        indent=2,
    )
    try:
        planned = call_model_tool(
            system_prompt=load_prompt("coach_planner_prompt.txt"),
            user_prompt=user_payload,
            tools=COACH_NATIVE_TOOLS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=700,
        )
    except Exception:
        try:
            planned = call_model_json(
                system_prompt=load_prompt("coach_planner_prompt.txt"),
                user_prompt=json.dumps(
                    {
                        "user_message": user_message,
                        "app_context": build_chat_context(result, session_state),
                        "coordinator_decision": decision,
                        "available_tools": COACH_TOOL_SCHEMAS,
                        "fallback_tool_call": fallback,
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                temperature=0.0,
                max_tokens=700,
            )
        except Exception:
            return fallback
    return sanitize_tool_call(planned, fallback)


def execute_coach_tool_call(tool_call: dict[str, Any], context: dict[str, Any]) -> str:
    tool_name = str(tool_call.get("tool_name", "no_action")).strip()
    tool = coach_tool_registry().get(tool_name)
    if not tool:
        return ""
    return str(tool["handler"](context, dict(tool_call.get("arguments", {}))) or "")


def coach_tool_registry() -> dict[str, dict[str, Any]]:
    return {
        "no_action": {
            "description": "No app state change.",
            "handler": handle_no_action_tool,
        },
        "cancel_workout": {
            "description": "Cancel today's workout.",
            "handler": handle_cancel_workout_tool,
        },
        "adjust_sets": {
            "description": "Adjust today's set policy.",
            "handler": handle_adjust_sets_tool,
        },
        "adjust_intensity": {
            "description": "Adjust today's intensity.",
            "handler": handle_adjust_intensity_tool,
        },
        "replace_exercise": {
            "description": "Replace same-focus exercises.",
            "handler": handle_replace_exercise_tool,
        },
        "replace_food": {
            "description": "Replace today's food.",
            "handler": handle_replace_food_tool,
        },
        "update_today_plan": {
            "description": "Patch today's workout plan.",
            "handler": handle_update_today_plan_tool,
        },
    }


def fallback_tool_call_from_decision(decision: dict[str, Any]) -> dict[str, Any]:
    route = str(decision.get("route", "none"))
    normalized = dict(decision.get("normalized", {}))
    action = str(decision.get("planner_action", ""))
    if route == "none":
        return {"tool_name": "no_action", "arguments": {}, "source": "fallback"}
    if route == "safety" or normalized.get("cancel_today") or normalized.get("injury_reported"):
        return {"tool_name": "cancel_workout", "arguments": normalized, "source": "fallback"}
    if action == "replace_exercise":
        return {"tool_name": "replace_exercise", "arguments": normalized, "source": "fallback"}
    if str(normalized.get("set_adjustment", "")).strip():
        return {"tool_name": "adjust_sets", "arguments": normalized, "source": "fallback"}
    if action == "adjust_intensity" or str(normalized.get("intensity_adjustment", "")).strip():
        return {"tool_name": "adjust_intensity", "arguments": normalized, "source": "fallback"}
    if action == "replace_food" or coerce_string_list(normalized.get("temporary_food_avoidances"), []):
        return {"tool_name": "replace_food", "arguments": normalized, "source": "fallback"}
    return {"tool_name": "update_today_plan", "arguments": normalized, "source": "fallback"}


def sanitize_tool_call(planned: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(planned, dict):
        return fallback
    tool_name = str(planned.get("tool_name", "")).strip()
    if tool_name not in ALLOWED_COACH_TOOLS:
        return fallback
    if str(fallback.get("tool_name", "no_action")) != "no_action" and tool_name == "no_action":
        return fallback
    confidence = clamp_float(planned.get("confidence"), 0.0, 1.0, 0.0)
    if confidence < 0.45:
        return fallback
    arguments = dict(fallback.get("arguments", {}))
    planned_arguments = planned.get("arguments", {})
    if isinstance(planned_arguments, dict):
        arguments.update(planned_arguments)
    return {
        "tool_name": tool_name,
        "arguments": sanitize_tool_arguments(tool_name, arguments),
        "reason": str(planned.get("summary") or planned.get("reason") or "").strip(),
        "confidence": confidence,
        "source": "planner_agent",
    }


def sanitize_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    sanitized = sanitize_normalized_change_request(arguments)
    if tool_name == "adjust_sets":
        set_adjustment = str(arguments.get("set_adjustment") or sanitized.get("set_adjustment") or "").strip()
        if set_adjustment not in {"increase", "decrease", "target"}:
            set_adjustment = ""
        sanitized["set_adjustment"] = set_adjustment
        sanitized["set_target"] = clamp_int(arguments.get("set_target") or sanitized.get("set_target"), 0, 12, 0)
        sanitized["intensity_adjustment"] = ""
    elif tool_name == "adjust_intensity":
        intensity = str(arguments.get("intensity_adjustment") or arguments.get("intensity") or sanitized.get("intensity_adjustment") or "").strip()
        sanitized["intensity_adjustment"] = intensity if intensity in {"higher", "lower"} else ""
        sanitized["set_adjustment"] = ""
    elif tool_name == "cancel_workout":
        sanitized["cancel_today"] = True
    return sanitized


def action_message_for_tool_call(tool_call: dict[str, Any], decision: dict[str, Any]) -> str:
    tool_name = str(tool_call.get("tool_name", ""))
    if tool_name == "cancel_workout":
        return "Today's workout was cancelled by AI Coach."
    if tool_name == "replace_food":
        return "Today's nutrition was updated by AI Coach."
    return str(decision.get("action_message", "Today's plan was updated by AI Coach."))


def handle_no_action_tool(context: dict[str, Any], arguments: dict[str, Any]) -> str:
    return ""


def handle_cancel_workout_tool(context: dict[str, Any], arguments: dict[str, Any]) -> str:
    return cancel_today_from_chat(arguments, context["previous_result"], context["session_state"])


def handle_adjust_sets_tool(context: dict[str, Any], arguments: dict[str, Any]) -> str:
    return patch_today_sets_from_chat(arguments, context["previous_result"], context["session_state"])


def handle_adjust_intensity_tool(context: dict[str, Any], arguments: dict[str, Any]) -> str:
    return patch_today_intensity_from_chat(arguments, context["previous_result"], context["session_state"])


def handle_replace_exercise_tool(context: dict[str, Any], arguments: dict[str, Any]) -> str:
    return replace_today_exercise_from_chat(context["user_message"], context["previous_result"], context["session_state"])


def handle_replace_food_tool(context: dict[str, Any], arguments: dict[str, Any]) -> str:
    return update_nutrition_from_chat(arguments, context["previous_result"], context["session_state"])


def handle_update_today_plan_tool(context: dict[str, Any], arguments: dict[str, Any]) -> str:
    return execute_ai_plan_patch(
        user_message=context["user_message"],
        profile_inputs=context["profile_inputs"],
        previous_result=context["previous_result"],
        normalized=arguments,
        session_state=context["session_state"],
    )


def route_coach_message(user_message: str, result: FitnessAgentState | dict[str, Any], session_state: dict[str, Any]) -> dict[str, Any]:
    fallback = fallback_coordinator_decision(user_message, result)
    try:
        routed = call_model_json(
            system_prompt=load_prompt("coach_coordinator_prompt.txt"),
            user_prompt=json.dumps(
                {
                    "user_message": user_message,
                    "app_context": build_chat_context(result, session_state),
                    "fallback_decision": fallback,
                },
                ensure_ascii=True,
                indent=2,
            ),
            temperature=0.0,
            max_tokens=600,
        )
    except Exception:
        return fallback
    return sanitize_coordinator_decision(routed, fallback)


def fallback_coordinator_decision(user_message: str, result: FitnessAgentState | dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_change_request(user_message)
    normalized = augment_chat_safety_change(user_message, normalized)
    normalized = augment_chat_set_change(user_message, normalized)
    normalized = augment_chat_intensity_change(user_message, normalized)
    normalized = augment_chat_food_change(user_message, normalized, result)
    normalized = augment_chat_focus_change(user_message, normalized)

    if coerce_string_list(normalized.get("temporary_food_avoidances"), []):
        return {
            "route": "planner",
            "planner_action": "replace_food",
            "normalized": normalized,
            "action_message": "Today's nutrition was updated by AI Coach.",
        }
    if chat_message_requests_exercise_replacement(user_message):
        return {
            "route": "planner",
            "planner_action": "replace_exercise",
            "normalized": normalized,
            "action_message": "Today's plan was updated by AI Coach.",
        }
    if not chat_request_should_update_today(user_message, normalized):
        return {"route": "none", "normalized": normalized}
    if normalized.get("cancel_today") or normalized.get("injury_reported"):
        return {"route": "safety", "planner_action": "cancel_workout", "normalized": normalized}
    if str(normalized.get("set_adjustment", "")).strip() or str(normalized.get("intensity_adjustment", "")).strip():
        return {
            "route": "planner",
            "planner_action": "adjust_intensity",
            "normalized": normalized,
            "action_message": "Today's plan was updated by AI Coach.",
        }
    return {
        "route": "planner",
        "planner_action": "update_today_plan",
        "normalized": normalized,
        "action_message": "Today's plan was updated by AI Coach.",
    }


def sanitize_coordinator_decision(routed: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    target_agent = str(routed.get("target_agent", "")).strip().lower()
    planner_action = str(routed.get("planner_action", "")).strip().lower()
    confidence = clamp_float(routed.get("confidence"), 0.0, 1.0, 0.0)
    route_map = {"none": "none", "safety": "safety", "planner": "planner"}
    action_map = {
        "none": "none",
        "cancel_workout": "cancel_workout",
        "adjust_intensity": "adjust_intensity",
        "replace_exercise": "replace_exercise",
        "replace_food": "replace_food",
        "update_today_plan": "update_today_plan",
    }
    if target_agent not in route_map or planner_action not in action_map:
        return fallback
    if confidence < 0.45:
        return fallback
    if str(fallback.get("route", "none")) != "none" and target_agent == "none":
        return fallback

    sanitized = dict(fallback)
    sanitized["route"] = route_map[target_agent]
    sanitized["planner_action"] = action_map[planner_action]
    sanitized["coordinator_reason"] = str(routed.get("reason", "")).strip()
    sanitized["coordinator_confidence"] = confidence

    if sanitized["route"] == "none":
        sanitized["planner_action"] = "none"
    elif sanitized["route"] == "safety":
        sanitized["planner_action"] = "cancel_workout"
    elif sanitized["planner_action"] == "none":
        sanitized["planner_action"] = str(fallback.get("planner_action", "update_today_plan"))

    if sanitized["planner_action"] == "replace_food":
        sanitized["action_message"] = "Today's nutrition was updated by AI Coach."
    elif sanitized["planner_action"] != "none":
        sanitized["action_message"] = "Today's plan was updated by AI Coach."
    return sanitized


def cancel_today_from_chat(
    normalized: dict[str, Any],
    previous_result: FitnessAgentState | dict[str, Any],
    session_state: dict[str, Any],
) -> str:
    if not normalized.get("cancel_today"):
        return ""

    injury_reported = bool(normalized.get("injury_reported"))
    injury_areas = coerce_string_list(normalized.get("injury_areas"), [])
    current_date = current_interaction_date(active_date=session_state.get("active_date"), result=previous_result)
    updated_result = deepcopy(previous_result)
    sessions = updated_result.get("current_plan", {}).get("workout_sessions", [])
    today_session = select_today_session(sort_workout_sessions(sessions), current_date)
    if not today_session:
        parsed_date = iso_to_date(current_date)
        sessions.insert(
            0,
            {
                "day": parsed_date.strftime("%A"),
                "scheduled_date": current_date,
                "cycle_number": updated_result.get("current_plan", {}).get("cycle_number", 1),
                "cycle_session_index": 0,
                "is_ad_hoc": True,
                "is_cancelled": True,
                "focus": "Workout Cancelled",
                "duration_minutes": 0,
                "warmup": [],
                "exercises": [],
                "cooldown": [],
                "safety_notes": cancel_safety_notes(injury_reported=injury_reported, injury_areas=injury_areas),
                "injury_reported": injury_reported,
                "injury_areas": injury_areas,
            },
        )
    else:
        today_session["is_cancelled"] = True
        today_session["focus"] = "Workout Cancelled"
        today_session["duration_minutes"] = 0
        today_session["warmup"] = []
        today_session["exercises"] = []
        today_session["cooldown"] = []
        today_session["safety_notes"] = cancel_safety_notes(injury_reported=injury_reported, injury_areas=injury_areas)
        today_session["injury_reported"] = injury_reported
        today_session["injury_areas"] = injury_areas

    protected_future_sessions: list[str] = []
    if injury_reported:
        protected_future_sessions = protect_future_sessions_for_injury(
            updated_result=updated_result,
            current_date=current_date,
            injury_areas=injury_areas,
        )
        record_memory_injury_event(
            session_state=session_state,
            date_iso=current_date,
            injury_areas=injury_areas,
            source="ai_coach",
            summary=f"User reported injury around {', '.join(injury_areas) if injury_areas else 'an unspecified area'}.",
            risk_level="medium",
        )
    record_memory_plan_modification(
        session_state=session_state,
        date_iso=current_date,
        action_type="cancel_workout" if not injury_reported else "injury_cancel_workout",
        summary="Cancelled today's workout after AI Coach safety routing.",
        injury_areas=injury_areas if injury_reported else [],
    )
    refresh_youtube_resources(updated_result)
    session_state["agent_result"] = updated_result
    if injury_reported:
        area_text = f" around {', '.join(injury_areas)}" if injury_areas else ""
        future_text = ""
        if protected_future_sessions:
            future_text = f" Related future sessions were protected: {', '.join(protected_future_sessions)}."
        return f"Today's workout has been cancelled because you reported an injury{area_text}.{future_text}"
    return "Today's workout has been cancelled as requested. No injury reason was recorded."


def patch_today_intensity_from_chat(
    normalized: dict[str, Any],
    previous_result: FitnessAgentState | dict[str, Any],
    session_state: dict[str, Any],
) -> str:
    intensity = str(normalized.get("intensity_adjustment", "")).strip().lower()
    if intensity not in {"higher", "lower"}:
        return ""
    if normalized.get("focus_category") or normalized.get("cancel_today") or normalized.get("injury_reported"):
        return ""

    current_date = current_interaction_date(active_date=session_state.get("active_date"), result=previous_result)
    updated_result = deepcopy(previous_result)
    today_session = select_today_session(
        sort_workout_sessions(updated_result.get("current_plan", {}).get("workout_sessions", [])),
        current_date,
    )
    if not today_session or today_session.get("is_cancelled") or not today_session.get("exercises"):
        session_state["agent_result"] = updated_result
        return "Today has no active workout to adjust, so the current plan was left unchanged."

    fitness_level = str(previous_result.get("user_profile", {}).get("fitness_level", "intermediate")).strip().lower()
    updated_exercises = patch_exercises_for_intensity(
        exercises=list(today_session.get("exercises", [])),
        fitness_level=fitness_level,
        intensity=intensity,
        focus=focus_tag_from_session(today_session),
    )
    if not updated_exercises:
        return ""

    today_session["exercises"] = updated_exercises
    today_session.pop("is_cancelled", None)
    today_session["duration_minutes"] = int(today_session.get("duration_minutes") or 60)
    refresh_youtube_resources(updated_result)
    session_state["agent_result"] = updated_result
    record_memory_plan_modification(
        session_state=session_state,
        date_iso=current_date,
        action_type=f"{intensity}_intensity",
        summary=f"Adjusted today's workout to {intensity} intensity.",
    )

    label = "lower" if intensity == "lower" else "higher"
    exercise_text = ", ".join(
        f"{exercise.get('name', '')} {exercise.get('sets', '')}x{exercise.get('reps', '')}".strip()
        for exercise in updated_exercises
    )
    return f"Today's workout was adjusted to {label} intensity: {exercise_text}."


def patch_today_sets_from_chat(
    normalized: dict[str, Any],
    previous_result: FitnessAgentState | dict[str, Any],
    session_state: dict[str, Any],
) -> str:
    set_adjustment = str(normalized.get("set_adjustment", "")).strip().lower()
    if set_adjustment not in {"increase", "decrease", "target"}:
        return ""
    if normalized.get("focus_category") or normalized.get("cancel_today") or normalized.get("injury_reported"):
        return ""

    current_date = current_interaction_date(active_date=session_state.get("active_date"), result=previous_result)
    updated_result = deepcopy(previous_result)
    today_session = select_today_session(
        sort_workout_sessions(updated_result.get("current_plan", {}).get("workout_sessions", [])),
        current_date,
    )
    if not today_session or today_session.get("is_cancelled") or not today_session.get("exercises"):
        session_state["agent_result"] = updated_result
        return "Today has no active workout to adjust, so the current plan was left unchanged."

    raw_target = clamp_int(normalized.get("set_target"), 0, 5, 0)
    target = max(3, raw_target) if raw_target else 0
    updated_exercises: list[dict[str, Any]] = []
    changed = False
    for exercise in today_session.get("exercises", []):
        patched = dict(exercise)
        current_sets = clamp_int(patched.get("sets"), 3, 5, 4)
        if set_adjustment == "increase":
            next_sets = min(5, current_sets + 1)
            note = "Set volume increased: keep the extra set crisp and stop if form drops."
        elif set_adjustment == "decrease":
            next_sets = max(3, current_sets - 1)
            note = "Set volume reduced: keep every set controlled and leave reps in reserve."
        else:
            next_sets = target or current_sets
            note = "Set volume adjusted by request: keep movement quality consistent across all sets."
        changed = changed or next_sets != current_sets
        patched["sets"] = next_sets
        existing_note = str(patched.get("notes", "")).strip()
        if note not in existing_note:
            patched["notes"] = f"{existing_note} {note}".strip()
        updated_exercises.append(patched)

    if not updated_exercises:
        return ""
    today_session["exercises"] = updated_exercises
    today_session.pop("is_cancelled", None)
    today_session["duration_minutes"] = int(today_session.get("duration_minutes") or 60)
    refresh_youtube_resources(updated_result)
    session_state["agent_result"] = updated_result
    action_type = f"sets_{set_adjustment}"
    record_memory_plan_modification(
        session_state=session_state,
        date_iso=current_date,
        action_type=action_type,
        summary="Adjusted today's workout sets without changing exercise count.",
    )

    exercise_text = ", ".join(
        f"{exercise.get('name', '')} {exercise.get('sets', '')}x{exercise.get('reps', '')}".strip()
        for exercise in updated_exercises
    )
    if not changed:
        return f"Today's workout already matches the requested set boundary: {exercise_text}."
    return f"Today's workout sets were updated without adding exercises: {exercise_text}."


def update_nutrition_from_chat(
    normalized: dict[str, Any],
    previous_result: FitnessAgentState | dict[str, Any],
    session_state: dict[str, Any],
) -> str:
    avoidances = coerce_string_list(normalized.get("temporary_food_avoidances"), [])
    if not avoidances or normalized_request_has_workout_change(normalized):
        return ""

    updated_result = deepcopy(previous_result)
    current_plan = updated_result.get("current_plan", {})
    meals = list(current_plan.get("meal_suggestions", []))
    if not meals:
        return ""

    updated_meals: list[dict[str, Any]] = []
    replacements: list[tuple[str, str]] = []
    used_foods = {
        str(meal.get("food_name", "")).strip()
        for meal in meals
        if str(meal.get("food_name", "")).strip()
    }
    for meal in meals:
        food_name = str(meal.get("food_name", "")).strip()
        if not food_matches_any_avoidance(food_name, avoidances):
            updated_meals.append(meal)
            continue
        replacement = replacement_food_for_meal(meal=meal, avoidances=avoidances, used_foods=used_foods)
        if replacement:
            replacement_name = str(replacement.get("food_name", "")).strip()
            replacements.append((food_name, replacement_name))
            used_foods.add(replacement_name)
            updated_meals.append(replacement)

    if not replacements:
        return ""

    current_plan["meal_suggestions"] = updated_meals
    session_state["agent_result"] = updated_result
    current_date = current_interaction_date(active_date=session_state.get("active_date"), result=previous_result)
    record_memory_food_avoidance(session_state=session_state, date_iso=current_date, avoidances=avoidances, scope="today_only")
    record_memory_plan_modification(
        session_state=session_state,
        date_iso=current_date,
        action_type="replace_food",
        summary=f"Replaced today's foods: {', '.join(old for old, _ in replacements)}.",
    )
    replacement_text = ", ".join(f"{old} -> {new}" for old, new in replacements)
    return f"Today's nutrition was updated: {replacement_text}. Your workout plan was left unchanged."


def replace_today_exercise_from_chat(
    user_message: str,
    previous_result: FitnessAgentState | dict[str, Any],
    session_state: dict[str, Any],
) -> str:
    if not chat_message_requests_exercise_replacement(user_message):
        return ""

    current_date = current_interaction_date(active_date=session_state.get("active_date"), result=previous_result)
    current_plan = previous_result.get("current_plan", {})
    today_session = select_today_session(sort_workout_sessions(current_plan.get("workout_sessions", [])), current_date)
    if not today_session or today_session.get("is_cancelled"):
        return ""

    exercises = today_session.get("exercises", [])
    target_index = exercise_index_from_chat_message(user_message, exercises)
    if target_index is None and not chat_requests_general_exercise_replacement(user_message):
        return ""

    updated_result = deepcopy(previous_result)
    updated_sessions = updated_result.get("current_plan", {}).get("workout_sessions", [])
    updated_today = select_today_session(sort_workout_sessions(updated_sessions), current_date)
    if not updated_today:
        return ""

    updated_exercises = list(updated_today.get("exercises", []))
    current_names = [str(exercise.get("name", "")).strip() for exercise in updated_exercises if str(exercise.get("name", "")).strip()]
    focus_tag = focus_tag_from_session(today_session)
    level = str(previous_result.get("user_profile", {}).get("fitness_level", ""))

    if target_index is None:
        replacements_made = []
        excluded_names = list(current_names)
        for index, original_exercise in enumerate(updated_exercises):
            original_name = str(original_exercise.get("name", "")).strip()
            if not original_name:
                continue
            replacements = search_similar_exercises(
                exercise_name=original_name,
                focus=focus_tag,
                level=level,
                exclude=excluded_names,
                limit=5,
            )
            if not replacements:
                continue
            replacement = replacements[0]
            updated_exercises[index] = replacement_exercise_payload(original_exercise, replacement)
            replacement_name = str(replacement.get("name", "")).strip()
            excluded_names.append(replacement_name)
            replacements_made.append((original_name, replacement_name))
        if not replacements_made:
            return ""
        updated_today["exercises"] = updated_exercises
        refresh_youtube_resources(updated_result)
        session_state["agent_result"] = updated_result
        record_memory_plan_modification(
            session_state=session_state,
            date_iso=current_date,
            action_type="replace_exercise",
            summary="Replaced today's exercises with same-focus alternatives.",
        )
        replacement_text = ", ".join(f"{old} -> {new}" for old, new in replacements_made)
        return f"Today's plan is now updated with same-focus exercise alternatives: {replacement_text}. Sets and reps stayed the same."

    original_exercise = updated_exercises[target_index]
    original_name = str(original_exercise.get("name", "")).strip()
    if not original_name:
        return ""
    replacements = search_similar_exercises(
        exercise_name=original_name,
        focus=focus_tag,
        level=level,
        exclude=current_names,
        limit=5,
    )
    if not replacements:
        return ""

    replacement = replacements[0]
    updated_exercises[target_index] = replacement_exercise_payload(original_exercise, replacement)
    updated_today["exercises"] = updated_exercises
    refresh_youtube_resources(updated_result)
    session_state["agent_result"] = updated_result
    record_memory_plan_modification(
        session_state=session_state,
        date_iso=current_date,
        action_type="replace_exercise",
        summary=f"Replaced {original_name} with {replacement.get('name', '')}.",
    )
    return f"Today's plan is now updated: replaced {original_name} with {replacement.get('name', '')} while keeping the same focus, sets, and reps."


def execute_ai_plan_patch(
    *,
    user_message: str,
    profile_inputs: dict[str, Any],
    previous_result: FitnessAgentState | dict[str, Any],
    normalized: dict[str, Any],
    session_state: dict[str, Any],
) -> str:
    state = build_change_request_state(
        profile_inputs=profile_inputs,
        previous_result=previous_result,
        change_request=user_message,
        normalized_change_request=normalized,
        session_state=session_state,
    )
    result = run_agent(state)
    patched_result = merge_ai_patch_result(previous_result, result, normalized)
    session_state["agent_result"] = patched_result
    record_memory_plan_modification(
        session_state=session_state,
        date_iso=str(state.get("current_date", "")),
        action_type="ai_plan_patch",
        summary=str(normalized.get("summary") or state.get("plan_change_request") or "AI Coach updated today's plan."),
        injury_areas=coerce_string_list(normalized.get("injury_areas"), []),
    )
    today_session = select_today_session(
        sort_workout_sessions(patched_result.get("current_plan", {}).get("workout_sessions", [])),
        current_interaction_date(active_date=session_state.get("active_date"), result=patched_result),
    )
    return summarize_session_for_chat(today_session)


def build_change_request_state(
    *,
    profile_inputs: dict[str, Any],
    previous_result: FitnessAgentState | dict[str, Any],
    change_request: str,
    normalized_change_request: dict[str, Any],
    session_state: dict[str, Any],
) -> FitnessAgentState:
    target_date = current_interaction_date(active_date=session_state.get("active_date"), result=previous_result)
    previous_feedback = dict(previous_result.get("latest_feedback", {}))
    previous_state = dict(previous_result.get("current_state", {}))
    change_request_context = build_change_request_context(change_request, normalized_change_request)
    previous_notes = " ".join(part for part in [previous_state.get("notes", ""), change_request_context] if part)
    previous_feedback["performance_notes"] = previous_notes

    base_state = build_initial_state(
        profile_inputs=profile_inputs,
        thread_id=session_state["thread_id"],
        active_date=target_date,
        memory_store=normalize_memory_store(session_state.get("memory_store")),
    )
    return {
        "thread_id": session_state["thread_id"],
        "current_date": target_date,
        "profile_notes": profile_inputs.get("profile_notes", ""),
        "plan_change_request": change_request,
        "normalized_change_request": normalized_change_request,
        "user_profile": base_state["user_profile"],
        "constraints": base_state["constraints"],
        "goals": base_state["goals"],
        "current_state": {**previous_state, "date": target_date, "notes": previous_notes},
        "latest_feedback": {**previous_feedback, "date": target_date},
        "current_plan": previous_result.get("current_plan", {}),
        "plan_history": previous_result.get("plan_history", []),
        "daily_history": previous_result.get("daily_history", []),
        "feedback_history": previous_result.get("feedback_history", []),
        "state_history": append_unique_history_item(
            previous_result.get("state_history", []),
            previous_result.get("current_state", {}),
            "date",
        ),
        "memory_context": base_state.get("memory_context", {}),
    }


def merge_ai_patch_result(
    previous_result: FitnessAgentState | dict[str, Any],
    result: FitnessAgentState | dict[str, Any],
    normalized: dict[str, Any],
) -> FitnessAgentState:
    patched_result: FitnessAgentState = dict(result)
    previous_plan = deepcopy(previous_result.get("current_plan", {}))
    result_plan = deepcopy(result.get("current_plan", {}))

    request_type = str(normalized.get("request_type", "")).strip()
    workout_changed = normalized_request_has_workout_change(normalized)
    nutrition_changed = bool(
        normalized.get("temporary_food_avoidances")
        or normalized.get("permanent_food_preferences")
        or request_type == "nutrition_change"
    )
    if workout_changed and not nutrition_changed and previous_plan:
        result_plan["nutrition_targets"] = deepcopy(previous_plan.get("nutrition_targets", result_plan.get("nutrition_targets", {})))
        result_plan["meal_suggestions"] = deepcopy(previous_plan.get("meal_suggestions", result_plan.get("meal_suggestions", [])))
    elif nutrition_changed and not workout_changed and previous_plan:
        result_plan["workout_sessions"] = deepcopy(previous_plan.get("workout_sessions", result_plan.get("workout_sessions", [])))

    if workout_changed and previous_plan and not str(normalized.get("set_adjustment", "")).strip():
        target_date = str(result.get("current_date") or previous_result.get("current_date") or "")
        previous_today = select_today_session(sort_workout_sessions(previous_plan.get("workout_sessions", [])), target_date)
        result_today = select_today_session(sort_workout_sessions(result_plan.get("workout_sessions", [])), target_date)
        previous_set_policy = set_policy_from_session(previous_today)
        if result_today and previous_set_policy != 4:
            apply_set_policy_to_session(result_today, previous_set_policy)

    if previous_plan:
        for key in ["summary", "objective_alignment", "coaching_focus", "recovery_actions"]:
            if key in previous_plan:
                result_plan[key] = deepcopy(previous_plan[key])

    patched_result["current_plan"] = result_plan
    patched_result["plan_history"] = deepcopy(previous_result.get("plan_history", []))
    patched_result["daily_history"] = deepcopy(previous_result.get("daily_history", []))
    patched_result["feedback_history"] = deepcopy(previous_result.get("feedback_history", []))
    patched_result["state_history"] = deepcopy(previous_result.get("state_history", []))
    refresh_youtube_resources(patched_result)
    return patched_result


def build_chat_context(result: FitnessAgentState | dict[str, Any], session_state: dict[str, Any]) -> str:
    if not result:
        return "No plan has been generated yet."
    current_date = current_interaction_date(active_date=session_state.get("active_date"), result=result)
    current_plan = result.get("current_plan", {})
    today_session = select_today_session(sort_workout_sessions(current_plan.get("workout_sessions", [])), current_date)
    exercises = [
        f"{exercise.get('name', '')} {exercise.get('sets', '')}x{exercise.get('reps', '')}".strip()
        for exercise in today_session.get("exercises", [])
    ] if today_session else []
    latest_feedback = result.get("latest_feedback", {})
    return json.dumps(
        {
            "active_date": current_date,
            "goal": result.get("goals", {}).get("primary_goal", ""),
            "fitness_level": result.get("user_profile", {}).get("fitness_level", ""),
            "planning_rules": {
                "baseline_beginner_exercises": 2,
                "baseline_intermediate_exercises": 3,
                "baseline_advanced_exercises": 4,
                "default_sets_per_exercise": 4,
                "explicit_set_change_bounds": "3-5 sets",
                "minimum_exercises": 2,
            },
            "cycle": {
                "number": current_plan.get("cycle_number"),
                "start": current_plan.get("cycle_start_date"),
                "end": current_plan.get("cycle_end_date"),
            },
            "today_session": {
                "focus": today_session.get("focus", "") if today_session else "",
                "scheduled_date": today_session.get("scheduled_date", "") if today_session else "",
                "is_cancelled": bool(today_session.get("is_cancelled")) if today_session else False,
                "exercises": exercises,
            },
            "nutrition_targets": current_plan.get("nutrition_targets", {}),
            "latest_feedback": {
                "date": latest_feedback.get("date", ""),
                "emoji": latest_feedback.get("feeling_emoji", ""),
                "notes": latest_feedback.get("performance_notes", ""),
            },
        },
        ensure_ascii=True,
    )


def normalize_change_request(change_request: str) -> dict[str, Any]:
    if not change_request.strip():
        return {}
    try:
        normalized = call_model_json(
            system_prompt=load_prompt("change_request_prompt.txt"),
            user_prompt=change_request,
            temperature=0.0,
            max_tokens=600,
        )
    except Exception:
        normalized = {}
    return sanitize_normalized_change_request(normalized)


def sanitize_normalized_change_request(normalized: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(normalized, dict):
        normalized = {}
    request_type = str(normalized.get("request_type", "unclear")).strip()
    scope = str(normalized.get("scope", "unclear")).strip()
    focus_category = str(normalized.get("focus_category", "")).strip()
    return {
        "request_type": request_type if request_type in ALLOWED_CHANGE_REQUEST_TYPES else "unclear",
        "scope": scope if scope in ALLOWED_CHANGE_REQUEST_SCOPES else "unclear",
        "focus_category": focus_category if focus_category in ALLOWED_FOCUS_CATEGORIES else "",
        "injury_reported": bool(normalized.get("injury_reported", False)),
        "injury_areas": coerce_string_list(normalized.get("injury_areas"), []),
        "cancel_today": bool(normalized.get("cancel_today", False)),
        "intensity_adjustment": str(normalized.get("intensity_adjustment", "")).strip(),
        "set_adjustment": str(normalized.get("set_adjustment", "")).strip(),
        "set_target": clamp_int(normalized.get("set_target"), 0, 12, 0),
        "duration_adjustment": str(normalized.get("duration_adjustment", "")).strip(),
        "temporary_food_avoidances": coerce_string_list(normalized.get("temporary_food_avoidances"), []),
        "permanent_food_preferences": coerce_string_list(normalized.get("permanent_food_preferences"), []),
        "summary": str(normalized.get("summary", "")).strip(),
        "confidence": clamp_float(normalized.get("confidence"), 0.0, 1.0, 0.0),
    }


def build_change_request_context(change_request: str, normalized_change_request: dict[str, Any]) -> str:
    summary = str(normalized_change_request.get("summary", "")).strip()
    request_type = str(normalized_change_request.get("request_type", "unclear")).strip()
    scope = str(normalized_change_request.get("scope", "unclear")).strip()
    focus = str(normalized_change_request.get("focus_category", "")).strip()
    pieces = [f"User requested: {change_request}", f"Type={request_type}", f"Scope={scope}"]
    if focus:
        pieces.append(f"Focus={focus}")
    if normalized_change_request.get("intensity_adjustment"):
        pieces.append(f"Intensity={normalized_change_request['intensity_adjustment']}")
    if normalized_change_request.get("set_adjustment"):
        pieces.append(f"Set adjustment={normalized_change_request['set_adjustment']}")
    if normalized_change_request.get("set_target"):
        pieces.append(f"Set target={normalized_change_request['set_target']}")
    if summary:
        pieces.append(f"Summary={summary}")
    return " | ".join(pieces)


def patch_exercises_for_intensity(
    *,
    exercises: list[dict[str, Any]],
    fitness_level: str,
    intensity: str,
    focus: str,
) -> list[dict[str, Any]]:
    patched = [dict(exercise) for exercise in exercises]
    current_set_policy = set_policy_from_exercises(patched)
    if intensity == "lower":
        target_count = 2 if fitness_level in {"beginner", "intermediate"} else 3
        reps = "6-8" if fitness_level == "beginner" else "10-12"
        note = "Lower intensity: keep the load conservative, move with control, and stop well before form breaks."
    else:
        reps = "8-10" if fitness_level == "beginner" else "12-15"
        note = "Higher intensity: use controlled tempo, full range of motion, and a brief pause while keeping clean form."

    if intensity == "lower":
        patched = patched[: max(2, min(target_count, len(patched)))]
    elif patched:
        current_names = [str(exercise.get("name", "")).strip() for exercise in patched if str(exercise.get("name", "")).strip()]
        seed_name = current_names[0] if current_names else ""
        additions = search_similar_exercises(
            exercise_name=seed_name,
            focus=focus,
            level=fitness_level,
            exclude=current_names,
            limit=3,
        ) if seed_name else []
        if additions:
            patched.append(replacement_exercise_payload({"sets": current_set_policy, "reps": reps}, additions[0]))
    for exercise in patched:
        exercise["sets"] = current_set_policy
        exercise["reps"] = reps
        existing_note = str(exercise.get("notes", "")).strip()
        if note not in existing_note:
            exercise["notes"] = f"{existing_note} {note}".strip()
    return patched


def protect_future_sessions_for_injury(
    *,
    updated_result: FitnessAgentState | dict[str, Any],
    current_date: str,
    injury_areas: list[str],
) -> list[str]:
    current_plan = updated_result.get("current_plan", {})
    sessions = sort_workout_sessions(current_plan.get("workout_sessions", []))
    current_cycle = cycle_number_for_date(current_plan, current_date)
    protected_labels: list[str] = []
    for session in sessions:
        session_date = safe_iso_date(session.get("scheduled_date"))
        if not session_date or session_date <= current_date:
            continue
        if session.get("cycle_number", current_cycle) != current_cycle:
            continue
        if session.get("is_cancelled") or not session_stresses_injury_area(session, injury_areas):
            continue
        session["is_cancelled"] = True
        session["focus"] = "Recovery (injury protection)"
        session["duration_minutes"] = 0
        session["warmup"] = []
        session["exercises"] = []
        session["cooldown"] = []
        session["injury_reported"] = True
        session["injury_areas"] = injury_areas
        session["safety_notes"] = [
            f"Protected because a recent injury was reported around {', '.join(injury_areas)}.",
            "Resume this focus only after later feedback indicates symptoms have improved.",
        ]
        protected_labels.append(f"{session_date} {session.get('day', '')}".strip())
    return protected_labels


def cycle_number_for_date(current_plan: dict[str, Any], date_iso: str) -> int:
    for session in current_plan.get("workout_sessions", []):
        if same_iso_date(session.get("scheduled_date"), date_iso):
            return int(session.get("cycle_number") or current_plan.get("cycle_number") or 1)
    return int(current_plan.get("cycle_number") or 1)


def session_stresses_injury_area(session: dict[str, Any], injury_areas: list[str]) -> bool:
    session_text = " ".join(
        [
            str(session.get("focus", "")),
            " ".join(str(exercise.get("name", "")) for exercise in session.get("exercises", [])),
            " ".join(str(exercise.get("target_muscle", "")) for exercise in session.get("exercises", [])),
        ]
    ).lower()
    stress_aliases = {
        "back": ["back", "lat", "row", "pulldown", "deadlift", "hinge", "squat"],
        "knee": ["lower body", "leg", "squat", "lunge", "knee", "glute"],
        "shoulder": ["shoulder", "press", "chest", "push", "row"],
        "ankle": ["lower body", "leg", "lunge", "squat", "jump", "conditioning"],
        "hip": ["lower body", "leg", "glute", "squat", "lunge", "hinge"],
        "wrist": ["push", "press", "curl", "row", "plank"],
        "elbow": ["push", "press", "curl", "row"],
    }
    for area in injury_areas or ["reported injury area"]:
        normalized_area = str(area).lower()
        aliases = stress_aliases.get(normalized_area, [normalized_area])
        if any(alias in session_text for alias in aliases):
            return True
    return False


def cancel_safety_notes(*, injury_reported: bool, injury_areas: list[str]) -> list[str]:
    if injury_reported:
        area_text = f" around {', '.join(injury_areas)}" if injury_areas else ""
        return [
            f"Today's workout was cancelled because an injury was reported{area_text}.",
            "Do not train through injury symptoms; consider medical guidance before resuming.",
        ]
    return ["Today's workout was cancelled at your request."]


def augment_chat_safety_change(user_message: str, normalized: dict[str, Any]) -> dict[str, Any]:
    if normalized.get("injury_reported") or normalized.get("cancel_today"):
        return normalized
    injury_areas = injury_areas_from_chat_text(user_message)
    cancel_requested = chat_text_requests_cancel(user_message)
    if not injury_areas and not cancel_requested:
        return normalized
    augmented = dict(normalized)
    augmented["request_type"] = "recovery_change" if injury_areas else "workout_change"
    augmented["scope"] = "today_only"
    augmented["cancel_today"] = True
    augmented["injury_reported"] = bool(injury_areas)
    augmented["injury_areas"] = injury_areas
    summary = "User reported injury; cancel today's workout." if injury_areas else "User requested cancelling today's workout without an injury reason."
    augmented["summary"] = " ".join(part for part in [str(augmented.get("summary", "")).strip(), summary] if part)
    augmented["confidence"] = max(float(augmented.get("confidence") or 0.0), 0.85)
    return augmented


def augment_chat_food_change(user_message: str, normalized: dict[str, Any], result: FitnessAgentState | dict[str, Any]) -> dict[str, Any]:
    if coerce_string_list(normalized.get("temporary_food_avoidances"), []):
        return normalized
    avoidances = food_avoidances_from_chat_text(user_message, result)
    if not avoidances:
        return normalized
    augmented = dict(normalized)
    augmented["request_type"] = "nutrition_change"
    augmented["scope"] = "today_only"
    augmented["temporary_food_avoidances"] = avoidances
    summary = f"User wants to avoid {', '.join(avoidances)} today."
    augmented["summary"] = " ".join(part for part in [str(augmented.get("summary", "")).strip(), summary] if part)
    augmented["confidence"] = max(float(augmented.get("confidence") or 0.0), 0.85)
    return augmented


def augment_chat_focus_change(user_message: str, normalized: dict[str, Any]) -> dict[str, Any]:
    if normalized.get("focus_category") or normalized.get("injury_reported") or normalized.get("cancel_today"):
        return normalized
    focus_category = focus_category_from_chat_text(user_message)
    if not focus_category:
        return normalized
    augmented = dict(normalized)
    augmented["request_type"] = "workout_change"
    augmented["scope"] = "today_only"
    augmented["focus_category"] = focus_category
    summary = f"User asked to train focus category={focus_category} today."
    augmented["summary"] = " ".join(part for part in [str(augmented.get("summary", "")).strip(), summary] if part)
    augmented["confidence"] = max(float(augmented.get("confidence") or 0.0), 0.85)
    return augmented


def augment_chat_set_change(user_message: str, normalized: dict[str, Any]) -> dict[str, Any]:
    if normalized.get("set_adjustment") or normalized.get("injury_reported") or normalized.get("cancel_today"):
        return normalized
    set_adjustment, set_target = set_adjustment_from_text(user_message)
    if not set_adjustment:
        return normalized
    augmented = dict(normalized)
    request_type = str(augmented.get("request_type", ""))
    augmented["request_type"] = request_type if request_type in ALLOWED_CHANGE_REQUEST_TYPES and request_type not in {"none", "unclear", ""} else "workout_change"
    scope = str(augmented.get("scope", ""))
    augmented["scope"] = scope if scope in ALLOWED_CHANGE_REQUEST_SCOPES and scope not in {"unclear", ""} else "today_only"
    augmented["set_adjustment"] = set_adjustment
    augmented["set_target"] = set_target
    augmented["intensity_adjustment"] = ""
    summary = str(augmented.get("summary", "")).strip()
    if set_adjustment == "target":
        set_text = f"User explicitly requested {set_target} sets today."
    elif set_adjustment == "increase":
        set_text = "User explicitly requested more sets today."
    else:
        set_text = "User explicitly requested fewer sets today."
    augmented["summary"] = " ".join(part for part in [summary, set_text] if part)
    augmented["confidence"] = max(float(augmented.get("confidence") or 0.0), 0.88)
    return augmented


def augment_chat_intensity_change(user_message: str, normalized: dict[str, Any]) -> dict[str, Any]:
    if normalized.get("set_adjustment") or normalized.get("intensity_adjustment") or normalized.get("injury_reported") or normalized.get("cancel_today"):
        return normalized
    fallback_intensity = chat_intensity_from_text(user_message)
    if not fallback_intensity:
        return normalized
    augmented = dict(normalized)
    request_type = str(augmented.get("request_type", ""))
    augmented["request_type"] = request_type if request_type in ALLOWED_CHANGE_REQUEST_TYPES and request_type not in {"none", "unclear", ""} else "workout_change"
    scope = str(augmented.get("scope", ""))
    augmented["scope"] = scope if scope in ALLOWED_CHANGE_REQUEST_SCOPES and scope not in {"unclear", ""} else "today_only"
    augmented["intensity_adjustment"] = fallback_intensity
    summary = str(augmented.get("summary", "")).strip()
    intensity_text = "higher intensity" if fallback_intensity == "higher" else "lower intensity"
    augmented["summary"] = " ".join(part for part in [summary, f"User asked for {intensity_text} today."] if part)
    return augmented


def chat_text_requests_cancel(text: str) -> bool:
    normalized = text.lower()
    cancel_terms = ["cancel today's plan", "cancel today", "cancel workout", "skip today", "no workout today", "取消今天", "取消训练", "今天不练"]
    return any(term in normalized for term in cancel_terms)


def injury_areas_from_chat_text(text: str) -> list[str]:
    normalized = text.lower()
    injury_terms = ["injured", "injury", "hurt", "pain", "strain", "sprain", "受伤", "疼", "痛"]
    if not any(term in normalized for term in injury_terms):
        return []
    area_map = {
        "back": ["back", "lower back", "背", "腰"],
        "knee": ["knee", "knees", "膝盖"],
        "shoulder": ["shoulder", "shoulders", "肩"],
        "ankle": ["ankle", "ankles", "脚踝"],
        "wrist": ["wrist", "wrists", "手腕"],
        "elbow": ["elbow", "elbows", "肘"],
        "hip": ["hip", "hips", "髋"],
    }
    areas = [area for area, aliases in area_map.items() if any(alias in normalized for alias in aliases)]
    return areas or ["reported injury area"]


def food_avoidances_from_chat_text(user_message: str, result: FitnessAgentState | dict[str, Any]) -> list[str]:
    text = user_message.lower()
    loose_text = normalize_loose(user_message)
    food_change_terms = ["don't want", "dont want", "do not want", "not want", "avoid", "no ", "replace", "swap", "instead of", "不想吃", "不要吃", "不吃", "换掉", "替换"]
    if not any(term in text for term in food_change_terms):
        return []
    current_food_names = [
        str(meal.get("food_name", "")).strip()
        for meal in result.get("current_plan", {}).get("meal_suggestions", [])
        if str(meal.get("food_name", "")).strip()
    ]
    matched: list[str] = []
    for food_name in current_food_names:
        loose_food = normalize_loose(food_name)
        first_word = normalize_loose(food_name.split(",", 1)[0].split("(", 1)[0])
        if loose_food and loose_food in loose_text:
            matched.append(food_name)
        elif first_word and first_word in loose_text:
            matched.append(food_name)
    if matched:
        return dedupe_preserve_order(matched)
    known_food_terms = ["broccoli", "salmon", "chicken", "rice", "yogurt", "egg", "beef", "tofu", "oats", "banana"]
    return [food for food in known_food_terms if food in text]


def focus_category_from_chat_text(user_message: str) -> str:
    text = user_message.lower()
    if injury_areas_from_chat_text(user_message):
        return ""
    action_terms = ["do", "train", "workout", "add", "switch", "change", "focus", "i want", "can i", "could i", "练", "训练", "加练", "换成", "改成"]
    if not any(term in text for term in action_terms):
        return ""
    focus_aliases = [
        ("upper_shoulders", ["shoulder", "shoulders", "delts", "肩"]),
        ("back_training", ["back training", "train back", "work back", "lats", "row", "背"]),
        ("lower_legs_glutes", ["lower body", "legs", "leg", "glutes", "glute", "squat", "腿", "臀"]),
        ("functional_core", ["core", "abs", "腹", "核心"]),
        ("functional_power", ["power", "explosive", "爆发"]),
        ("functional_conditioning", ["conditioning", "cardio", "functional", "体能", "功能性", "调节"]),
        ("upper_chest_arms", ["chest", "arms", "biceps", "triceps", "push", "胸", "手臂", "二头", "三头"]),
    ]
    for category, aliases in focus_aliases:
        if any(alias in text for alias in aliases):
            return category
    return ""


def chat_intensity_from_text(text: str) -> str:
    normalized = text.lower()
    set_adjustment, _ = set_adjustment_from_text(text)
    if set_adjustment:
        return ""
    lower_terms = ["tired", "uncomfortable", "not feeling good", "not good", "low energy", "poor sleep", "bad sleep", "didn't sleep", "dont sleep", "don't sleep", "slept badly", "sleep badly", "exhausted", "fatigued", "make it easier", "easier", "lighter", "less", "reduce", "不舒服", "累", "状态不好", "轻一点", "减量"]
    higher_terms = ["energized", "excited", "feel good", "feeling good", "strong", "do more", "more", "harder", "challenge", "too easy", "increase", "加量", "加强", "状态很好", "很兴奋", "多一点"]
    if any(term in normalized for term in lower_terms):
        return "lower"
    if any(term in normalized for term in higher_terms):
        return "higher"
    return ""


def set_adjustment_from_text(text: str) -> tuple[str, int]:
    normalized = text.lower()
    if not any(term in normalized for term in ["set", "sets", "组"]):
        return "", 0

    target_match = re.search(r"\b(?:to|do|make it|make them)?\s*([2-9])\s*sets?\b", normalized)
    if target_match:
        return "target", int(target_match.group(1))
    chinese_target_match = re.search(r"([二三四五六七八九23456789])\s*组", normalized)
    if chinese_target_match and any(term in normalized for term in ["改", "做", "变", "到", "成"]):
        target = {"二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}.get(chinese_target_match.group(1), 0) or int(chinese_target_match.group(1))
        return "target", target

    increase_terms = [
        "add set",
        "add sets",
        "more set",
        "more sets",
        "extra set",
        "extra sets",
        "one more set",
        "increase sets",
        "increase set",
        "加组",
        "多一组",
        "多几组",
        "增加组",
    ]
    decrease_terms = [
        "reduce set",
        "reduce sets",
        "fewer set",
        "fewer sets",
        "less set",
        "less sets",
        "one less set",
        "decrease sets",
        "decrease set",
        "少一组",
        "少几组",
        "减少组",
        "减组",
    ]
    if any(term in normalized for term in decrease_terms):
        return "decrease", 0
    if any(term in normalized for term in increase_terms):
        return "increase", 0
    return "", 0


def chat_request_should_update_today(user_message: str, normalized: dict[str, Any]) -> bool:
    text = user_message.lower()
    update_terms = ["update", "change", "switch", "add", "make today", "today", "can i", "could i", "i want", "want", "do more", "energized", "excited", "strong", "easier", "lighter", "harder", "injury", "injured", "hurt", "pain", "sleep", "slept", "didn't sleep", "dont sleep", "don't sleep", "poor sleep", "let's", "练", "改", "换", "加", "今天"]
    has_update_language = any(term in text for term in update_terms)
    has_structured_change = bool(normalized.get("focus_category") or normalized.get("cancel_today") or normalized.get("set_adjustment") or normalized.get("intensity_adjustment") or normalized.get("temporary_food_avoidances") or normalized.get("permanent_food_preferences"))
    request_type = str(normalized.get("request_type", ""))
    if normalized.get("injury_reported") or normalized.get("cancel_today"):
        return has_structured_change
    return has_update_language and has_structured_change and request_type in {"workout_change", "nutrition_change", "mixed_change", "recovery_change"}


def chat_message_requests_exercise_replacement(user_message: str) -> bool:
    text = user_message.lower()
    replacement_terms = ["replace", "swap", "change", "alternative", "instead", "换", "替换", "不要", "不想做", "换掉", "改掉"]
    return any(term in text for term in replacement_terms)


def chat_requests_general_exercise_replacement(user_message: str) -> bool:
    text = user_message.lower()
    exercise_terms = ["exercise", "exercises", "action", "actions", "movement", "movements", "动作"]
    general_terms = ["some", "all", "another", "alternative", "alternatives", "different", "new", "几个", "一些", "全部", "换一换", "换几个", "换掉"]
    return any(term in text for term in exercise_terms) and any(term in text for term in general_terms)


def exercise_index_from_chat_message(user_message: str, exercises: list[dict[str, Any]]) -> int | None:
    text = user_message.lower()
    normalized_text = normalize_loose(text)
    for index, exercise in enumerate(exercises):
        name = str(exercise.get("name", "")).strip()
        if name and normalize_loose(name) in normalized_text:
            return index
    ordinal_terms = [
        (0, ["first exercise", "first movement", "1st exercise", "第一个动作", "第1个动作", "第一个", "第1个"]),
        (1, ["second exercise", "second movement", "2nd exercise", "第二个动作", "第2个动作", "第二个", "第2个"]),
        (2, ["third exercise", "third movement", "3rd exercise", "第三个动作", "第3个动作", "第三个", "第3个"]),
        (3, ["fourth exercise", "fourth movement", "4th exercise", "第四个动作", "第4个动作", "第四个", "第4个"]),
        (4, ["fifth exercise", "fifth movement", "5th exercise", "第五个动作", "第5个动作", "第五个"]),
    ]
    for index, terms in ordinal_terms:
        if index < len(exercises) and any(term in text for term in terms):
            return index
    return None


def replacement_exercise_payload(original_exercise: dict[str, Any], replacement: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": replacement.get("name", ""),
        "target_muscle": ", ".join(replacement.get("target_muscle", [])),
        "sets": original_exercise.get("sets", 4),
        "reps": original_exercise.get("reps", ""),
        "equipment": ", ".join(replacement.get("equipment", [])),
        "notes": replacement.get("notes", ""),
    }


def set_policy_from_session(session: dict[str, Any]) -> int:
    return set_policy_from_exercises(session.get("exercises", []) if session else [])


def set_policy_from_exercises(exercises: list[dict[str, Any]]) -> int:
    set_values = [
        clamp_int(exercise.get("sets"), 3, 5, 4)
        for exercise in exercises
        if isinstance(exercise, dict)
    ]
    if not set_values:
        return 4
    counts = {sets: set_values.count(sets) for sets in set_values}
    return max(counts, key=lambda sets: (counts[sets], sets))


def apply_set_policy_to_session(session: dict[str, Any], set_policy: int) -> None:
    bounded_policy = clamp_int(set_policy, 3, 5, 4)
    for exercise in session.get("exercises", []):
        if isinstance(exercise, dict):
            exercise["sets"] = bounded_policy


def focus_tag_from_session(session: dict[str, Any]) -> str:
    focus = str(session.get("focus", "")).lower()
    if "shoulder" in focus:
        return "upper_shoulders"
    if "back" in focus:
        return "back_training"
    if "lower" in focus or "legs" in focus or "glutes" in focus:
        return "lower_legs_glutes"
    if "core" in focus or "abs" in focus:
        return "functional_core"
    if "power" in focus:
        return "functional_power"
    if "conditioning" in focus:
        return "functional_conditioning"
    if "chest" in focus or "arms" in focus:
        return "upper_chest_arms"
    return ""


def normalized_request_has_workout_change(normalized: dict[str, Any]) -> bool:
    return bool(normalized.get("focus_category") or normalized.get("cancel_today") or normalized.get("injury_reported") or normalized.get("intensity_adjustment") or str(normalized.get("request_type", "")) in {"workout_change", "mixed_change", "recovery_change"})


def food_matches_any_avoidance(food_name: str, avoidances: list[str]) -> bool:
    normalized_name = normalize_loose(food_name)
    return any(normalize_loose(avoidance) and normalize_loose(avoidance) in normalized_name for avoidance in avoidances)


def replacement_food_for_meal(*, meal: dict[str, Any], avoidances: list[str], used_foods: set[str]) -> dict[str, Any] | None:
    current_food = get_food_by_name(str(meal.get("food_name", "")))
    if not current_food:
        return None
    category = str(current_food.get("category", ""))
    candidates = find_foods(category=category, limit=8)
    for candidate in candidates:
        candidate_name = str(candidate.get("name", "")).strip()
        if not candidate_name or candidate_name in used_foods or food_matches_any_avoidance(candidate_name, avoidances):
            continue
        grams = extract_grams(str(meal.get("serving_size", "100g")), default=100.0)
        macro = calculate_food_macros(str(candidate.get("id", "")), grams)
        return {
            "food_name": candidate_name,
            "serving_size": f"{int(grams)}g",
            "calories": int(round(macro["calories"])),
            "protein_g": macro["protein_g"],
            "carbs_g": macro["carbs_g"],
            "fat_g": macro["fat_g"],
            "meal_slot": str(meal.get("meal_slot", "meal")),
        }
    return None


def extract_grams(serving_size: str, default: float) -> float:
    digits = "".join(char for char in serving_size if char.isdigit() or char == ".")
    return float(digits) if digits else default


def summarize_session_for_chat(session: dict[str, Any]) -> str:
    if not session:
        return "Today is not a scheduled training day."
    if session.get("is_cancelled"):
        return "Today's workout is cancelled by the app safety rules."
    exercises = [f"{exercise.get('name', '')} {exercise.get('sets', '')}x{exercise.get('reps', '')}".strip() for exercise in session.get("exercises", [])]
    exercise_text = ", ".join(exercises) if exercises else "no exercises"
    return f"Today's plan is now {session.get('focus', 'training')}: {exercise_text}."


def select_today_session(workout_sessions: list[dict[str, Any]], reference_date: str) -> dict[str, Any]:
    for session in workout_sessions:
        if same_iso_date(session.get("scheduled_date"), reference_date):
            return session
    return {}


def refresh_youtube_resources(result: FitnessAgentState | dict[str, Any]) -> None:
    exercise_names = [str(exercise.get("name", "")).strip() for session in result.get("current_plan", {}).get("workout_sessions", []) for exercise in session.get("exercises", []) if str(exercise.get("name", "")).strip()]
    result["youtube_resources"] = build_video_resources(exercise_names)


def record_memory_plan_modification(*, session_state: dict[str, Any], date_iso: str, action_type: str, summary: str, injury_areas: list[str] | None = None) -> None:
    store = normalize_memory_store(session_state.get("memory_store"))
    session_state["memory_store"] = append_memory_item(
        store,
        "plan_modification_logs",
        {
            "date": safe_iso_date(date_iso) or date_iso,
            "action_type": action_type,
            "summary": summary,
            "injury_areas": injury_areas or [],
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def record_memory_injury_event(*, session_state: dict[str, Any], date_iso: str, injury_areas: list[str], source: str, summary: str, risk_level: str = "medium") -> None:
    store = normalize_memory_store(session_state.get("memory_store"))
    safe_date = safe_iso_date(date_iso) or date_iso
    areas = injury_areas or ["reported injury area"]
    for area in areas:
        normalized_area = normalize_loose(area) or "reportedinjuryarea"
        store = append_memory_item(
            store,
            "injury_events",
            {
                "id": f"{safe_date}-{normalized_area}",
                "date": safe_date,
                "area": area,
                "risk_level": risk_level,
                "source": source,
                "summary": summary,
                "status": "active",
                "expires_after_days": 7,
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
            },
            unique_key="id",
        )
    session_state["memory_store"] = store


def record_memory_food_avoidance(*, session_state: dict[str, Any], date_iso: str, avoidances: list[str], scope: str) -> None:
    store = normalize_memory_store(session_state.get("memory_store"))
    for food in avoidances:
        store = append_memory_item(
            store,
            "food_preferences",
            {
                "date": safe_iso_date(date_iso) or date_iso,
                "food": food,
                "preference": "avoid",
                "scope": scope,
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
    session_state["memory_store"] = store


def commit_memory(session_state: dict[str, Any], action_message: str) -> None:
    should_save = True
    try:
        memory_decision = call_model_json(
            system_prompt=load_prompt("coach_memory_prompt.txt"),
            user_prompt=json.dumps(
                {
                    "action_message": action_message,
                    "active_date": session_state.get("active_date"),
                    "state_keys_available": PERSISTED_SESSION_KEYS,
                },
                ensure_ascii=True,
                indent=2,
            ),
            temperature=0.0,
            max_tokens=400,
        )
        should_save = bool(memory_decision.get("should_save", True))
    except Exception:
        should_save = True
    session_state["last_action_message"] = action_message
    if not should_save and not action_message:
        session_state["last_action_message"] = ""


def normalize_loose(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = normalize_loose(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(value)
    return deduped


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


def clamp_float(value: Any, minimum: float, maximum: float, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, numeric))


def clamp_int(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, numeric))


def coerce_string_list(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return fallback
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    return cleaned or fallback
