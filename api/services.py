"""Service layer that exposes FitnessAgent behavior without Streamlit UI code."""

from __future__ import annotations

from threading import RLock
from typing import Any
from uuid import uuid4

from agent.graph import run_agent
from agent.services.coach_chat_service import call_chat_assistant
from agent.services.feedback_service import (
    daily_feedback_summary,
    generate_next_cycle_after_feedback,
    record_daily_feedback_and_advance,
)
from agent.services.memory import (
    compact_conversation_memory,
    default_memory_store,
    memory_context_for_planning,
    normalize_memory_store,
)
from agent.services.persistence import (
    PERSISTED_SESSION_KEYS,
    delete_app_state,
    json_safe,
    load_app_state,
    save_app_state,
)
from agent.services.plan_enrichment import hydrate_agent_result_for_display
from agent.services.planning_helpers import (
    current_interaction_date,
    default_equipment_access,
    next_calendar_date,
    session_for_history_date,
    sort_days,
    target_starts_new_cycle,
)
from agent.services.state_builders import build_initial_state
from agent.state import FitnessAgentState
from api.schemas import DailyFeedbackRequest, GeneratePlanRequest

_LOCK = RLock()


def get_state() -> dict[str, Any]:
    with _LOCK:
        session_state: dict[str, Any] = {}
        _prepare_session(session_state)
        return _state_snapshot(session_state)


def reset_state() -> dict[str, Any]:
    with _LOCK:
        delete_app_state()
        session_state = _default_session_state()
        return _state_snapshot(session_state)


def generate_plan(request: GeneratePlanRequest) -> dict[str, Any]:
    with _LOCK:
        session_state: dict[str, Any] = {}
        _prepare_session(session_state)
        profile_inputs = _profile_inputs_from_request(request)
        session_state["thread_id"] = session_state.get("thread_id") or f"api-{uuid4()}"
        session_state["profile_inputs"] = profile_inputs
        session_state["active_date"] = profile_inputs["start_date"]
        session_state["pending_homepage_date_picker"] = profile_inputs["start_date"]

        initial_state = build_initial_state(
            profile_inputs=profile_inputs,
            thread_id=session_state["thread_id"],
            active_date=session_state.get("active_date", profile_inputs["start_date"]),
            memory_store=session_state.get("memory_store", default_memory_store()),
            session_state=session_state,
        )
        result = run_agent(initial_state)
        session_state["agent_result"] = result
        session_state["daily_history"] = result.get("daily_history", [])
        session_state["completed_training_days"] = []
        session_state["week_history"] = []
        session_state["assistant_chat_messages"] = []
        session_state["last_action_message"] = "Plan generated."
        save_app_state(session_state)
        return _state_snapshot(session_state)


def chat(message: str) -> tuple[str, dict[str, Any]]:
    with _LOCK:
        session_state: dict[str, Any] = {}
        _prepare_session(session_state)
        if not message.strip():
            raise ValueError("Chat message cannot be empty.")
        if not session_state.get("agent_result"):
            raise ValueError("Create a plan before using AI Coach chat.")

        user_message = message.strip()
        reply = call_chat_assistant(user_message, session_state)
        messages = session_state.setdefault("assistant_chat_messages", [])
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": reply})
        summary, window = compact_conversation_memory(
            messages,
            session_state.get("conversation_summary", ""),
        )
        session_state["conversation_summary"] = summary
        session_state["assistant_chat_messages"] = window
        _refresh_agent_memory_context(session_state)
        save_app_state(session_state)
        return reply, _state_snapshot(session_state)


def make_tomorrow_plan(request: DailyFeedbackRequest) -> dict[str, Any]:
    with _LOCK:
        session_state: dict[str, Any] = {}
        _prepare_session(session_state)
        profile_inputs = session_state.get("profile_inputs")
        result = session_state.get("agent_result")
        if not profile_inputs or not result:
            raise ValueError("Create a plan before making tomorrow's plan.")

        current_reference_date = current_interaction_date(
            active_date=session_state.get("active_date"),
            result=result,
        )
        current_session = session_for_history_date(result, current_reference_date, {})
        completed_training_days = set(session_state.get("completed_training_days", []) or [])
        if current_session.get("scheduled_date") and not current_session.get("is_cancelled"):
            completed_training_days.add(current_session["scheduled_date"])

        target_date = next_calendar_date(current_reference_date)
        should_rollover_cycle = target_starts_new_cycle(result.get("current_plan", {}), target_date)
        if should_rollover_cycle:
            current_cycle_label = f"{result.get('current_plan', {}).get('cycle_number', 1)}"
            session_state.setdefault("week_history", []).append(
                {
                    "week_start": current_cycle_label,
                    "summary": result.get("current_plan", {}).get("summary", "Completed week"),
                }
            )
            session_state["completed_training_days"] = []
        else:
            session_state["completed_training_days"] = sorted(completed_training_days)

        session_state["active_date"] = target_date
        session_state["pending_homepage_date_picker"] = target_date
        session_state["last_action_message"] = ""

        updated_result, memory_store = record_daily_feedback_and_advance(
            previous_result=result,
            current_session=current_session,
            feedback_date=current_reference_date,
            target_date=target_date,
            current_weight_kg=float(request.current_weight_kg),
            current_body_fat_pct=float(request.current_body_fat_pct),
            workout_feeling=request.workout_feeling,
            feeling_emoji=request.feeling_emoji,
            memory_store=session_state.get("memory_store", default_memory_store()),
        )
        session_state["memory_store"] = memory_store
        if should_rollover_cycle:
            updated_result = generate_next_cycle_after_feedback(
                profile_inputs=profile_inputs,
                previous_result=updated_result,
                feedback_date=current_reference_date,
                target_date=target_date,
                current_weight_kg=float(request.current_weight_kg),
                current_body_fat_pct=float(request.current_body_fat_pct),
                workout_feeling=request.workout_feeling,
                feeling_emoji=request.feeling_emoji,
                thread_id=session_state["thread_id"],
                memory_store=session_state.get("memory_store", default_memory_store()),
            )

        session_state["agent_result"] = updated_result
        session_state["daily_history"] = updated_result.get("daily_history", [])
        _refresh_agent_memory_context(session_state)
        session_state["last_feedback_summary"] = daily_feedback_summary(
            workout_feeling=request.workout_feeling,
            feeling_emoji=request.feeling_emoji,
        )
        save_app_state(session_state)
        return _state_snapshot(session_state)


def _prepare_session(session_state: dict[str, Any]) -> None:
    defaults = _default_session_state()
    session_state.update({key: value for key, value in defaults.items() if key not in session_state})
    payload = load_app_state()
    for key in PERSISTED_SESSION_KEYS:
        if key in payload:
            session_state[key] = payload[key]
    session_state["memory_store"] = normalize_memory_store(session_state.get("memory_store"))
    if session_state.get("agent_result"):
        session_state["agent_result"] = hydrate_agent_result_for_display(session_state["agent_result"])
        _refresh_agent_memory_context(session_state)


def _default_session_state() -> dict[str, Any]:
    return {
        "thread_id": f"api-{uuid4()}",
        "profile_inputs": None,
        "agent_result": None,
        "active_date": "",
        "completed_training_days": [],
        "week_history": [],
        "daily_history": [],
        "memory_store": default_memory_store(),
        "assistant_chat_messages": [],
        "conversation_summary": "",
        "last_feedback_summary": "",
        "last_action_message": "",
    }


def _profile_inputs_from_request(request: GeneratePlanRequest) -> dict[str, Any]:
    return {
        "age": request.age,
        "sex": request.sex,
        "height_cm": float(request.height_cm),
        "weight_kg": float(request.weight_kg),
        "body_fat_pct": float(request.body_fat_pct),
        "fitness_level": request.fitness_level,
        "activity_level": request.activity_level,
        "primary_goal": request.primary_goal,
        "timeline_weeks": int(request.timeline_weeks),
        "target_weight_kg": float(request.target_weight_kg),
        "target_body_fat_pct": float(request.target_body_fat_pct),
        "sessions_per_week": int(request.sessions_per_week),
        "minutes_per_session": int(request.minutes_per_session),
        "available_days": sort_days(request.available_days),
        "equipment_access": default_equipment_access(),
        "start_date": request.start_date,
        "injuries_text": "",
        "allergies_text": request.allergies_text,
        "dietary_preferences": request.dietary_preferences,
        "profile_notes": request.profile_notes,
    }


def _state_snapshot(session_state: dict[str, Any]) -> dict[str, Any]:
    result: FitnessAgentState | dict[str, Any] = session_state.get("agent_result") or {}
    if result:
        result = hydrate_agent_result_for_display(result)
    active_date = session_state.get("active_date") or result.get("current_date", "")
    return json_safe(
        {
            "active_date": active_date,
            "profile_inputs": session_state.get("profile_inputs"),
            "agent_result": result,
            "memory_store": session_state.get("memory_store", default_memory_store()),
            "daily_history": session_state.get("daily_history", []),
            "assistant_chat_messages": session_state.get("assistant_chat_messages", []),
            "conversation_summary": session_state.get("conversation_summary", ""),
            "last_feedback_summary": session_state.get("last_feedback_summary", ""),
            "last_action_message": session_state.get("last_action_message", ""),
        }
    )


def _refresh_agent_memory_context(session_state: dict[str, Any]) -> None:
    result = session_state.get("agent_result")
    if not isinstance(result, dict) or not result:
        return
    target_date = session_state.get("active_date") or result.get("current_date", "")
    result["memory_context"] = memory_context_for_planning(
        session_state.get("memory_store", default_memory_store()),
        target_date,
        profile_inputs=session_state.get("profile_inputs") or {},
        result=result,
        session_state=session_state,
    )
