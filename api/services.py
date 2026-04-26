"""Service layer that exposes FitnessAgent behavior without Streamlit UI code."""

from __future__ import annotations

from threading import RLock
from typing import Any
from uuid import uuid4

import app as streamlit_app
from agent.graph import run_agent
from agent.state import FitnessAgentState
from api.runtime import backend_streamlit_context
from api.schemas import DailyFeedbackRequest, GeneratePlanRequest

_LOCK = RLock()


def get_state() -> dict[str, Any]:
    with _LOCK:
        with backend_streamlit_context() as session_state:
            _prepare_session(session_state)
            return _state_snapshot(session_state)


def reset_state() -> dict[str, Any]:
    with _LOCK:
        with backend_streamlit_context() as session_state:
            _prepare_session(session_state)
            streamlit_app._reset_app_state()
            return _state_snapshot(session_state)


def generate_plan(request: GeneratePlanRequest) -> dict[str, Any]:
    with _LOCK:
        with backend_streamlit_context() as session_state:
            _prepare_session(session_state)
            profile_inputs = _profile_inputs_from_request(request)
            session_state["thread_id"] = session_state.get("thread_id") or f"api-{uuid4()}"
            session_state["profile_inputs"] = profile_inputs
            session_state["active_date"] = profile_inputs["start_date"]
            session_state["pending_homepage_date_picker"] = profile_inputs["start_date"]

            initial_state = streamlit_app._build_initial_state(profile_inputs)
            result = run_agent(initial_state)
            session_state["agent_result"] = result
            session_state["daily_history"] = result.get("daily_history", [])
            session_state["completed_training_days"] = []
            session_state["week_history"] = []
            session_state["assistant_chat_messages"] = []
            session_state["last_action_message"] = "Plan generated."
            streamlit_app._save_persisted_session_state()
            return _state_snapshot(session_state)


def chat(message: str) -> tuple[str, dict[str, Any]]:
    with _LOCK:
        with backend_streamlit_context() as session_state:
            _prepare_session(session_state)
            if not message.strip():
                raise ValueError("Chat message cannot be empty.")
            if not session_state.get("agent_result"):
                raise ValueError("Create a plan before using AI Coach chat.")

            reply = streamlit_app._call_chat_assistant(message.strip())
            messages = session_state.setdefault("assistant_chat_messages", [])
            messages.append({"role": "user", "content": message.strip()})
            messages.append({"role": "assistant", "content": reply})
            streamlit_app._save_persisted_session_state()
            return reply, _state_snapshot(session_state)


def make_tomorrow_plan(request: DailyFeedbackRequest) -> dict[str, Any]:
    with _LOCK:
        with backend_streamlit_context() as session_state:
            _prepare_session(session_state)
            profile_inputs = session_state.get("profile_inputs")
            result = session_state.get("agent_result")
            if not profile_inputs or not result:
                raise ValueError("Create a plan before making tomorrow's plan.")

            current_reference_date = streamlit_app._current_interaction_date(result)
            current_session = streamlit_app._session_for_history_date(result, current_reference_date, {})
            completed_training_days = set(session_state.get("completed_training_days", []) or [])
            if current_session.get("scheduled_date") and not current_session.get("is_cancelled"):
                completed_training_days.add(current_session["scheduled_date"])

            target_date = streamlit_app._next_calendar_date(current_reference_date)
            should_rollover_cycle = streamlit_app._target_starts_new_cycle(result.get("current_plan", {}), target_date)
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

            updated_result = streamlit_app._record_daily_feedback_and_advance(
                previous_result=result,
                current_session=current_session,
                feedback_date=current_reference_date,
                target_date=target_date,
                current_weight_kg=float(request.current_weight_kg),
                current_body_fat_pct=float(request.current_body_fat_pct),
                workout_feeling=request.workout_feeling,
                feeling_emoji=request.feeling_emoji,
            )
            if should_rollover_cycle:
                updated_result = streamlit_app._generate_next_cycle_after_feedback(
                    profile_inputs=profile_inputs,
                    previous_result=updated_result,
                    feedback_date=current_reference_date,
                    target_date=target_date,
                    current_weight_kg=float(request.current_weight_kg),
                    current_body_fat_pct=float(request.current_body_fat_pct),
                    workout_feeling=request.workout_feeling,
                    feeling_emoji=request.feeling_emoji,
                )

            session_state["agent_result"] = updated_result
            session_state["daily_history"] = updated_result.get("daily_history", [])
            session_state["last_feedback_summary"] = streamlit_app._daily_feedback_summary(
                workout_feeling=request.workout_feeling,
                feeling_emoji=request.feeling_emoji,
            )
            streamlit_app._save_persisted_session_state()
            return _state_snapshot(session_state)


def _prepare_session(session_state: dict[str, Any]) -> None:
    streamlit_app._initialize_session_state()
    streamlit_app._load_persisted_session_state()
    session_state["memory_store"] = streamlit_app._normalize_memory_store(
        session_state.get("memory_store")
    )


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
        "available_days": streamlit_app._sort_days(request.available_days),
        "equipment_access": streamlit_app._default_equipment_access(),
        "start_date": request.start_date,
        "injuries_text": "",
        "allergies_text": request.allergies_text,
        "dietary_preferences": request.dietary_preferences,
        "profile_notes": request.profile_notes,
    }


def _state_snapshot(session_state: dict[str, Any]) -> dict[str, Any]:
    result: FitnessAgentState | dict[str, Any] = session_state.get("agent_result") or {}
    active_date = session_state.get("active_date") or result.get("current_date", "")
    return streamlit_app._json_safe(
        {
            "active_date": active_date,
            "profile_inputs": session_state.get("profile_inputs"),
            "agent_result": result,
            "memory_store": session_state.get("memory_store", streamlit_app._default_memory_store()),
            "daily_history": session_state.get("daily_history", []),
            "assistant_chat_messages": session_state.get("assistant_chat_messages", []),
            "last_feedback_summary": session_state.get("last_feedback_summary", ""),
            "last_action_message": session_state.get("last_action_message", ""),
        }
    )
