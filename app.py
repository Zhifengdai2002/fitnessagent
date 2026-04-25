from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

from agent.config import load_settings
from agent.graph import run_agent
from agent.llm import call_model_json, call_model_text, load_prompt
from agent.state import FitnessAgentState
from agent.tools import (
    build_video_resources,
    calculate_food_macros,
    find_foods,
    get_food_by_name,
    search_similar_exercises,
)

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_INDEX = {day: index for index, day in enumerate(WEEKDAY_ORDER)}
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
FEELING_EMOJI_OPTIONS = ["😊", "😐", "😫"]
FEELING_EMOJI_LABELS = {
    "😊": "Good",
    "😐": "Okay",
    "😫": "Hard",
}


def main() -> None:
    settings = load_settings()

    st.set_page_config(page_title="FitnessAgent", layout="wide")
    _initialize_session_state()
    _apply_pending_date_picker()

    _apply_light_theme()
    st.title("FitnessAgent")

    agent_output_container = st.container()
    feedback_container = st.container()

    _render_sidebar(settings)

    result = st.session_state.get("agent_result")
    with agent_output_container:
        if result:
            _render_agent_output(result)
        else:
            st.info("Fill in the User Profile in the sidebar and click `Run FitnessAgent` to create your first plan.")

    with feedback_container:
        _render_daily_feedback_section()
    _render_floating_chat_assistant(settings)


def _initialize_session_state() -> None:
    defaults = {
        "thread_id": f"streamlit-{uuid4()}",
        "profile_inputs": None,
        "agent_result": None,
        "last_feedback_summary": "",
        "active_date": date.today().isoformat(),
        "homepage_date_picker": date.today(),
        "pending_homepage_date_picker": None,
        "completed_training_days": [],
        "week_history": [],
        "daily_history": [],
        "assistant_chat_messages": [],
        "assistant_chat_open": False,
        "last_action_message": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _apply_pending_date_picker() -> None:
    pending_date = st.session_state.get("pending_homepage_date_picker")
    if not pending_date:
        return
    parsed_date = _iso_to_date(str(pending_date))
    st.session_state["homepage_date_picker"] = parsed_date
    st.session_state["active_date"] = parsed_date.isoformat()
    st.session_state["pending_homepage_date_picker"] = None


def _apply_light_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --fa-bg: #f8f5ef;
            --fa-sidebar: #f2ece2;
            --fa-border: #ded5c7;
            --fa-text: #1f2937;
            --fa-muted: #5b6472;
            --fa-accent: #0f766e;
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #fbf8f2 0%, #f5f1e8 100%);
            color: var(--fa-text);
        }
        [data-testid="stHeader"] {
            background: rgba(248, 245, 239, 0.9);
        }
        [data-testid="stSidebar"] {
            background: var(--fa-sidebar);
            border-right: 1px solid var(--fa-border);
        }
        h1, h2, h3, h4, h5, h6, p, label, li {
            color: var(--fa-text);
        }
        [data-testid="stMarkdownContainer"] p {
            color: var(--fa-text);
        }
        [data-testid="stCaptionContainer"] {
            color: var(--fa-muted);
        }
        a {
            color: var(--fa-accent) !important;
        }
        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 12px;
        }
        [data-baseweb="input"],
        [data-baseweb="base-input"],
        [data-baseweb="select"] > div,
        [data-baseweb="textarea"] > div,
        textarea,
        input {
            color: var(--fa-text) !important;
            caret-color: var(--fa-text) !important;
        }
        input::placeholder,
        textarea::placeholder {
            color: #7b8794 !important;
        }
        [data-testid="stInfo"] {
            background: #f0f9f8;
            border: 1px solid #b7dfdb;
        }
        [data-testid="stSuccess"] {
            background: #edfdf5;
            border: 1px solid #bde8d0;
        }
        [data-testid="stWarning"] {
            background: #fffbeb;
            border: 1px solid #f4df9b;
        }
        [data-testid="stAlert"] {
            border-radius: 14px;
        }
        .st-key-floating_chat_assistant {
            position: fixed;
            right: 24px;
            bottom: 24px;
            width: min(390px, calc(100vw - 32px));
            max-height: 72vh;
            overflow-y: auto;
            z-index: 9999;
            background: #fffaf2;
            border: 1px solid #d8d1c6;
            border-radius: 14px;
            box-shadow: 0 18px 48px rgba(31, 41, 55, 0.18);
            padding: 16px;
        }
        .st-key-floating_chat_launcher {
            position: fixed;
            right: 24px;
            bottom: 24px;
            z-index: 9999;
            width: 64px;
            height: 64px;
            background: #fffaf2;
            border: 1px solid #d8d1c6;
            border-radius: 50%;
            box-shadow: 0 18px 48px rgba(31, 41, 55, 0.18);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 8px;
        }
        .st-key-floating_chat_launcher div.stButton > button {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            padding: 0;
            font-size: 1.45rem;
            line-height: 1;
        }
        .st-key-floating_chat_assistant h3 {
            margin-top: 0;
            font-size: 1.05rem;
        }
        .st-key-floating_chat_assistant [data-testid="stMarkdownContainer"] p {
            font-size: 0.92rem;
            line-height: 1.35;
        }
        .st-key-floating_chat_assistant [data-testid="stForm"] {
            border: 0;
            padding: 0;
        }
        @media (max-width: 760px) {
            .st-key-floating_chat_assistant {
                right: 12px;
                bottom: 12px;
                width: calc(100vw - 24px);
                max-height: 58vh;
            }
            .st-key-floating_chat_launcher {
                right: 12px;
                bottom: 12px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(settings) -> None:
    with st.sidebar:
        st.header("User Profile")
        with st.form("profile_form", clear_on_submit=False):
            age = st.number_input("Age", min_value=16, max_value=90, value=26)
            sex = st.selectbox("Sex", ["male", "female", "other", "prefer_not_to_say"], index=0)
            height_cm = st.number_input("Height (cm)", min_value=130.0, max_value=230.0, value=175.0)
            weight_kg = st.number_input("Weight (kg)", min_value=35.0, max_value=250.0, value=78.0)
            body_fat_pct = st.number_input("Body Fat (%)", min_value=3.0, max_value=60.0, value=24.0, step=0.1)
            fitness_level = st.selectbox("Fitness Level", ["beginner", "intermediate", "advanced"], index=0)
            start_date = st.date_input(
                "Start Date",
                value=_iso_to_date(st.session_state.get("active_date", date.today().isoformat())),
            )

            st.subheader("Goals")
            primary_goal = st.selectbox(
                "Primary Goal",
                ["weight_loss", "strength", "sculpting"],
                format_func=lambda value: {
                    "weight_loss": "Weight Loss",
                    "strength": "Strength",
                    "sculpting": "Sculpting",
                }[value],
                index=0,
            )
            timeline_weeks = st.slider("Timeline (weeks)", min_value=4, max_value=24, value=12)
            target_weight_kg = st.number_input("Target Weight (kg)", min_value=35.0, max_value=250.0, value=72.0)
            target_body_fat_pct = st.number_input("Target Body Fat (%)", min_value=3.0, max_value=60.0, value=18.0, step=0.1)

            st.subheader("Constraints")
            sessions_per_week = st.slider("Sessions per Cycle", min_value=3, max_value=5, value=4)
            available_days = st.multiselect(
                "Available Days",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                default=["Monday", "Wednesday", "Saturday"],
            )
            injuries_text = st.text_input("Injuries or Pain-Sensitive Areas", value="")
            allergies_text = st.text_input("Food Allergies", value="")
            dietary_preferences = st.multiselect(
                "Diet Preferences",
                ["vegetarian", "vegan", "gluten_free", "high_protein"],
                default=[],
            )
            profile_notes = st.text_area(
                "Notes",
                placeholder="Anything else I should know? Example: I prefer short morning workouts, I get bored easily, my left knee is sensitive, I want home-based training.",
            )

            submitted = st.form_submit_button("Run FitnessAgent", type="primary")

        with st.expander("System", expanded=False):
            st.write("Project root:", Path(__file__).resolve().parent)
            st.write("Model:", settings.model_name)
            st.write("Model base URL:", settings.model_base_url)
            st.write("Model API key loaded:", settings.has_model_api_key)
            st.write("YouTube key loaded:", settings.has_youtube_key)

    if submitted:
        sorted_available_days = _sort_days(available_days)
        start_date_iso = start_date.isoformat()
        st.session_state["active_date"] = start_date_iso
        st.session_state["homepage_date_picker"] = start_date
        st.session_state["completed_training_days"] = []
        st.session_state["week_history"] = []
        st.session_state["daily_history"] = []
        st.session_state["last_feedback_summary"] = ""
        st.session_state["last_action_message"] = ""

        profile_inputs = {
            "age": age,
            "sex": sex,
            "height_cm": float(height_cm),
            "weight_kg": float(weight_kg),
            "body_fat_pct": float(body_fat_pct),
            "fitness_level": fitness_level,
            "activity_level": "lightly_active",
            "start_date": start_date_iso,
            "primary_goal": primary_goal,
            "timeline_weeks": timeline_weeks,
            "target_weight_kg": float(target_weight_kg),
            "target_body_fat_pct": float(target_body_fat_pct),
            "sessions_per_week": sessions_per_week,
            "minutes_per_session": 60,
            "available_days": sorted_available_days,
            "equipment_access": _default_equipment_access(),
            "injuries_text": injuries_text,
            "allergies_text": allergies_text,
            "dietary_preferences": dietary_preferences,
            "profile_notes": profile_notes,
        }
        st.session_state["profile_inputs"] = profile_inputs
        _execute_agent(_build_initial_state(profile_inputs))


def _render_floating_chat_assistant(settings) -> None:
    if not st.session_state.get("assistant_chat_open", False):
        with st.container(key="floating_chat_launcher", border=False):
            if st.button("🤖", key="assistant_chat_open_button", help="Open AI Coach"):
                st.session_state["assistant_chat_open"] = True
                st.rerun()
        return

    with st.container(key="floating_chat_assistant", border=False):
        title_col, close_col = st.columns([3, 1])
        title_col.markdown("### AI Coach")
        if close_col.button("Minimize", key="assistant_chat_minimize"):
            st.session_state["assistant_chat_open"] = False
            st.rerun()

        if not settings.has_model_api_key:
            st.warning("Set MODEL_API_KEY or ZAI_API_KEY in .env to enable chat.")
            return

        messages = st.session_state.setdefault("assistant_chat_messages", [])
        if not messages:
            st.write("Ask about today's plan, meals, recovery, or how to adjust safely.")
        else:
            for message in messages[-6:]:
                role = "You" if message.get("role") == "user" else "Coach"
                st.markdown(f"**{role}:** {message.get('content', '')}")

        with st.form("floating_chat_form", clear_on_submit=True):
            user_message = st.text_area(
                "Message",
                placeholder="Ask your fitness assistant...",
                key="assistant_chat_input",
                height=80,
                label_visibility="collapsed",
            )
            send_col, clear_col = st.columns([1, 1])
            send_message = send_col.form_submit_button("Send")
            clear_chat = clear_col.form_submit_button("Clear")

        if clear_chat:
            st.session_state["assistant_chat_messages"] = []
            st.rerun()

        if send_message and user_message.strip():
            assistant_reply = _call_chat_assistant(user_message.strip())
            messages = st.session_state.setdefault("assistant_chat_messages", [])
            messages.append({"role": "user", "content": user_message.strip()})
            messages.append({"role": "assistant", "content": assistant_reply})
            st.rerun()


def _call_chat_assistant(user_message: str) -> str:
    update_summary = _maybe_update_today_from_chat(user_message)
    result = st.session_state.get("agent_result", {})
    context = _build_chat_context(result)
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.get("assistant_chat_messages", [])[-8:]
        if message.get("role") in {"user", "assistant"} and message.get("content")
    ]
    system_prompt = (
        "You are FitnessAgent's floating chat coach. Use the supplied app context, "
        "base every workout answer on FitnessAgent's hard planning rules: baseline beginner plans "
        "use 2 exercises, intermediate plans use 3 exercises, advanced plans use 4 exercises; every "
        "normal exercise is exactly 4 sets. For higher intensity, add one exercise, use higher reps "
        "(beginner 8-10, intermediate/advanced 12-15), and add challenge notes. For lower intensity, "
        "never go below 2 exercises, keep beginner at 2, reduce intermediate to 2, reduce advanced "
        "to 3, use lower reps (beginner 6-8, intermediate/advanced 10-12), and add conservative notes. "
        "Baseline beginner reps are 6-10; baseline intermediate and advanced reps are 10-15. "
        "Daily weight and body-fat check-ins are record-only. "
        "answer concisely, and stay within general fitness coaching. Do not diagnose "
        "medical issues. If the user reports injury, sharp pain, chest pain, dizziness, "
        "or other red flags, advise stopping training and consulting a qualified professional. "
        "If the app context says today's plan was updated, say that it has been updated and "
        "summarize the current Today's Plan instead of only giving generic advice. "
        "You can also replace a named exercise or an ordinal exercise such as 'the first exercise' "
        "with a same-type alternative while preserving the day's focus, sets, and reps. "
        "When updating today's plan, follow the cycle-conflict policy: temporary add plus "
        "same-focus duplicate later in the same cycle turns that later duplicate into rest; "
        "replacement plus same-focus duplicate swaps the two days' contents; cancellation only "
        "affects today; temporary add without duplicate does not change other days. "
        "Current app context:\n"
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


def _maybe_update_today_from_chat(user_message: str) -> str:
    profile_inputs = st.session_state.get("profile_inputs")
    previous_result = st.session_state.get("agent_result")
    if not profile_inputs or not previous_result:
        return ""

    replacement_summary = _maybe_replace_today_exercise_from_chat(user_message, previous_result)
    if replacement_summary:
        st.session_state["last_action_message"] = "Today's plan was updated by AI Coach."
        return replacement_summary

    normalized = _normalize_change_request(user_message)
    normalized = _augment_chat_safety_change(user_message, normalized)
    normalized = _augment_chat_intensity_change(user_message, normalized)
    if not _chat_request_should_update_today(user_message, normalized):
        return ""

    cancel_summary = _maybe_cancel_today_from_chat(normalized, previous_result)
    if cancel_summary:
        st.session_state["last_action_message"] = "Today's workout was cancelled by AI Coach."
        return cancel_summary

    intensity_summary = _maybe_patch_today_intensity_from_chat(normalized, previous_result)
    if intensity_summary:
        st.session_state["last_action_message"] = "Today's plan was updated by AI Coach."
        return intensity_summary

    nutrition_summary = _maybe_update_nutrition_from_chat(normalized, previous_result)
    if nutrition_summary:
        st.session_state["last_action_message"] = "Today's nutrition was updated by AI Coach."
        return nutrition_summary

    updated_state = _build_change_request_state(
        profile_inputs=profile_inputs,
        previous_result=previous_result,
        change_request=user_message,
        normalized_change_request=normalized,
    )
    _execute_ai_plan_patch(updated_state, previous_result, normalized)
    updated_result = st.session_state.get("agent_result", previous_result)
    today_session = _select_today_session(
        _sort_workout_sessions(updated_result.get("current_plan", {}).get("workout_sessions", [])),
        _current_interaction_date(updated_result),
    )
    st.session_state["last_action_message"] = "Today's plan was updated by AI Coach."
    return _summarize_session_for_chat(today_session)


def _maybe_cancel_today_from_chat(
    normalized: dict[str, Any],
    previous_result: FitnessAgentState | dict[str, Any],
) -> str:
    if not normalized.get("cancel_today"):
        return ""

    injury_reported = bool(normalized.get("injury_reported"))
    injury_areas = _coerce_string_list(normalized.get("injury_areas"), [])
    current_date = _current_interaction_date(previous_result)
    updated_result = deepcopy(previous_result)
    sessions = updated_result.get("current_plan", {}).get("workout_sessions", [])
    today_session = _select_today_session(_sort_workout_sessions(sessions), current_date)
    if not today_session:
        parsed_date = _iso_to_date(current_date)
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
                "safety_notes": _cancel_safety_notes(injury_reported=injury_reported, injury_areas=injury_areas),
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
        today_session["safety_notes"] = _cancel_safety_notes(injury_reported=injury_reported, injury_areas=injury_areas)
        today_session["injury_reported"] = injury_reported
        today_session["injury_areas"] = injury_areas

    _refresh_youtube_resources(updated_result)
    st.session_state["agent_result"] = updated_result
    if injury_reported:
        area_text = f" around {', '.join(injury_areas)}" if injury_areas else ""
        return f"Today's workout has been cancelled because you reported an injury{area_text}."
    return "Today's workout has been cancelled as requested. No injury reason was recorded."


def _cancel_safety_notes(*, injury_reported: bool, injury_areas: list[str]) -> list[str]:
    if injury_reported:
        area_text = f" around {', '.join(injury_areas)}" if injury_areas else ""
        return [
            f"Today's workout was cancelled because an injury was reported{area_text}.",
            "Do not train through injury symptoms; consider medical guidance before resuming.",
        ]
    return ["Today's workout was cancelled at your request."]


def _augment_chat_safety_change(user_message: str, normalized: dict[str, Any]) -> dict[str, Any]:
    if normalized.get("injury_reported") or normalized.get("cancel_today"):
        return normalized
    injury_areas = _injury_areas_from_chat_text(user_message)
    cancel_requested = _chat_text_requests_cancel(user_message)
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


def _chat_text_requests_cancel(text: str) -> bool:
    normalized = text.lower()
    cancel_terms = [
        "cancel today's plan",
        "cancel today",
        "cancel workout",
        "skip today",
        "no workout today",
        "取消今天",
        "取消训练",
        "今天不练",
    ]
    return any(term in normalized for term in cancel_terms)


def _injury_areas_from_chat_text(text: str) -> list[str]:
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
    areas = [
        area
        for area, aliases in area_map.items()
        if any(alias in normalized for alias in aliases)
    ]
    return areas or ["reported injury area"]


def _maybe_patch_today_intensity_from_chat(
    normalized: dict[str, Any],
    previous_result: FitnessAgentState | dict[str, Any],
) -> str:
    intensity = str(normalized.get("intensity_adjustment", "")).strip().lower()
    if intensity not in {"higher", "lower"}:
        return ""
    if normalized.get("focus_category") or normalized.get("cancel_today") or normalized.get("injury_reported"):
        return ""

    current_date = _current_interaction_date(previous_result)
    updated_result = deepcopy(previous_result)
    today_session = _select_today_session(
        _sort_workout_sessions(updated_result.get("current_plan", {}).get("workout_sessions", [])),
        current_date,
    )
    if not today_session or today_session.get("is_cancelled") or not today_session.get("exercises"):
        st.session_state["agent_result"] = updated_result
        return "Today has no active workout to adjust, so the current plan was left unchanged."

    if intensity == "lower":
        today_session.pop("is_cancelled", None)
        today_session["duration_minutes"] = int(today_session.get("duration_minutes") or 60)

    if not today_session.get("exercises"):
        return ""

    fitness_level = str(previous_result.get("user_profile", {}).get("fitness_level", "intermediate")).strip().lower()
    updated_exercises = _patch_exercises_for_intensity(
        exercises=list(today_session.get("exercises", [])),
        fitness_level=fitness_level,
        intensity=intensity,
        focus=_focus_tag_from_session(today_session),
    )
    if not updated_exercises:
        return ""

    today_session["exercises"] = updated_exercises
    _refresh_youtube_resources(updated_result)
    st.session_state["agent_result"] = updated_result

    label = "lower" if intensity == "lower" else "higher"
    exercise_text = ", ".join(
        f"{exercise.get('name', '')} {exercise.get('sets', '')}x{exercise.get('reps', '')}".strip()
        for exercise in updated_exercises
    )
    return f"Today's workout was adjusted to {label} intensity: {exercise_text}."


def _patch_exercises_for_intensity(
    *,
    exercises: list[dict[str, Any]],
    fitness_level: str,
    intensity: str,
    focus: str,
) -> list[dict[str, Any]]:
    patched = [dict(exercise) for exercise in exercises]
    if intensity == "lower":
        target_count = 2 if fitness_level in {"beginner", "intermediate"} else 3
        reps = "6-8" if fitness_level == "beginner" else "10-12"
        note = "Lower intensity: keep the load conservative, move with control, and stop well before form breaks."
    else:
        target_count = len(patched) + 1
        reps = "8-10" if fitness_level == "beginner" else "12-15"
        note = "Higher intensity: use controlled tempo, full range of motion, and a brief pause while keeping clean form."

    if intensity == "lower":
        patched = patched[: max(2, min(target_count, len(patched)))]
    elif patched:
        current_names = [
            str(exercise.get("name", "")).strip()
            for exercise in patched
            if str(exercise.get("name", "")).strip()
        ]
        seed_name = current_names[0] if current_names else ""
        additions = search_similar_exercises(
            exercise_name=seed_name,
            focus=focus,
            level=fitness_level,
            exclude=current_names,
            limit=3,
        ) if seed_name else []
        if additions:
            patched.append(_replacement_exercise_payload({"sets": 4, "reps": reps}, additions[0]))
    for exercise in patched:
        exercise["sets"] = 4
        exercise["reps"] = reps
        existing_note = str(exercise.get("notes", "")).strip()
        if note not in existing_note:
            exercise["notes"] = f"{existing_note} {note}".strip()
    return patched


def _execute_ai_plan_patch(
    state: FitnessAgentState,
    previous_result: FitnessAgentState | dict[str, Any],
    normalized: dict[str, Any],
) -> None:
    try:
        result = run_agent(state)
    except Exception as exc:
        st.error(str(exc))
        return

    patched_result = _merge_ai_patch_result(previous_result, result, normalized)
    st.session_state["agent_result"] = patched_result


def _merge_ai_patch_result(
    previous_result: FitnessAgentState | dict[str, Any],
    result: FitnessAgentState | dict[str, Any],
    normalized: dict[str, Any],
) -> FitnessAgentState:
    patched_result: FitnessAgentState = dict(result)
    previous_plan = deepcopy(previous_result.get("current_plan", {}))
    result_plan = deepcopy(result.get("current_plan", {}))

    request_type = str(normalized.get("request_type", "")).strip()
    workout_changed = _normalized_request_has_workout_change(normalized)
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

    if previous_plan:
        for key in ["summary", "objective_alignment", "coaching_focus", "recovery_actions"]:
            if key in previous_plan:
                result_plan[key] = deepcopy(previous_plan[key])

    patched_result["current_plan"] = result_plan
    patched_result["plan_history"] = deepcopy(previous_result.get("plan_history", []))
    patched_result["daily_history"] = deepcopy(previous_result.get("daily_history", []))
    patched_result["feedback_history"] = deepcopy(previous_result.get("feedback_history", []))
    patched_result["state_history"] = deepcopy(previous_result.get("state_history", []))
    _refresh_youtube_resources(patched_result)
    return patched_result


def _maybe_update_nutrition_from_chat(
    normalized: dict[str, Any],
    previous_result: FitnessAgentState | dict[str, Any],
) -> str:
    avoidances = _coerce_string_list(normalized.get("temporary_food_avoidances"), [])
    if not avoidances:
        return ""
    if _normalized_request_has_workout_change(normalized):
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
        if not _food_matches_any_avoidance(food_name, avoidances):
            updated_meals.append(meal)
            continue

        replacement = _replacement_food_for_meal(
            meal=meal,
            avoidances=avoidances,
            used_foods=used_foods,
        )
        if replacement:
            replacement_name = str(replacement.get("food_name", "")).strip()
            replacements.append((food_name, replacement_name))
            used_foods.add(replacement_name)
            updated_meals.append(replacement)

    if not replacements:
        return ""

    current_plan["meal_suggestions"] = updated_meals
    st.session_state["agent_result"] = updated_result
    replacement_text = ", ".join(f"{old} -> {new}" for old, new in replacements)
    return f"Today's nutrition was updated: {replacement_text}. Your workout plan was left unchanged."


def _normalized_request_has_workout_change(normalized: dict[str, Any]) -> bool:
    return bool(
        normalized.get("focus_category")
        or normalized.get("cancel_today")
        or normalized.get("injury_reported")
        or normalized.get("intensity_adjustment")
        or str(normalized.get("request_type", "")) in {"workout_change", "mixed_change", "recovery_change"}
    )


def _food_matches_any_avoidance(food_name: str, avoidances: list[str]) -> bool:
    normalized_name = _normalize_loose(food_name)
    return any(_normalize_loose(avoidance) and _normalize_loose(avoidance) in normalized_name for avoidance in avoidances)


def _replacement_food_for_meal(
    *,
    meal: dict[str, Any],
    avoidances: list[str],
    used_foods: set[str],
) -> dict[str, Any] | None:
    current_food = get_food_by_name(str(meal.get("food_name", "")))
    if not current_food:
        return None
    category = str(current_food.get("category", ""))
    candidates = find_foods(category=category, limit=8)
    for candidate in candidates:
        candidate_name = str(candidate.get("name", "")).strip()
        if not candidate_name or candidate_name in used_foods:
            continue
        if _food_matches_any_avoidance(candidate_name, avoidances):
            continue
        grams = _extract_grams(str(meal.get("serving_size", "100g")), default=100.0)
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


def _extract_grams(serving_size: str, default: float) -> float:
    digits = "".join(char for char in serving_size if char.isdigit() or char == ".")
    return float(digits) if digits else default


def _maybe_replace_today_exercise_from_chat(user_message: str, previous_result: FitnessAgentState | dict[str, Any]) -> str:
    text = user_message.lower()
    replacement_terms = [
        "replace",
        "swap",
        "change",
        "alternative",
        "instead",
        "换",
        "替换",
        "不要",
        "不想做",
        "换掉",
        "改掉",
    ]
    if not any(term in text for term in replacement_terms):
        return ""

    current_date = _current_interaction_date(previous_result)
    current_plan = previous_result.get("current_plan", {})
    today_session = _select_today_session(
        _sort_workout_sessions(current_plan.get("workout_sessions", [])),
        current_date,
    )
    if not today_session or today_session.get("is_cancelled"):
        return ""

    exercises = today_session.get("exercises", [])
    target_index = _exercise_index_from_chat_message(user_message, exercises)
    if target_index is None and not _chat_requests_general_exercise_replacement(user_message):
        return ""

    updated_result = deepcopy(previous_result)
    updated_sessions = updated_result.get("current_plan", {}).get("workout_sessions", [])
    updated_today = _select_today_session(_sort_workout_sessions(updated_sessions), current_date)
    if not updated_today:
        return ""

    updated_exercises = list(updated_today.get("exercises", []))
    current_names = [
        str(exercise.get("name", "")).strip()
        for exercise in updated_exercises
        if str(exercise.get("name", "")).strip()
    ]
    focus_tag = _focus_tag_from_session(today_session)
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
            updated_exercises[index] = _replacement_exercise_payload(original_exercise, replacement)
            replacement_name = str(replacement.get("name", "")).strip()
            excluded_names.append(replacement_name)
            replacements_made.append((original_name, replacement_name))

        if not replacements_made:
            return ""

        updated_today["exercises"] = updated_exercises
        _refresh_youtube_resources(updated_result)
        st.session_state["agent_result"] = updated_result
        replacement_text = ", ".join(f"{old} -> {new}" for old, new in replacements_made)
        return (
            "Today's plan is now updated with same-focus exercise alternatives: "
            f"{replacement_text}. Sets and reps stayed the same."
        )

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
    updated_exercises[target_index] = _replacement_exercise_payload(original_exercise, replacement)
    updated_today["exercises"] = updated_exercises
    _refresh_youtube_resources(updated_result)
    st.session_state["agent_result"] = updated_result

    return (
        f"Today's plan is now updated: replaced {original_name} with "
        f"{replacement.get('name', '')} while keeping the same focus, sets, and reps."
    )


def _chat_requests_general_exercise_replacement(user_message: str) -> bool:
    text = user_message.lower()
    exercise_terms = [
        "exercise",
        "exercises",
        "action",
        "actions",
        "movement",
        "movements",
        "动作",
    ]
    general_terms = [
        "some",
        "all",
        "another",
        "alternative",
        "alternatives",
        "different",
        "new",
        "几个",
        "一些",
        "全部",
        "换一换",
        "换几个",
        "换掉",
    ]
    return any(term in text for term in exercise_terms) and any(term in text for term in general_terms)


def _replacement_exercise_payload(original_exercise: dict[str, Any], replacement: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": replacement.get("name", ""),
        "target_muscle": ", ".join(replacement.get("target_muscle", [])),
        "sets": original_exercise.get("sets", 4),
        "reps": original_exercise.get("reps", ""),
        "equipment": ", ".join(replacement.get("equipment", [])),
        "notes": replacement.get("notes", ""),
    }


def _exercise_index_from_chat_message(user_message: str, exercises: list[dict[str, Any]]) -> int | None:
    text = user_message.lower()
    normalized_text = _normalize_loose(text)
    for index, exercise in enumerate(exercises):
        name = str(exercise.get("name", "")).strip()
        if name and _normalize_loose(name) in normalized_text:
            return index

    ordinal_terms = [
        (0, ["first exercise", "first movement", "1st exercise", "第一个动作", "第1个动作", "第一个", "第1个"]),
        (1, ["second exercise", "second movement", "2nd exercise", "第二个动作", "第2个动作", "第二个", "第2个"]),
        (2, ["third exercise", "third movement", "3rd exercise", "第三个动作", "第3个动作", "第三个", "第3个"]),
        (3, ["fourth exercise", "fourth movement", "4th exercise", "第四个动作", "第4个动作", "第四个", "第4个"]),
        (4, ["fifth exercise", "fifth movement", "5th exercise", "第五个动作", "第5个动作", "第五个", "第5个"]),
    ]
    for index, terms in ordinal_terms:
        if index < len(exercises) and any(term in text for term in terms):
            return index
    return None


def _focus_tag_from_session(session: dict[str, Any]) -> str:
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


def _refresh_youtube_resources(result: FitnessAgentState | dict[str, Any]) -> None:
    exercise_names = [
        str(exercise.get("name", "")).strip()
        for session in result.get("current_plan", {}).get("workout_sessions", [])
        for exercise in session.get("exercises", [])
        if str(exercise.get("name", "")).strip()
    ]
    result["youtube_resources"] = build_video_resources(exercise_names)


def _normalize_loose(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _chat_request_should_update_today(user_message: str, normalized: dict[str, Any]) -> bool:
    text = user_message.lower()
    update_terms = [
        "update",
        "change",
        "switch",
        "add",
        "make today",
        "today",
        "can i",
        "could i",
        "i want",
        "want",
        "do more",
        "energized",
        "excited",
        "strong",
        "easier",
        "lighter",
        "harder",
        "injury",
        "injured",
        "hurt",
        "pain",
        "sleep",
        "slept",
        "didn't sleep",
        "dont sleep",
        "don't sleep",
        "poor sleep",
        "let's",
        "练",
        "改",
        "换",
        "加",
        "今天",
    ]
    has_update_language = any(term in text for term in update_terms)
    has_structured_change = bool(
        normalized.get("focus_category")
        or normalized.get("cancel_today")
        or normalized.get("intensity_adjustment")
        or normalized.get("temporary_food_avoidances")
        or normalized.get("permanent_food_preferences")
    )
    request_type = str(normalized.get("request_type", ""))
    if normalized.get("injury_reported") or normalized.get("cancel_today"):
        return has_structured_change
    return has_update_language and has_structured_change and request_type in {
        "workout_change",
        "nutrition_change",
        "mixed_change",
        "recovery_change",
    }


def _augment_chat_intensity_change(user_message: str, normalized: dict[str, Any]) -> dict[str, Any]:
    if normalized.get("intensity_adjustment") or normalized.get("injury_reported") or normalized.get("cancel_today"):
        return normalized

    fallback_intensity = _chat_intensity_from_text(user_message)
    if not fallback_intensity:
        return normalized

    augmented = dict(normalized)
    augmented["request_type"] = augmented.get("request_type") if augmented.get("request_type") in ALLOWED_CHANGE_REQUEST_TYPES else "workout_change"
    if augmented["request_type"] in {"none", "unclear", ""}:
        augmented["request_type"] = "workout_change"
    augmented["scope"] = augmented.get("scope") if augmented.get("scope") in ALLOWED_CHANGE_REQUEST_SCOPES else "today_only"
    if augmented["scope"] in {"unclear", ""}:
        augmented["scope"] = "today_only"
    augmented["intensity_adjustment"] = fallback_intensity
    summary = str(augmented.get("summary", "")).strip()
    intensity_text = "higher intensity" if fallback_intensity == "higher" else "lower intensity"
    augmented["summary"] = " ".join(part for part in [summary, f"User asked for {intensity_text} today."] if part)
    return augmented


def _chat_intensity_from_text(text: str) -> str:
    normalized = text.lower()
    lower_terms = [
        "tired",
        "uncomfortable",
        "not feeling good",
        "not good",
        "low energy",
        "poor sleep",
        "bad sleep",
        "didn't sleep",
        "dont sleep",
        "don't sleep",
        "slept badly",
        "sleep badly",
        "exhausted",
        "fatigued",
        "make it easier",
        "easier",
        "lighter",
        "less",
        "reduce",
        "不舒服",
        "累",
        "状态不好",
        "轻一点",
        "减量",
    ]
    higher_terms = [
        "energized",
        "excited",
        "feel good",
        "feeling good",
        "strong",
        "do more",
        "more",
        "harder",
        "challenge",
        "too easy",
        "increase",
        "加量",
        "加强",
        "状态很好",
        "很兴奋",
        "多一点",
    ]
    if any(term in normalized for term in lower_terms):
        return "lower"
    if any(term in normalized for term in higher_terms):
        return "higher"
    return ""


def _summarize_session_for_chat(session: dict[str, Any]) -> str:
    if not session:
        return "Today is not a scheduled training day."
    if session.get("is_cancelled"):
        return "Today's workout is cancelled by the app safety rules."
    exercises = [
        f"{exercise.get('name', '')} {exercise.get('sets', '')}x{exercise.get('reps', '')}".strip()
        for exercise in session.get("exercises", [])
    ]
    exercise_text = ", ".join(exercises) if exercises else "no exercises"
    return f"Today's plan is now {session.get('focus', 'training')}: {exercise_text}."


def _build_chat_context(result: FitnessAgentState | dict[str, Any]) -> str:
    if not result:
        return "No plan has been generated yet."
    current_date = _current_interaction_date(result)
    current_plan = result.get("current_plan", {})
    today_session = _select_today_session(
        _sort_workout_sessions(current_plan.get("workout_sessions", [])),
        current_date,
    )
    exercises = [
        f"{exercise.get('name', '')} {exercise.get('sets', '')}x{exercise.get('reps', '')}".strip()
        for exercise in today_session.get("exercises", [])
    ] if today_session else []
    latest_feedback = result.get("latest_feedback", {})
    daily_history = result.get("daily_history", [])[-3:]
    return json.dumps(
        {
            "active_date": current_date,
            "goal": result.get("goals", {}).get("primary_goal", ""),
            "fitness_level": result.get("user_profile", {}).get("fitness_level", ""),
            "planning_rules": {
                "baseline_beginner_exercises": 2,
                "baseline_intermediate_exercises": 3,
                "baseline_advanced_exercises": 4,
                "higher_exercise_delta": 1,
                "lower_beginner_exercise_delta": 0,
                "lower_intermediate_advanced_exercise_delta": -1,
                "minimum_exercises": 2,
                "sets_per_exercise": 4,
                "baseline_beginner_reps": "6-10",
                "higher_beginner_reps": "8-10",
                "lower_beginner_reps": "6-8",
                "baseline_intermediate_advanced_reps": "10-15",
                "higher_intermediate_advanced_reps": "12-15",
                "lower_intermediate_advanced_reps": "10-12",
                "weight_body_fat": "record-only daily check-in values",
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
            "recent_daily_history": daily_history,
        },
        ensure_ascii=True,
    )


def _render_agent_output(result: FitnessAgentState) -> None:
    current_plan = result.get("current_plan", {})
    evaluation_result = result.get("evaluation_result", {})
    current_reference_date = _current_interaction_date(result)

    selected_date = st.date_input("Plan Date", key="homepage_date_picker")
    if selected_date.isoformat() != current_reference_date:
        current_reference_date = selected_date.isoformat()
        st.session_state["active_date"] = current_reference_date
    st.caption(f"Viewing plan for {selected_date.isoformat()} ({selected_date.strftime('%A')})")

    today_container = st.container()
    weekly_container = st.container()
    notes_container = st.container()
    history_container = st.container()

    with today_container:
        st.caption(result.get("coaching_message", ""))
        if result.get("needs_revision"):
            st.warning("The agent thinks this plan still needs some adjustment based on the latest context.")
        action_message = st.session_state.get("last_action_message", "")
        if action_message:
            st.success(action_message)

        st.subheader("Today's Plan")
        st.caption("Feel free to ask AI Coach to adjust plan 😊")
        today_session = _select_today_session(
            _sort_workout_sessions(current_plan.get("workout_sessions", [])),
            current_reference_date,
        )
        session_videos = _videos_for_session(today_session, result.get("youtube_resources", []))

        plan_col, support_col = st.columns([1.25, 1])
        with plan_col:
            if today_session:
                if today_session.get("is_cancelled"):
                    st.markdown(f"**{_session_display_label(today_session)}**")
                    st.warning("Today's workout is cancelled by a hard safety rule.")
                    if today_session.get("safety_notes"):
                        st.write("Safety Notes:", " ".join(today_session.get("safety_notes", [])))
                else:
                    st.markdown(f"**{_session_display_label(today_session)}**")
                    st.write(f"Training time: {int(today_session.get('duration_minutes', 60))} minutes")
                    st.write("Warm-up:", ", ".join(today_session.get("warmup", [])))
                    for exercise in today_session.get("exercises", []):
                        st.markdown(
                            f"- **{exercise.get('name', '')}**: "
                            f"{exercise.get('sets', '')} x {exercise.get('reps', '')}"
                        )
                    st.write("Cooldown:", ", ".join(today_session.get("cooldown", [])))
                    if today_session.get("safety_notes"):
                        st.write("Safety Notes:", " ".join(today_session.get("safety_notes", [])))
            else:
                st.info("Today is not a scheduled training day. Focus on recovery, walking, or easy mobility.")

        with support_col:
            st.markdown("**Today's Nutrition**")
            _render_nutrition_targets(current_plan.get("nutrition_targets", {}))
            for meal in current_plan.get("meal_suggestions", []):
                st.markdown(
                    f"- **{meal.get('meal_slot', '').title()}**: {meal.get('food_name', '')} "
                    f"({meal.get('serving_size', '')})"
                )
            st.markdown("**Video Resources**")
            for resource in session_videos:
                st.markdown(f"- [{resource.get('exercise_name', '')}]({resource.get('url', '')})")

    with weekly_container:
        st.subheader("Training Cycle")
        cycle_number = current_plan.get("cycle_number", 1)
        cycle_start_date = current_plan.get("cycle_start_date", "")
        cycle_end_date = current_plan.get("cycle_end_date", "")
        st.caption(f"第 {cycle_number} 周期 · {cycle_start_date} to {cycle_end_date}")
        st.write(current_plan.get("summary", ""))
        cycle_sessions = [
            session
            for session in _sort_workout_sessions(current_plan.get("workout_sessions", []))
            if not session.get("is_ad_hoc")
        ]
        for session in cycle_sessions:
            session_label = _session_display_label(session)
            with st.expander(session_label, expanded=False):
                st.write(f"Training time: {int(session.get('duration_minutes', 60))} minutes")
                for exercise in session.get("exercises", []):
                    st.markdown(
                        f"- **{exercise.get('name', '')}**: "
                        f"{exercise.get('sets', '')} x {exercise.get('reps', '')}"
                    )

    with notes_container:
        st.subheader("Coach Notes")
        if _has_real_feedback(result) and evaluation_result:
            st.markdown("**Evaluation Summary**")
            st.write(evaluation_result.get("summary", ""))
            reasons = evaluation_result.get("reasons", [])
            if reasons:
                st.markdown("**Revision Reasons**")
                st.markdown("\n".join(f"- {reason}" for reason in reasons))
        else:
            st.write("No evaluation yet. Daily logs are saved in History while the cycle plan stays fixed.")

    with history_container:
        st.subheader("History")
        daily_history = result.get("daily_history") or st.session_state.get("daily_history", [])
        daily_history = _repair_daily_history_against_plan(daily_history, current_plan)
        if daily_history != (result.get("daily_history") or []):
            result["daily_history"] = daily_history
            if st.session_state.get("agent_result"):
                st.session_state["agent_result"]["daily_history"] = daily_history
            st.session_state["daily_history"] = daily_history
        if not daily_history:
            st.write("No daily history yet.")
        for cycle_number, cycle_items in _group_daily_history_by_cycle(daily_history):
            with st.expander(f"Cycle {cycle_number}", expanded=False):
                for item in reversed(cycle_items):
                    _render_daily_history_item(item)


def _render_daily_history_item(item: dict[str, Any]) -> None:
    actions = item.get("completed_actions", [])
    completed_plan = item.get("completed_plan", {})
    action_text = _history_action_text(completed_plan, actions)
    feedback = item.get("feedback", {})
    metric_text = (
        f"{item.get('weight_kg', '-')} kg, "
        f"{item.get('body_fat_pct', '-')}% body fat"
    )
    st.markdown(
        f"- **{item.get('date', '')}** {feedback.get('emoji', '')} "
        f"{metric_text} · {action_text}"
    )
    injury_note = _history_injury_note(item)
    if injury_note:
        st.caption(injury_note)
    feeling = str(feedback.get("workout_feeling", "")).strip()
    if feeling:
        st.caption(feeling)


def _repair_daily_history_against_plan(daily_history: list[dict[str, Any]], current_plan: dict[str, Any]) -> list[dict[str, Any]]:
    sessions = _sort_workout_sessions(current_plan.get("workout_sessions", []))
    if not daily_history or not sessions:
        return list(daily_history or [])

    repaired: list[dict[str, Any]] = []
    changed = False
    for item in daily_history:
        item_copy = deepcopy(item)
        item_date = _safe_iso_date(item_copy.get("date"))
        completed_plan = item_copy.get("completed_plan", {})
        completed_date = _safe_iso_date(completed_plan.get("scheduled_date")) if completed_plan else ""
        plan_session = _select_today_session(sessions, item_date)
        if _history_item_needs_plan_repair(item_copy, plan_session, completed_date, item_date):
            item_copy["completed_plan"] = deepcopy(plan_session)
            item_copy["completed_actions"] = _completed_actions_from_session(plan_session)
            item_copy["cycle_number"] = _history_item_cycle_number({"completed_plan": plan_session})
            feedback = dict(item_copy.get("feedback", {}))
            feedback["injury_areas"] = _history_injury_areas(plan_session, {"pain_points": []})
            item_copy["feedback"] = feedback
            changed = True
        repaired.append(item_copy)
    return repaired if changed else list(daily_history)


def _history_item_needs_plan_repair(
    item: dict[str, Any],
    plan_session: dict[str, Any],
    completed_date: str,
    item_date: str,
) -> bool:
    if not plan_session:
        return False
    completed_plan = item.get("completed_plan", {})
    if not completed_plan:
        return True
    if completed_date != item_date:
        return True
    plan_is_cancelled = bool(plan_session.get("is_cancelled"))
    completed_is_cancelled = bool(completed_plan.get("is_cancelled"))
    if plan_is_cancelled != completed_is_cancelled:
        return True
    plan_actions = _completed_actions_from_session(plan_session)
    completed_actions = [
        str(action).strip()
        for action in item.get("completed_actions", [])
        if str(action).strip()
    ]
    if plan_actions and not completed_actions:
        return True
    if completed_actions and completed_actions != plan_actions:
        return True
    if plan_actions and completed_plan.get("exercises") != plan_session.get("exercises"):
        return True
    return False


def _history_action_text(completed_plan: dict[str, Any], actions: list[str]) -> str:
    if completed_plan.get("is_cancelled"):
        return "Workout cancelled"
    if actions:
        return ", ".join(actions)
    if completed_plan and completed_plan.get("scheduled_date"):
        return "Workout recorded with no exercises"
    return "No scheduled workout"


def _history_injury_note(item: dict[str, Any]) -> str:
    feedback = item.get("feedback", {})
    injury_areas = _coerce_string_list(feedback.get("injury_areas"), [])
    if not injury_areas:
        injury_areas = _coerce_string_list(item.get("injury_areas"), [])
    if injury_areas:
        return f"Injury noted: {', '.join(injury_areas)}"
    return ""


def _group_daily_history_by_cycle(daily_history: list[dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in daily_history:
        cycle_number = _history_item_cycle_number(item)
        grouped.setdefault(cycle_number, []).append(item)
    return sorted(grouped.items(), key=lambda group: group[0], reverse=True)


def _history_item_cycle_number(item: dict[str, Any]) -> int:
    completed_plan = item.get("completed_plan", {})
    try:
        return int(item.get("cycle_number") or completed_plan.get("cycle_number") or 1)
    except (TypeError, ValueError):
        return 1


def _render_daily_feedback_section() -> None:
    st.subheader("Daily Feedback")
    st.caption("Save today's final plan and overall feeling, then move Today's Plan to the next calendar day.")

    profile_inputs = st.session_state.get("profile_inputs")
    result = st.session_state.get("agent_result")
    current_state = result.get("current_state", {}) if result else {}

    with st.form("daily_feedback_form", clear_on_submit=True):
        current_weight_kg = st.number_input(
            "Current Weight (kg)",
            min_value=35.0,
            max_value=250.0,
            value=float(current_state.get("weight_kg", 77.5) or 77.5),
        )
        current_body_fat_pct = st.number_input(
            "Current Body Fat (%)",
            min_value=3.0,
            max_value=60.0,
            value=float(current_state.get("body_fat_pct", 24.0) or 24.0),
            step=0.1,
        )
        workout_feeling = st.text_area(
            "How's it going?",
            placeholder="Example: training felt okay, meals were solid, energy was a little low.",
            key="feedback_workout_feeling",
        )
        feeling_emoji = st.radio(
            "How are you feeling today?",
            FEELING_EMOJI_OPTIONS,
            horizontal=True,
            format_func=lambda value: f"{value} {FEELING_EMOJI_LABELS[value]}",
            key="feedback_emoji",
        )
        submitted = st.form_submit_button("Make Tomorrow's Plan")

    if submitted:
        if not profile_inputs or not result:
            st.error("Please create your first plan from the User Profile first.")
            return

        current_reference_date = _current_interaction_date(result)
        current_session = _session_for_history_date(result, current_reference_date, {})

        completed_training_days = set(st.session_state.get("completed_training_days", []))
        if current_session.get("scheduled_date") and not current_session.get("is_cancelled"):
            completed_training_days.add(current_session["scheduled_date"])

        planned_days = {
            session.get("scheduled_date", "")
            for session in _sort_workout_sessions(result.get("current_plan", {}).get("workout_sessions", []))
            if session.get("scheduled_date") and not session.get("is_ad_hoc")
        }
        week_is_complete = bool(planned_days) and planned_days.issubset(completed_training_days)

        target_date = _next_calendar_date(current_reference_date)
        if week_is_complete:
            current_cycle_label = f"{result.get('current_plan', {}).get('cycle_number', 1)}"
            st.session_state["week_history"].append(
                {
                    "week_start": current_cycle_label,
                    "summary": result.get("current_plan", {}).get("summary", "Completed week"),
                }
            )
            target_date = _next_calendar_date(current_reference_date)
            st.session_state["active_date"] = target_date
            st.session_state["pending_homepage_date_picker"] = target_date
            st.session_state["completed_training_days"] = []
            st.session_state["last_action_message"] = (
                "Today's plan and feedback were saved. This cycle is now marked complete."
            )
        else:
            st.session_state["active_date"] = target_date
            st.session_state["pending_homepage_date_picker"] = target_date
            st.session_state["completed_training_days"] = sorted(completed_training_days)
            st.session_state["last_action_message"] = (
                "Today's plan and feedback were saved. Tomorrow's cycle plan is now shown as Today's Plan."
            )

        updated_result = _record_daily_feedback_and_advance(
            previous_result=result,
            current_session=current_session,
            feedback_date=current_reference_date,
            target_date=target_date,
            current_weight_kg=float(current_weight_kg),
            current_body_fat_pct=float(current_body_fat_pct),
            workout_feeling=workout_feeling,
            feeling_emoji=feeling_emoji,
        )
        st.session_state["agent_result"] = updated_result
        st.session_state["daily_history"] = updated_result.get("daily_history", [])
        st.session_state["last_feedback_summary"] = _daily_feedback_summary(
            workout_feeling=workout_feeling,
            feeling_emoji=feeling_emoji,
        )
        st.rerun()

    feedback_summary = st.session_state.get("last_feedback_summary", "")
    if feedback_summary:
        st.info(f"Saved feedback: {feedback_summary}")


def _record_daily_feedback_and_advance(
    *,
    previous_result: FitnessAgentState,
    current_session: dict[str, Any],
    feedback_date: str,
    target_date: str,
    current_weight_kg: float,
    current_body_fat_pct: float,
    workout_feeling: str,
    feeling_emoji: str,
) -> FitnessAgentState:
    updated_result: FitnessAgentState = dict(previous_result)
    current_session = _session_for_history_date(previous_result, feedback_date, current_session)
    completed_actions = _completed_actions_from_session(current_session)
    feedback_notes = workout_feeling.strip()
    latest_feedback = _build_daily_feedback(
        feedback_date=feedback_date,
        completed_actions=completed_actions,
        workout_feeling=feedback_notes,
        feeling_emoji=feeling_emoji,
        current_weight_kg=current_weight_kg,
        current_body_fat_pct=current_body_fat_pct,
    )
    today_state = {
        **dict(previous_result.get("current_state", {})),
        "date": feedback_date,
        "weight_kg": current_weight_kg,
        "body_fat_pct": current_body_fat_pct,
        "notes": _daily_feedback_summary(workout_feeling=feedback_notes, feeling_emoji=feeling_emoji),
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
        "cycle_number": _history_item_cycle_number({"completed_plan": current_session}),
        "weight_kg": current_weight_kg,
        "body_fat_pct": current_body_fat_pct,
        "completed_actions": completed_actions,
        "completed_plan": deepcopy(current_session) if current_session else {},
        "feedback": {
            "workout_feeling": feedback_notes,
            "emoji": feeling_emoji,
            "emoji_label": FEELING_EMOJI_LABELS.get(feeling_emoji, ""),
            "injury_areas": _history_injury_areas(current_session, latest_feedback),
        },
    }

    updated_result["current_date"] = target_date
    updated_result["current_state"] = tomorrow_state
    updated_result["latest_feedback"] = latest_feedback
    updated_result["state_history"] = _append_unique_history_item(
        previous_result.get("state_history", []),
        today_state,
        "date",
    )
    updated_result["feedback_history"] = _append_unique_history_item(
        previous_result.get("feedback_history", []),
        latest_feedback,
        "date",
    )
    updated_result["daily_history"] = _append_unique_history_item(
        previous_result.get("daily_history", []),
        daily_entry,
        "date",
    )
    return updated_result


def _session_for_history_date(
    result: FitnessAgentState | dict[str, Any],
    feedback_date: str,
    current_session: dict[str, Any],
) -> dict[str, Any]:
    if current_session and _same_iso_date(current_session.get("scheduled_date"), feedback_date):
        return current_session
    sessions = _sort_workout_sessions(result.get("current_plan", {}).get("workout_sessions", []))
    for session in sessions:
        if _same_iso_date(session.get("scheduled_date"), feedback_date):
            return session
    return {}


def _same_iso_date(left: object, right: object) -> bool:
    left_date = _safe_iso_date(left)
    right_date = _safe_iso_date(right)
    return bool(left_date and right_date and left_date == right_date)


def _safe_iso_date(value: object) -> str:
    if value is None:
        return ""
    raw_value = str(value).strip()
    try:
        return datetime.fromisoformat(raw_value).date().isoformat()
    except (TypeError, ValueError):
        pass
    compact_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw_value)
    if compact_match:
        year, month, day = compact_match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    chinese_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", raw_value)
    if chinese_match:
        year, month, day = chinese_match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    slash_match = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})", raw_value)
    if slash_match:
        year, month, day = slash_match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    return raw_value


def _history_injury_areas(current_session: dict[str, Any], latest_feedback: dict[str, Any]) -> list[str]:
    session_areas = _coerce_string_list(current_session.get("injury_areas"), [])
    if session_areas:
        return session_areas
    if current_session.get("injury_reported"):
        return ["reported injury area"]
    return list(latest_feedback.get("pain_points", []))


def _build_daily_feedback(
    *,
    feedback_date: str,
    completed_actions: list[str],
    workout_feeling: str,
    feeling_emoji: str,
    current_weight_kg: float,
    current_body_fat_pct: float,
) -> dict[str, Any]:
    fatigue_level, motivation_level, recovery_score = _emoji_training_signals(feeling_emoji)
    pain_points = _pain_points_from_text(workout_feeling)
    pain_level = 5 if pain_points else 0
    summary = _daily_feedback_summary(workout_feeling=workout_feeling, feeling_emoji=feeling_emoji)
    return {
        "date": feedback_date,
        "completed_workouts": completed_actions,
        "completed_actions": completed_actions,
        "feeling_emoji": feeling_emoji,
        "adherence_score": 1.0 if completed_actions else 0.0,
        "fatigue_level": fatigue_level,
        "pain_level": pain_level,
        "pain_points": pain_points,
        "soreness_areas": _soreness_areas_from_text(workout_feeling),
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


def _completed_actions_from_session(session: dict[str, Any]) -> list[str]:
    if not session or session.get("is_cancelled"):
        return []
    return [
        str(exercise.get("name", "")).strip()
        for exercise in session.get("exercises", [])
        if str(exercise.get("name", "")).strip()
    ]


def _emoji_training_signals(feeling_emoji: str) -> tuple[int, int, float]:
    if feeling_emoji == "😊":
        return 2, 9, 0.85
    if feeling_emoji == "😫":
        return 8, 3, 0.45
    return 5, 6, 0.7


def _daily_feedback_summary(*, workout_feeling: str, feeling_emoji: str) -> str:
    label = FEELING_EMOJI_LABELS.get(feeling_emoji, "Logged")
    feeling = workout_feeling.strip()
    if feeling:
        return f"{feeling_emoji} {label}: {feeling}"
    return f"{feeling_emoji} {label}"


def _pain_points_from_text(text: str) -> list[str]:
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


def _soreness_areas_from_text(text: str) -> list[str]:
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


def _build_initial_state(profile_inputs: dict[str, Any]) -> FitnessAgentState:
    target_date = _display_reference_date(
        st.session_state.get("active_date", profile_inputs.get("start_date", date.today().isoformat()))
    )
    return {
        "thread_id": st.session_state["thread_id"],
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
            "injuries": _split_csv(profile_inputs["injuries_text"]),
            "pain_sensitive_areas": [],
            "food_allergies": _split_csv(profile_inputs["allergies_text"]),
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
    }


def _build_change_request_state(
    *,
    profile_inputs: dict[str, Any],
    previous_result: FitnessAgentState,
    change_request: str,
    normalized_change_request: dict[str, Any] | None = None,
) -> FitnessAgentState:
    target_date = _current_interaction_date(previous_result)
    previous_feedback = dict(previous_result.get("latest_feedback", {}))
    previous_state = dict(previous_result.get("current_state", {}))
    normalized_change_request = normalized_change_request or _normalize_change_request(change_request)
    change_request_context = _build_change_request_context(change_request, normalized_change_request)
    previous_notes = " ".join(
        part
        for part in [
            previous_state.get("notes", ""),
            change_request_context,
        ]
        if part
    )
    previous_feedback["performance_notes"] = previous_notes

    base_state = _build_initial_state(profile_inputs)
    return {
        "thread_id": st.session_state["thread_id"],
        "current_date": target_date,
        "profile_notes": profile_inputs.get("profile_notes", ""),
        "plan_change_request": change_request,
        "normalized_change_request": normalized_change_request,
        "user_profile": base_state["user_profile"],
        "constraints": base_state["constraints"],
        "goals": base_state["goals"],
        "current_state": {
            **previous_state,
            "date": target_date,
            "notes": previous_notes,
        },
        "latest_feedback": {
            **previous_feedback,
            "date": target_date,
        },
        "current_plan": previous_result.get("current_plan", {}),
        "plan_history": previous_result.get("plan_history", []),
        "daily_history": previous_result.get("daily_history", []),
        "feedback_history": previous_result.get("feedback_history", []),
        "state_history": _append_unique_history_item(
            previous_result.get("state_history", []),
            previous_result.get("current_state", {}),
            "date",
        ),
    }


def _build_feedback_update_state(
    *,
    profile_inputs: dict[str, Any],
    previous_result: FitnessAgentState,
    completed_workouts: str,
    sleep_hours: float,
    current_weight_kg: float,
    current_body_fat_pct: float,
    feelings: str,
    inferred_feedback: dict[str, Any],
    feedback_date: str,
    target_date: str,
    week_rollover: bool,
) -> FitnessAgentState:
    completed_workout_list = _split_csv(completed_workouts)
    previous_weight_kg = _previous_weight_from_result(previous_result)
    previous_body_fat_pct = _previous_body_fat_from_result(previous_result)
    feedback_notes = _compose_feedback_notes(
        feelings=feelings,
        previous_weight_kg=previous_weight_kg,
        current_weight_kg=current_weight_kg,
        previous_body_fat_pct=previous_body_fat_pct,
        current_body_fat_pct=current_body_fat_pct,
    )
    inferred_summary = str(inferred_feedback.get("summary", "")).strip()
    if inferred_summary and inferred_summary not in feedback_notes:
        feedback_notes = " ".join(part for part in [feedback_notes, inferred_summary] if part)
    normalized_feedback_request = _build_feedback_normalized_change_request(
        feelings=feelings,
        inferred_feedback=inferred_feedback,
    )
    feedback_plan_request = _build_feedback_plan_request(
        feelings=feelings,
        normalized_change_request=normalized_feedback_request,
    )
    if week_rollover:
        feedback_notes = " ".join(
            part
            for part in [
                feedback_notes,
                "Generate the next week's plan after incorporating this completed-week feedback.",
            ]
            if part
        )

    base_state = _build_initial_state(profile_inputs)
    return {
        "thread_id": st.session_state["thread_id"],
        "current_date": target_date,
        "profile_notes": profile_inputs.get("profile_notes", ""),
        "plan_change_request": feedback_plan_request,
        "normalized_change_request": normalized_feedback_request,
        "user_profile": base_state["user_profile"],
        "constraints": base_state["constraints"],
        "goals": base_state["goals"],
        "current_state": {
            "date": target_date,
            "weight_kg": current_weight_kg,
            "body_fat_pct": current_body_fat_pct,
            "sleep_hours": sleep_hours,
            "recovery_score": float(inferred_feedback["recovery_score"]),
            "notes": feedback_notes,
        },
        "latest_feedback": {
            "date": feedback_date,
            "completed_workouts": completed_workout_list,
            "adherence_score": float(inferred_feedback["adherence_score"]),
            "fatigue_level": int(inferred_feedback["fatigue_level"]),
            "pain_level": int(inferred_feedback["pain_level"]),
            "pain_points": list(inferred_feedback["pain_points"]),
            "soreness_areas": list(inferred_feedback["soreness_areas"]),
            "motivation_level": int(inferred_feedback["motivation_level"]),
            "performance_notes": feedback_notes,
            "manual_log": {
                "date": feedback_date,
                "sleep_hours": sleep_hours,
                "weight_kg": current_weight_kg,
                "body_fat_pct": current_body_fat_pct,
                "notes": feedback_notes,
            },
        },
        "current_plan": previous_result.get("current_plan", {}),
        "plan_history": previous_result.get("plan_history", []),
        "daily_history": previous_result.get("daily_history", []),
        "feedback_history": previous_result.get("feedback_history", []),
        "state_history": _append_unique_history_item(
            previous_result.get("state_history", []),
            previous_result.get("current_state", {}),
            "date",
        ),
    }


def _build_feedback_normalized_change_request(*, feelings: str, inferred_feedback: dict[str, Any]) -> dict[str, Any]:
    text = feelings.lower()
    pain_points = _coerce_string_list(inferred_feedback.get("pain_points"), [])
    injury_reported = _feedback_reports_injury(text, inferred_feedback)
    intensity_adjustment = ""

    if not injury_reported:
        fatigue_level = int(inferred_feedback.get("fatigue_level", 4))
        motivation_level = int(inferred_feedback.get("motivation_level", 7))
        recovery_score = float(inferred_feedback.get("recovery_score", 0.75))
        if _feedback_sounds_low(text) or fatigue_level >= 7 or recovery_score <= 0.5 or motivation_level <= 4:
            intensity_adjustment = "lower"
        elif _feedback_sounds_high(text) or (fatigue_level <= 2 and recovery_score >= 0.85 and motivation_level >= 8):
            intensity_adjustment = "higher"

    summary_bits = []
    if injury_reported:
        summary_bits.append("Injury or pain was reported; cancel the target day's workout.")
    elif intensity_adjustment == "lower":
        summary_bits.append("The user sounds under-recovered or not good; lower reps/notes and reduce one exercise only for intermediate or advanced users.")
    elif intensity_adjustment == "higher":
        summary_bits.append("The user sounds well-recovered or strong; add one exercise and use higher reps/notes.")

    return {
        "request_type": "recovery_change" if injury_reported else "workout_change",
        "scope": "today_only",
        "focus_category": "",
        "injury_reported": injury_reported,
        "injury_areas": pain_points,
        "cancel_today": injury_reported,
        "intensity_adjustment": intensity_adjustment,
        "duration_adjustment": "",
        "temporary_food_avoidances": [],
        "permanent_food_preferences": [],
        "summary": " ".join(summary_bits),
        "confidence": 0.9 if summary_bits else 0.7,
    }


def _build_feedback_plan_request(*, feelings: str, normalized_change_request: dict[str, Any]) -> str:
    feeling_text = feelings.strip()
    adjustment = str(normalized_change_request.get("intensity_adjustment", "")).strip()
    if normalized_change_request.get("injury_reported"):
        return f"Daily feedback for tomorrow: {feeling_text} Injury or pain means cancel the target day's workout."
    if adjustment:
        return f"Daily feedback for tomorrow: {feeling_text} Intensity adjustment={adjustment}."
    if feeling_text:
        return f"Daily feedback for tomorrow: {feeling_text} Use normal training volume unless the feedback clearly implies otherwise."
    return ""


def _feedback_reports_injury(text: str, inferred_feedback: dict[str, Any]) -> bool:
    injury_terms = [
        "injured",
        "injury",
        "hurt",
        "hurts",
        "pain",
        "painful",
        "ache",
        "strained",
        "strain",
        "sprained",
        "sprain",
        "pulled",
        "受伤",
        "疼",
        "痛",
        "拉伤",
        "扭伤",
    ]
    if any(term in text for term in injury_terms):
        return True
    return int(inferred_feedback.get("pain_level", 0)) >= 4 or bool(inferred_feedback.get("pain_points"))


def _feedback_sounds_low(text: str) -> bool:
    low_terms = [
        "not good",
        "bad",
        "awful",
        "tired",
        "exhausted",
        "fatigued",
        "low energy",
        "drained",
        "stressed",
        "struggling",
        "hard today",
        "sleep badly",
        "slept badly",
        "状态不好",
        "不太好",
        "很累",
        "太累",
        "没精神",
        "压力大",
    ]
    return any(term in text for term in low_terms)


def _feedback_sounds_high(text: str) -> bool:
    high_terms = [
        "feel good",
        "feeling good",
        "great",
        "excellent",
        "strong",
        "energetic",
        "easy",
        "too easy",
        "ready for more",
        "状态很好",
        "感觉很好",
        "精力很好",
        "很轻松",
        "太简单",
        "加大",
        "加强",
    ]
    return any(term in text for term in high_terms)


def _infer_feedback_signals(
    *,
    completed_workouts: str,
    sleep_hours: float,
    current_weight_kg: float,
    current_body_fat_pct: float,
    feelings: str,
    previous_weight_kg: float | None = None,
    previous_body_fat_pct: float | None = None,
    assume_completed: bool = False,
) -> dict[str, Any]:
    if assume_completed:
        return _assumed_completion_feedback(sleep_hours)
    prompt_payload = {
        "completed_workouts": _split_csv(completed_workouts),
        "sleep_hours": sleep_hours,
        "feelings": feelings,
    }
    try:
        inferred = call_model_json(
            system_prompt=load_prompt("feedback_prompt.txt"),
            user_prompt=json.dumps(prompt_payload, ensure_ascii=True, indent=2),
            temperature=0.1,
            max_tokens=1200,
        )
    except Exception:
        return _fallback_feedback_inference(
            completed_workouts,
            sleep_hours,
            feelings,
            current_weight_kg=current_weight_kg,
            previous_weight_kg=previous_weight_kg,
            current_body_fat_pct=current_body_fat_pct,
            previous_body_fat_pct=previous_body_fat_pct,
        )
    return _sanitize_inferred_feedback(
        inferred,
        completed_workouts,
        sleep_hours,
        feelings,
        current_weight_kg=current_weight_kg,
        previous_weight_kg=previous_weight_kg,
        current_body_fat_pct=current_body_fat_pct,
        previous_body_fat_pct=previous_body_fat_pct,
    )


def _assumed_completion_feedback(sleep_hours: float) -> dict[str, Any]:
    recovery_score = 0.8
    fatigue_level = 3
    if sleep_hours < 6.0:
        recovery_score = 0.6
        fatigue_level = 5
    elif sleep_hours < 7.0:
        recovery_score = 0.7
        fatigue_level = 4
    return {
        "adherence_score": 1.0,
        "fatigue_level": fatigue_level,
        "pain_level": 0,
        "motivation_level": 7,
        "recovery_score": recovery_score,
        "pain_points": [],
        "soreness_areas": [],
        "summary": "No extra notes were provided, so the agent assumed the planned work was completed as scheduled.",
    }


def _sanitize_inferred_feedback(
    inferred: dict[str, Any],
    completed_workouts: str,
    sleep_hours: float,
    feelings: str,
    current_weight_kg: float,
    previous_weight_kg: float | None,
    current_body_fat_pct: float,
    previous_body_fat_pct: float | None,
) -> dict[str, Any]:
    fallback = _fallback_feedback_inference(
        completed_workouts,
        sleep_hours,
        feelings,
        current_weight_kg=current_weight_kg,
        previous_weight_kg=previous_weight_kg,
        current_body_fat_pct=current_body_fat_pct,
        previous_body_fat_pct=previous_body_fat_pct,
    )
    return {
        "adherence_score": _clamp_float(inferred.get("adherence_score"), 0.0, 1.0, fallback["adherence_score"]),
        "fatigue_level": _clamp_int(inferred.get("fatigue_level"), 0, 10, fallback["fatigue_level"]),
        "pain_level": _clamp_int(inferred.get("pain_level"), 0, 10, fallback["pain_level"]),
        "motivation_level": _clamp_int(inferred.get("motivation_level"), 0, 10, fallback["motivation_level"]),
        "recovery_score": _clamp_float(inferred.get("recovery_score"), 0.0, 1.0, fallback["recovery_score"]),
        "pain_points": _coerce_string_list(inferred.get("pain_points"), fallback["pain_points"]),
        "soreness_areas": _coerce_string_list(inferred.get("soreness_areas"), fallback["soreness_areas"]),
        "summary": str(inferred.get("summary") or fallback["summary"]),
    }


def _fallback_feedback_inference(
    completed_workouts: str,
    sleep_hours: float,
    feelings: str,
    *,
    current_weight_kg: float,
    previous_weight_kg: float | None,
    current_body_fat_pct: float,
    previous_body_fat_pct: float | None,
) -> dict[str, Any]:
    text = feelings.lower()
    completed = _split_csv(completed_workouts)

    pain_keywords = ["pain", "hurt", "sore", "ache", "injury", "knee", "back", "ankle", "shoulder"]
    tired_keywords = ["tired", "exhausted", "drained", "fatigue", "sleepy", "low energy"]
    low_mood_keywords = ["unmotivated", "stressed", "down", "bad", "hard", "struggling"]

    pain_level = 0
    if any(keyword in text for keyword in pain_keywords):
        pain_level = 5
    fatigue_level = 4
    if sleep_hours < 6 or any(keyword in text for keyword in tired_keywords):
        fatigue_level = 7
    motivation_level = 7
    if any(keyword in text for keyword in low_mood_keywords):
        motivation_level = 4
    adherence_score = 0.85 if completed else 0.55
    recovery_score = 0.75
    if sleep_hours < 6:
        recovery_score = 0.5
    elif sleep_hours < 7:
        recovery_score = 0.65

    pain_points = [body_part for body_part in ["knee", "back", "shoulder", "ankle", "hip"] if body_part in text]
    soreness_areas = [
        area
        for area in ["legs", "quads", "hamstrings", "glutes", "chest", "back", "arms"]
        if area in text
    ]

    summary = "The agent inferred your update using a lightweight fallback."

    return {
        "adherence_score": adherence_score,
        "fatigue_level": fatigue_level,
        "pain_level": pain_level,
        "motivation_level": motivation_level,
        "recovery_score": recovery_score,
        "pain_points": pain_points,
        "soreness_areas": soreness_areas,
        "summary": summary,
    }


def _execute_agent(state: FitnessAgentState) -> None:
    try:
        result = run_agent(state)
    except Exception as exc:
        st.error(str(exc))
    else:
        st.session_state["agent_result"] = result


def _render_nutrition_targets(targets: dict[str, Any]) -> None:
    if not targets:
        st.write("No nutrition plan yet.")
        return
    st.write(
        f"Daily calories: {targets.get('daily_calories', '-')}, "
        f"protein: {targets.get('protein_g', '-')}g, "
        f"carbs: {targets.get('carbs_g', '-')}g, "
        f"fat: {targets.get('fat_g', '-')}g"
    )
    st.write(f"Hydration target: {targets.get('hydration_liters', '-')} L")


def _select_today_session(workout_sessions: list[dict[str, Any]], reference_date: str) -> dict[str, Any]:
    if not workout_sessions:
        return {}
    for session in workout_sessions:
        if _same_iso_date(session.get("scheduled_date"), reference_date):
            return session
    return {}


def _videos_for_session(session: dict[str, Any], resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not session:
        return []
    exercise_names = {exercise.get("name", "") for exercise in session.get("exercises", [])}
    seen_urls = set()
    filtered = []
    for resource in resources:
        if resource.get("exercise_name") not in exercise_names:
            continue
        url = resource.get("url")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        filtered.append(resource)
    return filtered


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _default_equipment_access() -> list[str]:
    return ["bodyweight", "dumbbell", "barbell", "bench", "rack", "cable_machine", "kettlebell", "box"]


def _week_start_from_iso(value: str) -> date:
    parsed = datetime.fromisoformat(value).date()
    return parsed - timedelta(days=parsed.weekday())


def _display_reference_date(value: str) -> str:
    iso_value = _safe_iso_date(value)
    if not iso_value:
        return date.today().isoformat()
    try:
        return datetime.fromisoformat(iso_value).date().isoformat()
    except ValueError:
        return date.today().isoformat()


def _current_interaction_date(result: FitnessAgentState | dict[str, Any] | None = None) -> str:
    picker_value = st.session_state.get("homepage_date_picker")
    if isinstance(picker_value, datetime):
        return picker_value.date().isoformat()
    if isinstance(picker_value, date):
        return picker_value.isoformat()
    if picker_value:
        return _display_reference_date(str(picker_value))
    result = result or {}
    return _display_reference_date(
        st.session_state.get("active_date", result.get("current_date", date.today().isoformat()))
    )


def _iso_to_date(value: str) -> date:
    try:
        return datetime.fromisoformat(_safe_iso_date(value)).date()
    except ValueError:
        return date.today()


def _sort_days(days: list[str]) -> list[str]:
    normalized_days: list[str] = []
    for day in days:
        cleaned = str(day).strip()
        if cleaned in WEEKDAY_INDEX and cleaned not in normalized_days:
            normalized_days.append(cleaned)
    return sorted(normalized_days, key=lambda item: WEEKDAY_INDEX[item])


def _sort_workout_sessions(workout_sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        workout_sessions,
        key=lambda session: (
            str(session.get("scheduled_date", "")) or "9999-12-31",
            WEEKDAY_INDEX.get(str(session.get("day", "")), 99),
        ),
    )


def _next_calendar_date(value: str) -> str:
    return (_iso_to_date(value) + timedelta(days=1)).isoformat()


def _default_completed_workouts_text(session: dict[str, Any], *, assume_completed: bool) -> str:
    if not assume_completed or not session:
        return ""
    focus = str(session.get("focus", "")).strip()
    if focus:
        return f"{focus} session"
    return "planned workout"


def _previous_weight_from_result(result: FitnessAgentState) -> float | None:
    current_state = result.get("current_state", {})
    value = current_state.get("weight_kg")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _previous_body_fat_from_result(result: FitnessAgentState) -> float | None:
    current_state = result.get("current_state", {})
    value = current_state.get("body_fat_pct")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _weight_change_note(previous_weight_kg: float | None, current_weight_kg: float) -> str:
    if previous_weight_kg is None:
        return ""
    delta = current_weight_kg - previous_weight_kg
    if delta >= 0.8:
        return (
            f"Weight is up from {previous_weight_kg:.1f}kg to {current_weight_kg:.1f}kg since the last check-in. "
            "Treat this as a real update and review nutrition, recovery, and adherence supportively."
        )
    return ""


def _body_fat_change_note(previous_body_fat_pct: float | None, current_body_fat_pct: float) -> str:
    if previous_body_fat_pct is None:
        return ""
    delta = current_body_fat_pct - previous_body_fat_pct
    if delta >= 0.4:
        return (
            f"Body fat is up from {previous_body_fat_pct:.1f}% to {current_body_fat_pct:.1f}% since the last check-in. "
            "Treat this as a real update and review nutrition quality, conditioning, and recovery supportively."
        )
    return ""


def _compose_feedback_notes(
    *,
    feelings: str,
    previous_weight_kg: float | None,
    current_weight_kg: float,
    previous_body_fat_pct: float | None,
    current_body_fat_pct: float,
) -> str:
    return feelings.strip()


def _session_display_label(session: dict[str, Any]) -> str:
    scheduled_date = str(session.get("scheduled_date", "")).strip()
    if scheduled_date:
        try:
            parsed = datetime.fromisoformat(scheduled_date)
            return f"{parsed.strftime('%a %Y-%m-%d')} · {session.get('focus', '')}"
        except ValueError:
            pass
    return f"{session.get('day', '')} - {session.get('focus', '')}"


def _build_change_request_context(change_request: str, normalized_change_request: dict[str, Any]) -> str:
    request = change_request.strip()
    if not request:
        return ""
    normalized_summary = str(normalized_change_request.get("summary", "")).strip()
    focus_category = str(normalized_change_request.get("focus_category", "")).strip()
    normalized_bits = []
    if normalized_summary:
        normalized_bits.append(f"normalized intent: {normalized_summary}")
    if focus_category:
        normalized_bits.append(f"focus category={focus_category}")
    normalized_text = "; ".join(normalized_bits).strip()
    return (
        "Temporary same-day request only: "
        f"{request} "
        f"{normalized_text} "
        "If food swaps are mentioned here, apply them to today's plan only unless the user explicitly says they are permanent."
    )


def _normalize_change_request(change_request: str) -> dict[str, Any]:
    request = change_request.strip()
    if not request:
        return {}

    payload = {
        "user_request": request,
        "allowed_focus_categories": sorted(category for category in ALLOWED_FOCUS_CATEGORIES if category),
        "allowed_scopes": sorted(ALLOWED_CHANGE_REQUEST_SCOPES),
        "allowed_request_types": sorted(ALLOWED_CHANGE_REQUEST_TYPES),
    }
    try:
        normalized = call_model_json(
            system_prompt=load_prompt("change_request_prompt.txt"),
            user_prompt=json.dumps(payload, ensure_ascii=True, indent=2),
            temperature=0.1,
            max_tokens=1000,
        )
    except Exception:
        return {}
    return _sanitize_normalized_change_request(normalized)


def _sanitize_normalized_change_request(normalized: dict[str, Any]) -> dict[str, Any]:
    request_type = str(normalized.get("request_type", "unclear")).strip().lower()
    if request_type not in ALLOWED_CHANGE_REQUEST_TYPES:
        request_type = "unclear"

    scope = str(normalized.get("scope", "unclear")).strip().lower()
    if scope not in ALLOWED_CHANGE_REQUEST_SCOPES:
        scope = "unclear"

    focus_category = str(normalized.get("focus_category", "")).strip()
    if focus_category not in ALLOWED_FOCUS_CATEGORIES:
        focus_category = ""

    return {
        "request_type": request_type,
        "scope": scope,
        "focus_category": focus_category,
        "injury_reported": bool(normalized.get("injury_reported", False)),
        "injury_areas": _coerce_string_list(normalized.get("injury_areas"), []),
        "cancel_today": bool(normalized.get("cancel_today", False)),
        "intensity_adjustment": str(normalized.get("intensity_adjustment", "")).strip(),
        "duration_adjustment": str(normalized.get("duration_adjustment", "")).strip(),
        "temporary_food_avoidances": _coerce_string_list(normalized.get("temporary_food_avoidances"), []),
        "permanent_food_preferences": _coerce_string_list(normalized.get("permanent_food_preferences"), []),
        "summary": str(normalized.get("summary", "")).strip(),
        "confidence": _clamp_float(normalized.get("confidence"), 0.0, 1.0, 0.0),
    }


def _append_unique_history_item(history: list[dict], item: dict, date_key: str) -> list[dict]:
    if not item:
        return list(history)
    updated_history = list(history)
    item_date = item.get(date_key)
    if item_date:
        for index, existing in enumerate(updated_history):
            if _same_iso_date(existing.get(date_key), item_date) or existing.get(date_key) == item_date:
                updated_history[index] = item
                return updated_history
    updated_history.append(item)
    return updated_history


def _has_real_feedback(result: FitnessAgentState) -> bool:
    latest_feedback = result.get("latest_feedback", {})
    if not latest_feedback:
        return False
    if latest_feedback.get("completed_workouts"):
        return True
    if latest_feedback.get("completed_actions") or latest_feedback.get("feeling_emoji"):
        return True
    if latest_feedback.get("pain_points") or latest_feedback.get("soreness_areas"):
        return True
    notes = str(latest_feedback.get("performance_notes", "")).strip()
    manual_notes = str(latest_feedback.get("manual_log", {}).get("notes", "")).strip()
    return bool(notes or manual_notes)


def _clamp_float(value: Any, minimum: float, maximum: float, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, numeric))


def _clamp_int(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, numeric))


def _coerce_string_list(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return fallback
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    return cleaned or fallback


if __name__ == "__main__":
    main()
