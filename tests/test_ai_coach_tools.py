from __future__ import annotations

from copy import deepcopy
from datetime import datetime

import pytest

from agent.services import coach_chat_service as coach
from agent.services.mysql_store import _structured_rows_from_payload
from agent.services.memory import default_memory_store
from agent.tools.rag_tool import search_similar_exercises


def _sample_result() -> dict:
    return {
        "current_date": "2026-04-27",
        "user_profile": {"fitness_level": "beginner"},
        "goals": {"primary_goal": "fat_loss"},
        "current_plan": {
            "cycle_number": 1,
            "cycle_start_date": "2026-04-24",
            "cycle_end_date": "2026-04-30",
            "nutrition_targets": {
                "daily_calories": 2184,
                "protein_g": 140,
                "carbs_g": 266,
                "fat_g": 62,
            },
            "meal_suggestions": [
                {"meal_slot": "breakfast", "food_name": "Chicken Breast", "serving_size": "150g"},
                {"meal_slot": "dinner", "food_name": "Broccoli", "serving_size": "150g"},
            ],
            "workout_sessions": [
                {
                    "day": "Monday",
                    "scheduled_date": "2026-04-27",
                    "cycle_number": 1,
                    "focus": "Lower Body (Legs + Glutes)",
                    "duration_minutes": 60,
                    "warmup": ["walk"],
                    "exercises": [
                        {"name": "Goblet Squat", "sets": 4, "reps": "6-10", "notes": ""},
                        {"name": "Glute Bridge", "sets": 4, "reps": "6-10", "notes": ""},
                    ],
                    "cooldown": ["stretch"],
                    "safety_notes": ["Move well."],
                },
                {
                    "day": "Wednesday",
                    "scheduled_date": "2026-04-29",
                    "cycle_number": 1,
                    "focus": "Back Training",
                    "duration_minutes": 60,
                    "warmup": ["band pull-aparts"],
                    "exercises": [
                        {"name": "Lat Pulldown", "sets": 4, "reps": "6-10", "notes": ""},
                        {"name": "Seated Cable Row", "sets": 4, "reps": "6-10", "notes": ""},
                    ],
                    "cooldown": ["lat stretch"],
                    "safety_notes": ["Keep spine neutral."],
                },
            ],
        },
        "latest_feedback": {},
        "plan_history": [],
        "daily_history": [],
        "feedback_history": [],
        "state_history": [],
    }


def _session_state() -> dict:
    return {
        "active_date": "2026-04-27",
        "thread_id": "test-thread",
        "memory_store": default_memory_store(),
    }


def test_mysql_structured_rows_mirror_daily_memory_and_chat() -> None:
    payload = {
        "daily_history": [
            {
                "date": "2026-04-27",
                "cycle_number": 1,
                "plan_focus": "Lower Body (Legs + Glutes)",
                "status": "completed",
                "weight_kg": 78.0,
                "body_fat_pct": 24.0,
                "completed_actions": [{"name": "Goblet Squat", "sets": 4, "reps": "6-10"}],
                "feedback": {"emoji": "😊", "workout_feeling": "solid day"},
            }
        ],
        "memory_store": {
            **default_memory_store(),
            "plan_modification_logs": [
                {
                    "date": "2026-04-27",
                    "action_type": "adjust_sets",
                    "summary": "Added one set.",
                    "injury_areas": [],
                }
            ],
            "injury_events": [
                {"date": "2026-04-29", "area": "back", "status": "active"},
            ],
        },
        "assistant_chat_messages": [
            {"role": "user", "content": "add sets"},
            {"role": "assistant", "content": "Sets increased."},
        ],
    }

    rows = _structured_rows_from_payload(payload, "demo-user", datetime(2026, 4, 27))

    assert rows["body_metrics"][0]["record_date"] == "2026-04-27"
    assert rows["body_metrics"][0]["weight_kg"] == 78.0
    assert rows["daily_feedback_records"][0]["workout_status"] == "completed"
    assert rows["daily_feedback_records"][0]["feeling_emoji"] == "😊"
    assert len(rows["chat_messages"]) == 2
    assert rows["plan_modification_logs"][0]["action_type"] == "adjust_sets"
    assert rows["memory_events"][0]["event_type"] == "injury_events"


def _tool_context(result: dict, session_state: dict, user_message: str = "") -> dict:
    return {
        "user_message": user_message,
        "profile_inputs": {
            "fitness_level": "beginner",
            "start_date": "2026-04-27",
            "available_days": ["Monday", "Wednesday"],
        },
        "previous_result": result,
        "session_state": session_state,
        "decision": {"action_message": "Today's plan was updated by AI Coach."},
    }


def _today(result: dict) -> dict:
    return result["current_plan"]["workout_sessions"][0]


def _future_back(result: dict) -> dict:
    return result["current_plan"]["workout_sessions"][1]


def test_tool_selection_distinguishes_sets_from_vague_intensity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coach, "call_model_json", lambda **_: (_ for _ in ()).throw(RuntimeError("no llm")))

    add_sets_decision = coach.fallback_coordinator_decision("add sets today", _sample_result())
    add_sets_tool = coach.fallback_tool_call_from_decision(add_sets_decision)
    assert add_sets_tool["tool_name"] == "adjust_sets"
    assert add_sets_tool["arguments"]["set_adjustment"] == "increase"

    do_more_decision = coach.fallback_coordinator_decision("do more today", _sample_result())
    do_more_tool = coach.fallback_tool_call_from_decision(do_more_decision)
    assert do_more_tool["tool_name"] == "adjust_intensity"
    assert do_more_tool["arguments"]["intensity_adjustment"] == "higher"


def test_rag_search_returns_same_focus_exercise_alternative() -> None:
    replacements = search_similar_exercises(
        exercise_name="Seated Dumbbell Shoulder Press",
        focus="upper_shoulders",
        level="beginner",
        exclude=["Seated Dumbbell Shoulder Press", "Dumbbell Lateral Raise"],
        limit=3,
    )

    assert replacements
    assert replacements[0]["name"] != "Seated Dumbbell Shoulder Press"
    assert "upper_shoulders" in replacements[0].get("focus_tags", [])


def test_adjust_sets_changes_sets_only_and_keeps_exercise_count() -> None:
    result = _sample_result()
    session_state = _session_state()

    summary = coach.execute_coach_tool_call(
        {"tool_name": "adjust_sets", "arguments": {"set_adjustment": "increase"}},
        _tool_context(result, session_state),
    )

    updated = session_state["agent_result"]
    today = _today(updated)
    assert "without adding exercises" in summary
    assert [exercise["sets"] for exercise in today["exercises"]] == [5, 5]
    assert [exercise["name"] for exercise in today["exercises"]] == ["Goblet Squat", "Glute Bridge"]


def test_add_exercise_after_add_sets_keeps_set_policy() -> None:
    result = _sample_result()
    session_state = _session_state()

    coach.execute_coach_tool_call(
        {"tool_name": "adjust_sets", "arguments": {"set_adjustment": "increase"}},
        _tool_context(result, session_state, "add sets"),
    )
    after_sets = session_state["agent_result"]

    coach.execute_coach_tool_call(
        {"tool_name": "adjust_intensity", "arguments": {"intensity_adjustment": "higher"}},
        _tool_context(after_sets, session_state, "add one more exercise"),
    )

    today = _today(session_state["agent_result"])
    assert len(today["exercises"]) >= 2
    assert {exercise["sets"] for exercise in today["exercises"]} == {5}


def test_set_policy_persists_after_later_workout_patch() -> None:
    previous = _sample_result()
    for exercise in _today(previous)["exercises"]:
        exercise["sets"] = 5
    patched = deepcopy(previous)
    patched["current_plan"]["workout_sessions"][0] = {
        **deepcopy(_today(previous)),
        "focus": "Upper Body (Shoulders)",
        "exercises": [
            {"name": "Standing Overhead Press", "sets": 4, "reps": "6-10"},
            {"name": "Dumbbell Lateral Raise", "sets": 4, "reps": "6-10"},
            {"name": "Face Pull", "sets": 4, "reps": "6-10"},
        ],
    }

    merged = coach.merge_ai_patch_result(
        previous,
        patched,
        {"request_type": "workout_change", "focus_category": "upper_shoulders"},
    )

    assert [exercise["sets"] for exercise in _today(merged)["exercises"]] == [5, 5, 5]


def test_replace_food_preserves_workout(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _sample_result()
    session_state = _session_state()
    original_workout = deepcopy(result["current_plan"]["workout_sessions"])

    def fake_replacement_food_for_meal(*, meal: dict, avoidances: list[str], used_foods: set[str]) -> dict:
        return {
            "meal_slot": meal["meal_slot"],
            "food_name": "Spinach",
            "serving_size": meal["serving_size"],
            "calories": 30,
            "protein_g": 3,
            "carbs_g": 5,
            "fat_g": 0,
        }

    monkeypatch.setattr(coach, "replacement_food_for_meal", fake_replacement_food_for_meal)

    summary = coach.execute_coach_tool_call(
        {"tool_name": "replace_food", "arguments": {"temporary_food_avoidances": ["broccoli"]}},
        _tool_context(result, session_state, "replace broccoli"),
    )

    updated = session_state["agent_result"]
    assert "Broccoli -> Spinach" in summary
    assert updated["current_plan"]["workout_sessions"] == original_workout
    assert "Spinach" in [meal["food_name"] for meal in updated["current_plan"]["meal_suggestions"]]


def test_replace_food_after_workout_change_preserves_changed_workout(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _sample_result()
    session_state = _session_state()

    coach.execute_coach_tool_call(
        {"tool_name": "adjust_sets", "arguments": {"set_adjustment": "increase"}},
        _tool_context(result, session_state, "add sets"),
    )
    changed_workout = deepcopy(session_state["agent_result"]["current_plan"]["workout_sessions"])

    def fake_replacement_food_for_meal(*, meal: dict, avoidances: list[str], used_foods: set[str]) -> dict:
        return {
            "meal_slot": meal["meal_slot"],
            "food_name": "Spinach",
            "serving_size": meal["serving_size"],
            "calories": 30,
            "protein_g": 3,
            "carbs_g": 5,
            "fat_g": 0,
        }

    monkeypatch.setattr(coach, "replacement_food_for_meal", fake_replacement_food_for_meal)

    coach.execute_coach_tool_call(
        {"tool_name": "replace_food", "arguments": {"temporary_food_avoidances": ["broccoli"]}},
        _tool_context(session_state["agent_result"], session_state, "replace broccoli"),
    )

    updated = session_state["agent_result"]
    assert updated["current_plan"]["workout_sessions"] == changed_workout
    assert [exercise["sets"] for exercise in _today(updated)["exercises"]] == [5, 5]


def test_cancel_today_without_injury_only_changes_today() -> None:
    result = _sample_result()
    session_state = _session_state()

    coach.execute_coach_tool_call(
        {"tool_name": "cancel_workout", "arguments": {"cancel_today": True, "injury_reported": False}},
        _tool_context(result, session_state, "cancel today's plan"),
    )

    updated = session_state["agent_result"]
    assert _today(updated)["is_cancelled"] is True
    assert _today(updated)["focus"] == "Workout Cancelled"
    assert _future_back(updated).get("is_cancelled") is not True
    assert _future_back(updated)["focus"] == "Back Training"


def test_back_injury_cancels_today_and_protects_related_future_session() -> None:
    result = _sample_result()
    session_state = _session_state()

    coach.execute_coach_tool_call(
        {
            "tool_name": "cancel_workout",
            "arguments": {
                "cancel_today": True,
                "injury_reported": True,
                "injury_areas": ["back"],
            },
        },
        _tool_context(result, session_state, "back injured"),
    )

    updated = session_state["agent_result"]
    assert _today(updated)["is_cancelled"] is True
    assert _future_back(updated)["is_cancelled"] is True
    assert _future_back(updated)["focus"] == "Recovery (injury protection)"
    injury_events = session_state["memory_store"]["injury_events"]
    assert injury_events
    assert injury_events[-1]["area"] == "back"
