from __future__ import annotations

from copy import deepcopy
from datetime import datetime

import pytest

import agent.tools.exercise_tool as exercise_tool
from agent.services import coach_chat_service as coach
from agent.services.feedback_service import record_memory_daily_feedback
from agent.services.mysql_store import _structured_rows_from_payload
from agent.services.memory import compact_conversation_memory, default_memory_store, memory_context_for_planning
from agent.tools.rag_tool import replacement_candidate_score, search_similar_exercises


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
    assert rows["exercise_feedback_records"][0]["exercise_name"] == "Goblet Squat"
    assert rows["exercise_feedback_records"][0]["sets_count"] == 4
    assert rows["exercise_feedback_records"][0]["reps"] == "6-10"
    assert len(rows["chat_messages"]) == 2
    assert rows["plan_modification_logs"][0]["action_type"] == "adjust_sets"
    assert rows["memory_events"][0]["event_type"] == "injury_events"


def test_daily_feedback_memory_records_exercise_level_feedback() -> None:
    daily_entry = {
        "date": "2026-04-27",
        "cycle_number": 1,
        "plan_focus": "Lower Body (Legs + Glutes)",
        "status": "completed",
        "weight_kg": 78.0,
        "body_fat_pct": 24.0,
        "completed_actions": ["Goblet Squat", "Glute Bridge"],
        "completed_plan": {
            "cycle_number": 1,
            "focus": "Lower Body (Legs + Glutes)",
            "exercises": [
                {"name": "Goblet Squat", "sets": 4, "reps": "6-10"},
                {"name": "Glute Bridge", "sets": 4, "reps": "6-10"},
            ],
        },
        "feedback": {
            "workout_feeling": "solid day",
            "emoji": "😊",
            "emoji_label": "Good",
            "injury_areas": [],
        },
    }

    store = record_memory_daily_feedback(
        memory_store=default_memory_store(),
        feedback_date="2026-04-27",
        daily_entry=daily_entry,
        latest_feedback={"performance_notes": "solid day"},
    )

    exercise_feedback = store["exercise_feedback_records"]
    assert [item["exercise_name"] for item in exercise_feedback] == ["Goblet Squat", "Glute Bridge"]
    assert exercise_feedback[0]["sets"] == 4
    assert exercise_feedback[0]["reps"] == "6-10"
    assert exercise_feedback[0]["feeling_emoji"] == "😊"


def test_memory_context_exposes_four_memory_layers() -> None:
    store = {
        **default_memory_store(),
        "exercise_feedback_records": [
            {
                "date": "2026-04-27",
                "exercise_name": "Goblet Squat",
                "focus": "Lower Body (Legs + Glutes)",
                "status": "completed",
                "feeling_emoji": "😊",
            }
        ],
        "injury_events": [
            {"date": "2026-04-27", "area": "ankle", "status": "active", "expires_after_days": 7}
        ],
    }
    context = memory_context_for_planning(
        store,
        "2026-04-28",
        profile_inputs={"age": 26, "fitness_level": "beginner", "available_days": ["Monday"]},
        result={
            "thread_id": "thread-1",
            "user_profile": {"user_id": "demo-user", "weight_kg": 78.0},
            "current_plan": {"plan_id": "plan-1", "cycle_number": 2},
        },
        session_state={
            "thread_id": "thread-1",
            "conversation_summary": "User likes controlled lower body work.",
            "assistant_chat_messages": [{"role": "user", "content": "add sets"}],
        },
    )

    assert context["session_metadata"]["current_plan_id"] == "plan-1"
    assert context["structured_profile"]["training_profile"]["fitness_level"] == "beginner"
    assert context["structured_profile"]["learned_preferences"]["liked_exercises"] == ["Goblet Squat"]
    assert context["structured_profile"]["learned_preferences"]["active_injury_areas"] == ["ankle"]
    assert context["conversation_summary"] == "User likes controlled lower body work."
    assert context["sliding_window"] == [{"role": "user", "content": "add sets"}]


def test_compact_conversation_memory_keeps_sliding_window() -> None:
    messages = [{"role": "user", "content": f"message {index}"} for index in range(14)]

    summary, window = compact_conversation_memory(messages, limit=4)

    assert "Earlier chat summary" in summary
    assert [item["content"] for item in window] == ["message 10", "message 11", "message 12", "message 13"]


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
    assert add_sets_tool["tool_name"] == "adjust_workout_volume"
    assert add_sets_tool["arguments"]["set_adjustment"] == "increase"

    do_more_decision = coach.fallback_coordinator_decision("do more today", _sample_result())
    do_more_tool = coach.fallback_tool_call_from_decision(do_more_decision)
    assert do_more_tool["tool_name"] == "adjust_workout_volume"
    assert do_more_tool["arguments"]["intensity_adjustment"] == "higher"


def test_tool_selection_routes_explicit_add_exercise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coach, "call_model_json", lambda **_: (_ for _ in ()).throw(RuntimeError("no llm")))

    decision = coach.fallback_coordinator_decision("Can you add one more exercise?", _sample_result())
    tool_call = coach.fallback_tool_call_from_decision(decision)

    assert decision["planner_action"] == "adjust_workout_volume"
    assert tool_call["tool_name"] == "adjust_workout_volume"
    assert tool_call["arguments"]["intent"] == "adjust_workout_volume"
    assert tool_call["arguments"]["exercise_count_adjustment"] == "increase"


def test_advice_question_does_not_update_today_when_it_only_asks_recommendation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(coach, "call_model_json", lambda **_: (_ for _ in ()).throw(RuntimeError("no llm")))

    decision = coach.fallback_coordinator_decision(
        "do you recommend doing shoulders today?",
        _sample_result(),
    )
    tool_call = coach.fallback_tool_call_from_decision(decision)

    assert decision["route"] == "none"
    assert decision["normalized"]["intent"] == "answer_question"
    assert tool_call["tool_name"] == "no_action"


def test_strong_replacement_intent_overrides_coordinator_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        coach,
        "call_model_json",
        lambda **_: {
            "target_agent": "planner",
            "planner_action": "update_today_plan",
            "confidence": 0.98,
            "reason": "LLM drifted to a broader plan patch.",
        },
    )

    decision = coach.route_coach_message(
        "change some actions",
        {
            **_sample_result(),
            "current_plan": {
                **_sample_result()["current_plan"],
                "workout_sessions": [
                    {
                        **_today(_sample_result()),
                        "focus": "Upper Body (Shoulders)",
                        "exercises": [
                            {"name": "Seated Dumbbell Shoulder Press", "sets": 4, "reps": "6-10"},
                            {"name": "Dumbbell Lateral Raise", "sets": 4, "reps": "6-10"},
                        ],
                    }
                ],
            },
        },
        _session_state(),
    )

    assert decision["planner_action"] == "replace_exercise"


def test_strong_tool_fallback_overrides_tool_planner_drift() -> None:
    tool_call = coach.sanitize_tool_call(
        {
            "tool_name": "update_today_plan",
            "arguments": {"focus_category": "upper_chest_arms"},
            "confidence": 0.99,
        },
        {
            "tool_name": "replace_exercise",
            "arguments": {"request_type": "workout_change"},
            "source": "fallback",
        },
    )

    assert tool_call["tool_name"] == "replace_exercise"


def test_typo_exericises_is_treated_as_general_exercise_replacement() -> None:
    assert coach.chat_requests_general_exercise_replacement("change some exericises")


def test_normalized_replace_exercise_intent_routes_without_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            {
                "intent": "replace_exercise",
                "request_type": "workout_change",
                "scope": "today_only",
                "keep_current_focus": True,
                "summary": "The user wants different same-focus exercises.",
                "confidence": 0.9,
            },
            {
                "target_agent": "planner",
                "planner_action": "update_today_plan",
                "confidence": 0.95,
                "reason": "Coordinator drift.",
            },
        ]
    )
    monkeypatch.setattr(coach, "call_model_json", lambda **_: next(responses))

    decision = coach.route_coach_message("can we do different ones", _sample_result(), _session_state())

    assert decision["planner_action"] == "replace_exercise"
    assert decision["normalized"]["keep_current_focus"] is True


def test_update_today_focus_replaces_active_date_session_without_planner_drift() -> None:
    result = _sample_result()
    session_state = _session_state()

    summary = coach.execute_coach_tool_call(
        {
            "tool_name": "update_today_plan",
            "arguments": {
                "intent": "update_today_plan",
                "request_type": "workout_change",
                "scope": "today_only",
                "focus_category": "functional_conditioning",
            },
        },
        _tool_context(result, session_state, "do functional training today"),
    )

    updated = session_state["agent_result"]
    today = coach.select_today_session(
        updated["current_plan"]["workout_sessions"],
        session_state["active_date"],
    )

    assert "Functional" in summary
    assert today["scheduled_date"] == "2026-04-27"
    assert today["focus"] == "Functional (Conditioning)"
    assert today["exercises"]
    assert _future_back(updated)["focus"] == "Back Training"


def test_update_today_focus_uses_active_date_not_stale_result_current_date() -> None:
    result = _sample_result()
    result["current_date"] = "2026-04-24"
    session_state = _session_state()
    session_state["active_date"] = "2026-04-29"

    coach.execute_coach_tool_call(
        {
            "tool_name": "update_today_plan",
            "arguments": {
                "intent": "update_today_plan",
                "request_type": "workout_change",
                "scope": "today_only",
                "focus_category": "functional_conditioning",
            },
        },
        _tool_context(result, session_state, "do functional training today"),
    )

    updated = session_state["agent_result"]
    original_monday = coach.select_today_session(updated["current_plan"]["workout_sessions"], "2026-04-27")
    active_day = coach.select_today_session(updated["current_plan"]["workout_sessions"], "2026-04-29")

    assert original_monday["focus"] == "Lower Body (Legs + Glutes)"
    assert active_day["focus"] == "Functional (Conditioning)"


def test_cycle_plan_tool_changes_future_session_focus_only() -> None:
    result = _sample_result()
    session_state = _session_state()

    summary = coach.execute_coach_tool_call(
        {
            "tool_name": "update_cycle_plan",
            "arguments": {
                "intent": "update_cycle_plan",
                "target_day": "Wednesday",
                "cycle_operation": "replace_focus",
                "focus_category": "upper_shoulders",
            },
        },
        _tool_context(result, session_state, "change Wednesday to shoulders"),
    )

    updated = session_state["agent_result"]
    assert "Training cycle updated" in summary
    assert _today(updated)["focus"] == "Lower Body (Legs + Glutes)"
    assert _future_back(updated)["focus"] == "Upper Body (Shoulders)"
    assert _future_back(updated)["scheduled_date"] == "2026-04-29"
    assert _future_back(updated)["exercises"]


def test_cycle_plan_tool_replaces_future_exercises_same_focus_only() -> None:
    result = _sample_result()
    session_state = _session_state()

    summary = coach.execute_coach_tool_call(
        {
            "tool_name": "update_cycle_plan",
            "arguments": {
                "intent": "update_cycle_plan",
                "target_day": "Wednesday",
                "cycle_operation": "replace_exercise",
                "target": "some exercises",
            },
        },
        _tool_context(result, session_state, "change some exercises on Wednesday"),
    )

    updated = session_state["agent_result"]
    future = _future_back(updated)
    assert "same-focus" in summary
    assert future["focus"] == "Back Training"
    assert [exercise["sets"] for exercise in future["exercises"]] == [4, 4]
    assert all(
        "back_training" in exercise_tool.get_exercise_by_name(exercise["name"]).get("focus_tags", [])
        for exercise in future["exercises"]
    )


def test_cycle_plan_tool_cancels_future_session_only() -> None:
    result = _sample_result()
    session_state = _session_state()

    summary = coach.execute_coach_tool_call(
        {
            "tool_name": "update_cycle_plan",
            "arguments": {
                "intent": "update_cycle_plan",
                "target_day": "Wednesday",
                "cycle_operation": "cancel_session",
            },
        },
        _tool_context(result, session_state, "cancel Wednesday workout"),
    )

    updated = session_state["agent_result"]
    assert "cancelled" in summary
    assert not _today(updated).get("is_cancelled")
    assert _future_back(updated)["is_cancelled"] is True
    assert _future_back(updated)["focus"] == "Workout Cancelled"


def test_fallback_routes_non_today_day_to_cycle_plan_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coach, "call_model_json", lambda **_: (_ for _ in ()).throw(RuntimeError("no llm")))

    decision = coach.fallback_coordinator_decision("change Wednesday to shoulders", _sample_result())
    tool_call = coach.fallback_tool_call_from_decision(decision)

    assert decision["planner_action"] == "update_cycle_plan"
    assert tool_call["tool_name"] == "update_cycle_plan"
    assert tool_call["arguments"]["target_day"] == "Wednesday"
    assert tool_call["arguments"]["focus_category"] == "upper_shoulders"


def test_fallback_routes_future_cancel_to_cycle_not_today(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coach, "call_model_json", lambda **_: (_ for _ in ()).throw(RuntimeError("no llm")))

    decision = coach.fallback_coordinator_decision("cancel Wednesday workout", _sample_result())
    tool_call = coach.fallback_tool_call_from_decision(decision)

    assert decision["planner_action"] == "update_cycle_plan"
    assert tool_call["tool_name"] == "update_cycle_plan"
    assert tool_call["arguments"]["cycle_operation"] == "cancel_session"
    assert tool_call["arguments"]["cancel_today"] is False


def test_cycle_target_parser_accepts_dot_month_day() -> None:
    target_date, target_day, target_session = coach.cycle_target_from_text(
        "change 5.12's plan",
        {"current_date": "2026-05-06"},
    )

    assert target_date == "2026-05-12"
    assert target_day == ""
    assert target_session == ""


def test_back_discomfort_protects_future_back_without_unrelated_today_cancel() -> None:
    result = _sample_result()
    result["current_plan"]["workout_sessions"][0]["exercises"] = [
        {"name": "Leg Press", "sets": 4, "reps": "6-10", "notes": ""},
        {"name": "Reverse Lunge", "sets": 4, "reps": "6-10", "notes": ""},
    ]
    session_state = _session_state()

    decision = coach.fallback_coordinator_decision("my back feels uncomfortable", result)
    tool_call = coach.fallback_tool_call_from_decision(decision)
    summary = coach.execute_coach_tool_call(tool_call, _tool_context(result, session_state, "my back feels uncomfortable"))

    updated = session_state["agent_result"]
    assert tool_call["tool_name"] == "cancel_workout"
    assert tool_call["arguments"]["recovery_signal_only"] is True
    assert tool_call["arguments"]["cancel_today"] is False
    assert "left unchanged" in summary
    assert _today(updated).get("is_cancelled") is not True
    assert _future_back(updated)["is_cancelled"] is True
    assert _future_back(updated)["focus"] == "Recovery (injury protection)"


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


def test_replace_exercise_tool_keeps_current_focus_with_vague_request() -> None:
    result = {
        **_sample_result(),
        "current_plan": {
            **_sample_result()["current_plan"],
            "workout_sessions": [
                {
                    **_today(_sample_result()),
                    "focus": "Upper Body (Shoulders)",
                    "exercises": [
                        {"name": "Seated Dumbbell Shoulder Press", "sets": 4, "reps": "6-10"},
                        {"name": "Dumbbell Lateral Raise", "sets": 4, "reps": "6-10"},
                    ],
                }
            ],
        },
    }
    session_state = _session_state()

    summary = coach.execute_coach_tool_call(
        {
            "tool_name": "replace_exercise",
            "arguments": {
                "intent": "replace_exercise",
                "keep_current_focus": True,
                "target": "some exercises",
            },
        },
        _tool_context(result, session_state, "can we do different ones"),
    )

    today = _today(session_state["agent_result"])
    assert "same-focus" in summary
    assert today["focus"] == "Upper Body (Shoulders)"
    assert all(
        "upper_shoulders" in exercise_tool.get_exercise_by_name(exercise["name"]).get("focus_tags", [])
        for exercise in today["exercises"]
    )


def test_same_focus_filter_rejects_wrong_focus_candidates() -> None:
    candidates = [
        {"name": "Leg Press", "focus_tags": ["lower_legs_glutes"]},
        {"name": "Face Pull", "focus_tags": ["upper_shoulders"]},
    ]

    filtered = coach.same_focus_replacement_candidates(candidates, "upper_shoulders")

    assert [candidate["name"] for candidate in filtered] == ["Face Pull"]


def test_rag_rerank_rewards_replacement_group_alias_and_rejects_wrong_focus() -> None:
    source = {
        "name": "Lat Pulldown",
        "replacement_group": "lat_pull",
        "movement_pattern": "vertical_pull",
        "primary_muscles": ["lats"],
        "focus_tags": ["back_training"],
    }
    good_candidate = {
        "name": "Assisted Pull-Up",
        "replacement_group": "vertical_pull",
        "movement_pattern": "vertical_pull",
        "primary_muscles": ["lats"],
        "focus_tags": ["back_training"],
        "difficulty": "beginner",
        "media_url": "https://example.com/pullup.mp4",
    }
    wrong_focus_candidate = {
        "name": "Leg Press",
        "replacement_group": "squat_pattern",
        "movement_pattern": "squat_pattern",
        "primary_muscles": ["quads"],
        "focus_tags": ["lower_legs_glutes"],
        "difficulty": "beginner",
    }

    assert (
        replacement_candidate_score(
            candidate=good_candidate,
            source=source,
            focus="back_training",
            level="beginner",
        )
        >= 140
    )
    assert (
        replacement_candidate_score(
            candidate=wrong_focus_candidate,
            source=source,
            focus="back_training",
            level="beginner",
        )
        < 0
    )


def test_rag_rerank_rewards_cached_video_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    source = {
        "name": "Dumbbell Lateral Raise",
        "replacement_group": "lateral_raise",
        "movement_pattern": "shoulder_abduction",
        "primary_muscles": ["side delts"],
        "focus_tags": ["upper_shoulders"],
    }
    candidate = {
        "name": "Cable Lateral Raise",
        "replacement_group": "lateral_raise",
        "movement_pattern": "shoulder_abduction",
        "primary_muscles": ["side delts"],
        "focus_tags": ["upper_shoulders"],
        "difficulty": "beginner",
    }

    monkeypatch.setattr("agent.tools.rag_tool.get_cached_video_resource", lambda name: None)
    without_video = replacement_candidate_score(
        candidate=candidate,
        source=source,
        focus="upper_shoulders",
        level="beginner",
    )
    monkeypatch.setattr(
        "agent.tools.rag_tool.get_cached_video_resource",
        lambda name: {
            "url": "https://www.youtube.com/watch?v=cached",
            "source": "youtube_api",
            "provider": "youtube",
        },
    )
    with_cached_video = replacement_candidate_score(
        candidate=candidate,
        source=source,
        focus="upper_shoulders",
        level="beginner",
    )

    assert with_cached_video == without_video + 10


def test_video_resources_use_youtube_api_even_when_imported_media_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        exercise_tool,
        "load_all_exercise_db",
        lambda: [
            {
                "name": "Imported Shoulder Raise",
                "media_url": "https://example.com/shoulder-raise.mp4",
                "source": "wger",
            },
            {"name": "No Local Media"},
        ],
    )
    monkeypatch.setattr(
        "agent.services.video_backfill.search_youtube_video",
        lambda exercise_name: {
            "title": f"{exercise_name} tutorial",
            "url": "https://www.youtube.com/watch?v=test",
            "source": "youtube_api",
        },
    )
    monkeypatch.setattr("agent.services.video_backfill.get_cached_video_resource", lambda exercise_name: None)
    monkeypatch.setattr(
        "agent.services.video_backfill.save_cached_video_resource",
        lambda exercise_name, resource: {
            "exercise_name": resource.get("exercise_name") or exercise_name,
            "title": resource.get("title", ""),
            "url": resource.get("url", ""),
            "source": resource.get("source", ""),
        },
    )

    resources = exercise_tool.build_video_resources(["Imported Shoulder Raise", "No Local Media"])

    assert resources[0]["url"] == "https://www.youtube.com/watch?v=test"
    assert resources[0]["source"] == "youtube_api"
    assert resources[1]["url"] == "https://www.youtube.com/watch?v=test"


def test_exercise_plan_payload_includes_teaching_fields() -> None:
    payload = exercise_tool.build_exercise_plan_payload(
        {
            "name": "Seated Dumbbell Shoulder Press",
            "target_muscle": ["shoulders", "front delts"],
            "primary_muscles": ["shoulders"],
            "secondary_muscles": ["triceps"],
            "equipment": ["dumbbell", "bench"],
            "movement_pattern": "vertical_push",
            "difficulty": "beginner",
            "notes": "Keep the ribs down and press without shrugging.",
            "source": "wger",
        },
        sets=4,
        reps="6-10",
        focus="upper_shoulders",
    )

    assert payload["primary_muscles"] == ["shoulders"]
    assert payload["coaching_cue"].startswith("Keep the ribs down")
    assert "upper_shoulders" in payload["why_this_exercise"]
    assert payload["common_mistake"]
    assert payload["regression"]
    assert payload["progression"]
    assert payload["knowledge_source"] == "wger"


def test_adjust_sets_changes_sets_only_and_keeps_exercise_count() -> None:
    result = _sample_result()
    session_state = _session_state()

    summary = coach.execute_coach_tool_call(
        {"tool_name": "adjust_workout_volume", "arguments": {"set_adjustment": "increase"}},
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
        {"tool_name": "adjust_workout_volume", "arguments": {"set_adjustment": "increase"}},
        _tool_context(result, session_state, "add sets"),
    )
    after_sets = session_state["agent_result"]

    coach.execute_coach_tool_call(
        {"tool_name": "adjust_workout_volume", "arguments": {"exercise_count_adjustment": "increase"}},
        _tool_context(after_sets, session_state, "add one more exercise"),
    )

    today = _today(session_state["agent_result"])
    assert len(today["exercises"]) >= 2
    assert {exercise["sets"] for exercise in today["exercises"]} == {5}


def test_add_one_more_exercise_updates_today_plan_and_keeps_focus_policy() -> None:
    result = _sample_result()
    result["current_date"] = "2026-04-28"
    result["current_plan"]["workout_sessions"] = [
        {
            "day": "Tuesday",
            "scheduled_date": "2026-04-28",
            "cycle_number": 1,
            "focus": "Upper Body (Chest + Arms)",
            "duration_minutes": 60,
            "warmup": ["band pull-aparts"],
            "exercises": [
                {"name": "Incline Push-Up", "sets": 4, "reps": "6-10", "notes": ""},
                {"name": "Push-Up", "sets": 4, "reps": "6-10", "notes": ""},
            ],
            "cooldown": ["stretch"],
            "safety_notes": ["Move well."],
        }
    ]
    session_state = _session_state()
    session_state["active_date"] = "2026-04-28"

    summary = coach.execute_coach_tool_call(
        {"tool_name": "adjust_workout_volume", "arguments": {"exercise_count_adjustment": "increase"}},
        _tool_context(result, session_state, "Can you add one more exercise?"),
    )

    today = _today(session_state["agent_result"])
    assert "one more same-focus exercise" in summary
    assert today["focus"] == "Upper Body (Chest + Arms)"
    assert len(today["exercises"]) == 3
    assert [exercise["sets"] for exercise in today["exercises"]] == [4, 4, 4]
    assert [exercise["reps"] for exercise in today["exercises"]] == ["6-10", "6-10", "6-10"]
    assert today["exercises"][2]["name"] not in {"Incline Push-Up", "Push-Up"}
    video_titles = [
        resource["title"]
        for resource in session_state["agent_result"].get("youtube_resources", [])
    ]
    assert any(today["exercises"][2]["name"] in title for title in video_titles)


def test_reduce_one_exercise_uses_volume_tool_and_never_below_two() -> None:
    result = _sample_result()
    _today(result)["exercises"].append(
        {"name": "Leg Press", "sets": 4, "reps": "6-10", "notes": ""}
    )
    session_state = _session_state()

    decision = coach.fallback_coordinator_decision("remove one exercise today", result)
    tool_call = coach.fallback_tool_call_from_decision(decision)

    assert decision["planner_action"] == "adjust_workout_volume"
    assert tool_call["tool_name"] == "adjust_workout_volume"
    assert tool_call["arguments"]["exercise_count_adjustment"] == "decrease"

    summary = coach.execute_coach_tool_call(tool_call, _tool_context(result, session_state, "remove one exercise today"))
    today = _today(session_state["agent_result"])
    assert "exercise count was reduced" in summary
    assert len(today["exercises"]) == 2
    assert today["focus"] == "Lower Body (Legs + Glutes)"
    assert [exercise["sets"] for exercise in today["exercises"]] == [4, 4]

    summary = coach.execute_coach_tool_call(tool_call, _tool_context(session_state["agent_result"], session_state))
    today = _today(session_state["agent_result"])
    assert "boundary" in summary
    assert len(today["exercises"]) == 2


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
        {"tool_name": "adjust_workout_volume", "arguments": {"set_adjustment": "increase"}},
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


def test_ai_coach_regression_sequence_preserves_state_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _sample_result()
    session_state = _session_state()

    coach.execute_coach_tool_call(
        {"tool_name": "adjust_workout_volume", "arguments": {"set_adjustment": "increase"}},
        _tool_context(result, session_state, "add sets"),
    )
    after_sets = session_state["agent_result"]

    coach.execute_coach_tool_call(
        {"tool_name": "replace_exercise", "arguments": {"intent": "replace_exercise", "target": "some exercises"}},
        _tool_context(after_sets, session_state, "change some exercises"),
    )
    after_replacement = session_state["agent_result"]
    replacement_today = _today(after_replacement)
    assert replacement_today["focus"] == "Lower Body (Legs + Glutes)"
    assert {exercise["sets"] for exercise in replacement_today["exercises"]} == {5}
    assert all(
        "lower_legs_glutes" in exercise_tool.get_exercise_by_name(exercise["name"]).get("focus_tags", [])
        for exercise in replacement_today["exercises"]
    )

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
    workout_before_food_change = deepcopy(after_replacement["current_plan"]["workout_sessions"])

    coach.execute_coach_tool_call(
        {"tool_name": "replace_food", "arguments": {"temporary_food_avoidances": ["broccoli"]}},
        _tool_context(after_replacement, session_state, "I don't want broccoli"),
    )
    after_food = session_state["agent_result"]

    assert after_food["current_plan"]["workout_sessions"] == workout_before_food_change
    assert "Spinach" in [meal["food_name"] for meal in after_food["current_plan"]["meal_suggestions"]]


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
