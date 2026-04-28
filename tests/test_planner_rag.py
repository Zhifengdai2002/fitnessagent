from __future__ import annotations

from agent.nodes.planner import _build_session_blueprint, _retrieve_plan_exercise_candidates


def test_planner_retrieves_rag_candidates_for_session() -> None:
    candidates = _retrieve_plan_exercise_candidates(
        focus_key="upper_shoulders",
        target_muscles=["shoulders", "side delts", "rear delts"],
        movement_type=None,
        fitness_level="beginner",
        training_goal="weight_loss",
        equipment_access=["bodyweight", "dumbbell", "bench"],
        excluded_conditions=[],
        excluded_exercises=["Dumbbell Lateral Raise"],
        limit=4,
    )

    assert candidates
    assert all(candidate["name"] != "Dumbbell Lateral Raise" for candidate in candidates)
    assert any("upper_shoulders" in candidate.get("focus_tags", []) for candidate in candidates)


def test_session_blueprint_uses_rag_candidate_pool() -> None:
    blueprint = _build_session_blueprint(
        day="Monday",
        scheduled_date="2026-04-27",
        cycle_number=1,
        cycle_session_index=1,
        focus="Upper Body (Shoulders)",
        duration_minutes=60,
        fitness_level="beginner",
        training_goal="weight_loss",
        equipment_access=["bodyweight", "dumbbell", "bench"],
        excluded_conditions=[],
        excluded_exercises=[],
    )

    assert blueprint["focus_key"] == "upper_shoulders"
    assert blueprint["target_exercise_count"] == 2
    assert len(blueprint["candidate_exercises"]) >= 2
    assert any(
        "upper_shoulders" in candidate.get("focus_tags", [])
        for candidate in blueprint["candidate_exercises"]
    )


def test_session_blueprint_finalization_adds_exercise_teaching_metadata() -> None:
    from agent.nodes.planner import _finalize_workout_session

    blueprint = _build_session_blueprint(
        day="Monday",
        scheduled_date="2026-04-27",
        cycle_number=1,
        cycle_session_index=1,
        focus="Upper Body (Shoulders)",
        duration_minutes=60,
        fitness_level="beginner",
        training_goal="weight_loss",
        equipment_access=["bodyweight", "dumbbell", "bench"],
        excluded_conditions=[],
        excluded_exercises=[],
    )
    session = _finalize_workout_session(
        blueprint=blueprint,
        session_payload={},
        fitness_level="beginner",
        training_goal="weight_loss",
    )

    assert session["exercises"]
    first = session["exercises"][0]
    assert first["coaching_cue"]
    assert first["why_this_exercise"]
    assert first["common_mistake"]
