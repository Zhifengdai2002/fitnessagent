from __future__ import annotations

from agent.rag.documents import (
    LOCAL_EXERCISE_FALLBACK_LIMIT,
    build_exercise_documents,
    load_local_exercise_fallback_source,
)
from agent.nodes.planner import _build_session_blueprint, _merge_plan_exercise_candidates, _retrieve_plan_exercise_candidates
from agent.rag.retriever import retrieve_exercises
from agent.tools import exercise_tool


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


def test_exercise_rag_uses_learned_exercise_preferences() -> None:
    base = retrieve_exercises(
        query="beginner upper body chest arms horizontal push",
        focus="upper_chest_arms",
        level="beginner",
        limit=8,
    )
    preferred = retrieve_exercises(
        query="beginner upper body chest arms horizontal push",
        focus="upper_chest_arms",
        level="beginner",
        learned_preferences={
            "liked_exercises": ["Machine Chest Press"],
            "difficult_exercises": ["Push-Up", "Incline Push-Up"],
        },
        limit=8,
    )

    assert any(item["name"] == "Machine Chest Press" for item in preferred)
    preferred_names = [item["name"] for item in preferred]
    if any(item["name"] == "Machine Chest Press" for item in base):
        assert preferred_names.index("Machine Chest Press") <= [item["name"] for item in base].index("Machine Chest Press")
    assert preferred_names.index("Machine Chest Press") < len(preferred_names)


def test_exercise_rag_keeps_local_candidates_as_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.rag.retriever._search_exercise_documents",
        lambda query, limit: [
            {
                "score": 0.99,
                "document": {
                    "metadata": {
                        "name": "Local Shoulder Raise",
                        "source": "local",
                        "focus_tags": ["upper_shoulders"],
                        "difficulty": "beginner",
                    },
                    "raw": {
                        "name": "Local Shoulder Raise",
                        "source": "local",
                        "focus_tags": ["upper_shoulders"],
                        "difficulty": "beginner",
                    },
                },
            },
            {
                "score": 0.5,
                "document": {
                    "metadata": {
                        "name": "Wger Shoulder Press",
                        "source": "wger",
                        "focus_tags": ["upper_shoulders"],
                        "difficulty": "beginner",
                    },
                    "raw": {
                        "name": "Wger Shoulder Press",
                        "source": "wger",
                        "focus_tags": ["upper_shoulders"],
                        "difficulty": "beginner",
                    },
                },
            },
        ],
    )

    candidates = retrieve_exercises(
        query="beginner shoulder exercise",
        focus="upper_shoulders",
        level="beginner",
        limit=2,
    )

    assert [candidate["name"] for candidate in candidates] == [
        "Wger Shoulder Press",
        "Local Shoulder Raise",
    ]
    assert candidates[1]["source"] == "local_fallback"


def test_find_exercises_uses_local_only_after_primary_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        exercise_tool,
        "load_all_exercise_db",
        lambda: [
            {
                "id": "local_beginner_squat",
                "name": "Local Beginner Squat",
                "target_muscle": ["quads"],
                "focus_tags": ["lower_legs_glutes"],
                "difficulty": "beginner",
                "training_goal_tags": ["weight_loss"],
                "source": "local",
            },
            {
                "id": "wger_leg_press",
                "name": "Wger Leg Press",
                "target_muscle": ["quads"],
                "focus_tags": ["lower_legs_glutes"],
                "difficulty": "advanced",
                "training_goal_tags": ["strength"],
                "source": "wger",
            },
        ],
    )

    candidates = exercise_tool.find_exercises(
        target_muscles=["quads"],
        focus_tags=["lower_legs_glutes"],
        difficulty="beginner",
        training_goal="weight_loss",
        limit=1,
    )

    assert [candidate["name"] for candidate in candidates] == ["Wger Leg Press"]


def test_exercise_documents_keep_local_json_as_small_fallback_only() -> None:
    fallback = load_local_exercise_fallback_source()
    documents = build_exercise_documents()
    local_documents = [doc for doc in documents if doc["metadata"]["source"] == "local_fallback"]
    primary_sources = {doc["metadata"]["source"] for doc in documents if doc["metadata"]["source"] != "local_fallback"}

    assert len(fallback) == LOCAL_EXERCISE_FALLBACK_LIMIT
    assert len(local_documents) <= LOCAL_EXERCISE_FALLBACK_LIMIT
    assert {"curated_rag", "wger"}.issubset(primary_sources)
    assert all(item["source"] == "local_fallback" for item in fallback)


def test_legacy_local_exercises_are_exact_lookup_only() -> None:
    candidates = exercise_tool.load_all_exercise_db()
    candidate_names = {candidate["name"] for candidate in candidates}

    assert "Barbell Bench Press" not in candidate_names
    legacy = exercise_tool.get_exercise_by_name("Barbell Bench Press")
    assert legacy
    assert legacy["source"] == "local_fallback"


def test_planner_merge_dedupes_generic_exercise_suffix() -> None:
    merged = _merge_plan_exercise_candidates(
        primary=[
            {
                "id": "wger_machine_chest_press_exercise",
                "name": "Machine Chest Press Exercise",
                "source": "wger",
            }
        ],
        fallback=[
            {
                "id": "rag_machine_chest_press",
                "name": "Machine Chest Press",
                "source": "curated_rag",
            },
            {
                "id": "rag_incline_push_up",
                "name": "Incline Push-Up",
                "source": "curated_rag",
            },
        ],
        excluded_exercises=[],
        limit=2,
    )

    assert [item["name"] for item in merged] == ["Machine Chest Press Exercise", "Incline Push-Up"]
