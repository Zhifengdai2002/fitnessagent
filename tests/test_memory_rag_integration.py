from __future__ import annotations

from agent.rag.retriever import retrieve_exercises, retrieve_foods
from agent.services.memory import default_memory_store, memory_context_for_planning
from agent.services.preference_scoring import learned_preferences_from_context


def test_memory_history_drives_rag_preferences() -> None:
    memory_store = {
        **default_memory_store(),
        "exercise_feedback_records": [
            {
                "date": "2026-04-24",
                "exercise_name": "Machine Chest Press",
                "focus": "upper_chest_arms",
                "status": "completed",
                "feeling_emoji": "😊",
            },
            {
                "date": "2026-04-25",
                "exercise_name": "Push-Up",
                "focus": "upper_chest_arms",
                "status": "completed",
                "feeling_emoji": "😫",
                "feeling": "too hard on wrists",
            },
        ],
        "food_preferences": [
            {
                "date": "2026-04-25",
                "food": "Broccoli",
                "scope": "avoid",
                "source": "ai_coach",
            }
        ],
        "injury_events": [
            {
                "date": "2026-04-25",
                "area": "wrist",
                "status": "active",
                "expires_after_days": 14,
            }
        ],
    }

    memory_context = memory_context_for_planning(
        memory_store,
        "2026-04-28",
        profile_inputs={"fitness_level": "beginner"},
    )
    learned_preferences = learned_preferences_from_context(memory_context)

    assert learned_preferences["liked_exercises"] == ["Machine Chest Press"]
    assert learned_preferences["difficult_exercises"] == ["Push-Up"]
    assert learned_preferences["avoided_foods"] == ["Broccoli"]
    assert learned_preferences["active_injury_areas"] == ["wrist"]

    exercises = retrieve_exercises(
        query="beginner upper body chest arms horizontal push",
        focus="upper_chest_arms",
        level="beginner",
        learned_preferences=learned_preferences,
        limit=8,
    )
    exercise_names = [exercise["name"] for exercise in exercises]

    assert "Machine Chest Press" in exercise_names
    if "Push-Up" in exercise_names:
        assert exercise_names.index("Machine Chest Press") < exercise_names.index("Push-Up")

    foods = retrieve_foods(
        query="vegetable high fiber dinner side",
        category="vegetable",
        learned_preferences=learned_preferences,
        limit=8,
    )
    food_names = {food["name"] for food in foods}

    assert "Broccoli" not in food_names
    assert foods
