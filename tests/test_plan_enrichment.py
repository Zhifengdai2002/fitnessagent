from __future__ import annotations

from agent.services.plan_enrichment import hydrate_agent_result_for_display


def test_hydrate_agent_result_backfills_teaching_fields_and_videos() -> None:
    result = {
        "current_plan": {
            "workout_sessions": [
                {
                    "focus": "Back Training",
                    "exercises": [
                        {
                            "name": "Lat Pulldown",
                            "sets": 4,
                            "reps": "6-10",
                        }
                    ],
                }
            ]
        },
        "youtube_resources": [],
    }

    hydrated = hydrate_agent_result_for_display(result)
    exercise = hydrated["current_plan"]["workout_sessions"][0]["exercises"][0]

    assert exercise["coaching_cue"]
    assert exercise["why_this_exercise"]
    assert exercise["common_mistake"]
    assert hydrated["youtube_resources"]
    assert hydrated["youtube_resources"][0]["exercise_name"] == "Lat Pulldown"


def test_hydrate_agent_result_preserves_existing_exercise_values() -> None:
    result = {
        "current_plan": {
            "workout_sessions": [
                {
                    "focus": "Back Training",
                    "exercises": [
                        {
                            "name": "Lat Pulldown",
                            "sets": 5,
                            "reps": "8-12",
                            "notes": "Keep the reps smooth.",
                        }
                    ],
                }
            ]
        }
    }

    hydrated = hydrate_agent_result_for_display(result)
    exercise = hydrated["current_plan"]["workout_sessions"][0]["exercises"][0]

    assert exercise["sets"] == 5
    assert exercise["reps"] == "8-12"
    assert exercise["notes"] == "Keep the reps smooth."
