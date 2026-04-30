from __future__ import annotations

from agent.rag.wger_importer import normalize_wger_exercises


def test_normalize_wger_exerciseinfo_record() -> None:
    normalized = normalize_wger_exercises(
        [
            {
                "id": 123,
                "translations": [
                    {
                        "name": "Seated Shoulder Press",
                        "description": "<p>Press the dumbbells overhead with control.</p>",
                    }
                ],
                "category": {"name": "Shoulders"},
                "muscles": [{"name_en": "Shoulders"}],
                "muscles_secondary": [{"name_en": "Triceps"}],
                "equipment": [{"name": "Dumbbell"}, {"name": "Bench"}],
                "images": [{"image": "https://example.com/shoulder-press.jpg"}],
            }
        ]
    )

    assert len(normalized) == 1
    exercise = normalized[0]
    assert exercise["id"] == "wger_123"
    assert exercise["name"] == "Seated Shoulder Press"
    assert exercise["source"] == "wger"
    assert exercise["movement_pattern"] == "vertical_push"
    assert exercise["replacement_group"] == "shoulder_press"
    assert "upper_shoulders" in exercise["focus_tags"]
    assert exercise["primary_muscles"] == ["Shoulders"]
    assert "Press the dumbbells overhead" in exercise["notes"]


def test_normalize_wger_prefers_english_translation() -> None:
    normalized = normalize_wger_exercises(
        [
            {
                "id": 805,
                "translations": [
                    {"language": 4, "name": "Empuje de tríceps en cable", "description": "ES"},
                    {"language": 2, "name": "Tricep Pushdown on Cable", "description": "EN"},
                ],
                "category": {"name": "Arms"},
                "muscles": [{"name_en": "Triceps"}],
                "equipment": [{"name": "Cable"}],
            }
        ]
    )

    assert normalized[0]["name"] == "Tricep Pushdown on Cable"
    assert normalized[0]["notes"] == "EN"


def test_normalize_wger_infers_obvious_equipment_when_missing() -> None:
    normalized = normalize_wger_exercises(
        [
            {
                "id": 901,
                "translations": [
                    {"language": 2, "name": "Cable Fly Upper Chest", "description": "Use a controlled arc."},
                ],
                "category": {"name": "Chest"},
                "muscles": [{"name_en": "Chest"}],
                "equipment": [],
            },
            {
                "id": 902,
                "translations": [
                    {"language": 2, "name": "Chest Press", "description": "Press with control."},
                ],
                "category": {"name": "Chest"},
                "muscles": [{"name_en": "Chest"}],
                "equipment": [],
            },
        ]
    )

    by_name = {exercise["name"]: exercise for exercise in normalized}
    assert by_name["Cable Fly Upper Chest"]["equipment"] == ["cable_machine"]
    assert by_name["Chest Press"]["equipment"] == ["machine"]
