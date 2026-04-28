"""Build retrievable documents from local fitness knowledge sources."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

EXERCISE_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "exercise_db.json"
WGER_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "external" / "wger_exercises.json"


def build_exercise_documents() -> list[dict[str, Any]]:
    """Return normalized exercise documents for local retrieval."""

    documents: list[dict[str, Any]] = []
    for exercise in load_exercise_documents_source():
        name = str(exercise.get("name", "")).strip()
        if not name:
            continue
        primary = _list_text(exercise.get("primary_muscles") or exercise.get("target_muscle"))
        secondary = _list_text(exercise.get("secondary_muscles"))
        focus = _list_text(exercise.get("focus_tags"))
        equipment = _list_text(exercise.get("equipment"))
        difficulty = str(exercise.get("difficulty", "")).strip()
        movement_pattern = str(exercise.get("movement_pattern") or exercise.get("movement_type") or "").strip()
        replacement_group = str(exercise.get("replacement_group", "")).strip()
        notes = str(exercise.get("notes", "")).strip()
        contraindications = _list_text(exercise.get("contraindications"))
        goals = _list_text(exercise.get("training_goal_tags"))

        text = "\n".join(
            piece
            for piece in [
                f"Exercise: {name}",
                f"Difficulty: {difficulty}" if difficulty else "",
                f"Movement pattern: {movement_pattern}" if movement_pattern else "",
                f"Replacement group: {replacement_group}" if replacement_group else "",
                f"Primary muscles: {primary}" if primary else "",
                f"Secondary muscles: {secondary}" if secondary else "",
                f"Focus tags: {focus}" if focus else "",
                f"Equipment: {equipment}" if equipment else "",
                f"Training goals: {goals}" if goals else "",
                f"Contraindications: {contraindications}" if contraindications else "",
                f"Coaching notes: {notes}" if notes else "",
            ]
            if piece
        )
        documents.append(
            {
                "id": f"exercise:{exercise.get('id') or _slug(name)}",
                "type": "exercise",
                "title": name,
                "text": text,
                "metadata": {
                    "source": "local_exercise_db",
                    "exercise_id": exercise.get("id", ""),
                    "name": name,
                    "difficulty": difficulty,
                    "movement_pattern": movement_pattern,
                    "replacement_group": replacement_group,
                    "primary_muscles": exercise.get("primary_muscles") or exercise.get("target_muscle") or [],
                    "secondary_muscles": exercise.get("secondary_muscles") or [],
                    "target_muscle": exercise.get("target_muscle") or [],
                    "focus_tags": exercise.get("focus_tags") or [],
                    "equipment": exercise.get("equipment") or [],
                    "contraindications": exercise.get("contraindications") or [],
                    "training_goal_tags": exercise.get("training_goal_tags") or [],
                    "youtube_url": exercise.get("youtube_url", ""),
                    "notes": notes,
                },
                "raw": exercise,
            }
        )
    return documents


@lru_cache(maxsize=1)
def load_exercise_documents_source() -> list[dict[str, Any]]:
    exercises: list[dict[str, Any]] = []
    with EXERCISE_DB_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        exercises.extend(item for item in payload if isinstance(item, dict))
    if WGER_CACHE_PATH.exists():
        try:
            wger_payload = json.loads(WGER_CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            wger_payload = []
        if isinstance(wger_payload, list):
            exercises.extend(item for item in wger_payload if isinstance(item, dict))
    return _dedupe_exercises(exercises)


def _dedupe_exercises(exercises: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for exercise in exercises:
        name = str(exercise.get("name", "")).strip()
        if not name:
            continue
        name_key = _slug(name)
        if name_key in seen_names:
            continue
        seen_names.add(name_key)
        deduped.append(exercise)
    return deduped


def _list_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return ", ".join(str(item).strip() for item in value if str(item).strip())


def _slug(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")
