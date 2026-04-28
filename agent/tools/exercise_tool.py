"""Exercise lookup helpers backed by the local exercise database."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote_plus


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "exercise_db.json"
EXTERNAL_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "external" / "wger_exercises.json"


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_many(values: Iterable[str] | None) -> set[str]:
    if not values:
        return set()
    return {_normalize(value) for value in values}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


@lru_cache(maxsize=1)
def load_exercise_db() -> list[dict[str, Any]]:
    """Load the local exercise database once per process."""

    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def load_external_exercise_db() -> list[dict[str, Any]]:
    """Load optional imported exercise sources."""

    if not EXTERNAL_DATA_PATH.exists():
        return []
    with EXTERNAL_DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def load_all_exercise_db() -> list[dict[str, Any]]:
    """Return local curated exercises plus imported exercises, deduped by name."""

    exercises: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for exercise in [*load_exercise_db(), *load_external_exercise_db()]:
        name = str(exercise.get("name", "")).strip()
        if not name:
            continue
        normalized_name = _normalize(name)
        if normalized_name in seen_names:
            continue
        seen_names.add(normalized_name)
        exercises.append(exercise)
    return exercises


def get_exercise_by_id(exercise_id: str) -> dict[str, Any] | None:
    """Return one exercise by id."""

    target_id = _normalize(exercise_id)
    for exercise in load_all_exercise_db():
        if _normalize(exercise["id"]) == target_id:
            return exercise
    return None


def get_exercise_by_name(name: str) -> dict[str, Any] | None:
    """Return one exercise by display name."""

    target_name = _normalize(name)
    for exercise in load_all_exercise_db():
        if _normalize(exercise["name"]) == target_name:
            return exercise
    return None


def find_exercises(
    *,
    target_muscles: Iterable[str] | None = None,
    focus_tags: Iterable[str] | None = None,
    movement_type: str | None = None,
    difficulty: str | None = None,
    available_equipment: Iterable[str] | None = None,
    training_goal: str | None = None,
    excluded_conditions: Iterable[str] | None = None,
    recommended_for: Iterable[str] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return safe focus matches, ranking level and goal as preferences."""

    target_muscle_set = _normalize_many(target_muscles)
    focus_tag_set = _normalize_many(focus_tags)
    excluded_condition_set = _normalize_many(excluded_conditions)
    recommended_for_set = _normalize_many(recommended_for)
    movement_type_normalized = _normalize(movement_type) if movement_type else None
    difficulty_normalized = _normalize(difficulty) if difficulty else None
    training_goal_normalized = _normalize(training_goal) if training_goal else None

    scored_matches: list[tuple[int, int, dict[str, Any]]] = []
    for index, exercise in enumerate(load_all_exercise_db()):
        exercise_muscles = _normalize_many(exercise.get("target_muscle", []))
        exercise_focus_tags = _normalize_many(exercise.get("focus_tags", []))
        contraindications = _normalize_many(exercise.get("contraindications", []))
        recommendation_tags = _normalize_many(exercise.get("recommended_for", []))
        goal_tags = _normalize_many(exercise.get("training_goal_tags", []))

        if target_muscle_set and not target_muscle_set.intersection(exercise_muscles):
            continue
        if focus_tag_set and exercise_focus_tags and not focus_tag_set.intersection(exercise_focus_tags):
            continue
        if movement_type_normalized and _normalize(exercise.get("movement_type", "")) != movement_type_normalized:
            continue
        if excluded_condition_set and excluded_condition_set.intersection(contraindications):
            continue

        score = _preference_score(
            exercise_difficulty=_normalize(exercise.get("difficulty", "")),
            requested_difficulty=difficulty_normalized,
            goal_tags=goal_tags,
            requested_goal=training_goal_normalized,
            recommendation_tags=recommendation_tags,
            requested_recommendations=recommended_for_set,
        )
        scored_matches.append((score, index, exercise))

    scored_matches.sort(key=lambda item: (-item[0], item[1]))
    return [exercise for _, _, exercise in scored_matches[:limit]]


def build_exercise_plan_payload(
    exercise: dict[str, Any],
    *,
    sets: int,
    reps: str,
    notes: str = "",
    focus: str = "",
) -> dict[str, Any]:
    """Build a user-facing exercise item with teaching metadata."""

    teaching = build_exercise_teaching_fields(exercise, focus=focus)
    base_notes = notes.strip() or str(exercise.get("notes", "")).strip()
    target_muscles = _as_list(exercise.get("target_muscle")) or _as_list(exercise.get("primary_muscles"))
    return {
        "name": str(exercise.get("name", "")).strip(),
        "target_muscle": ", ".join(target_muscles),
        "sets": sets,
        "reps": reps,
        "equipment": ", ".join(_as_list(exercise.get("equipment"))),
        "notes": base_notes,
        **teaching,
    }


def build_exercise_teaching_fields(exercise: dict[str, Any], *, focus: str = "") -> dict[str, Any]:
    """Return concise RAG-style teaching fields for one exercise."""

    name = str(exercise.get("name", "")).strip()
    primary = _as_list(exercise.get("primary_muscles") or exercise.get("target_muscle"))
    secondary = _as_list(exercise.get("secondary_muscles"))
    equipment = _as_list(exercise.get("equipment"))
    notes = str(exercise.get("notes", "")).strip()
    pattern = str(exercise.get("movement_pattern") or exercise.get("movement_type") or "").strip()
    difficulty = str(exercise.get("difficulty", "")).strip()
    source = str(exercise.get("source") or "local").strip()
    focus_label = focus or ", ".join(_as_list(exercise.get("focus_tags")))
    muscles_text = ", ".join(primary or _as_list(exercise.get("target_muscle")) or secondary)
    equipment_text = ", ".join(equipment) or "available equipment"

    coaching_cue = _first_sentence(notes) or _cue_for_pattern(pattern, name)
    return {
        "primary_muscles": primary,
        "secondary_muscles": secondary,
        "coaching_cue": coaching_cue,
        "why_this_exercise": _why_this_exercise(
            name=name,
            focus=focus_label,
            muscles=muscles_text,
            pattern=pattern,
            equipment=equipment_text,
        ),
        "common_mistake": _common_mistake_for_pattern(pattern),
        "regression": _regression_for_exercise(name=name, equipment=equipment),
        "progression": _progression_for_exercise(name=name, difficulty=difficulty, pattern=pattern),
        "knowledge_source": source,
    }


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    parts = [part.strip() for part in text.replace("\n", " ").split(".") if part.strip()]
    return parts[0] + "." if parts else text.strip()


def _cue_for_pattern(pattern: str, name: str) -> str:
    pattern_key = _normalize(pattern)
    if "squat" in pattern_key or "knee" in pattern_key:
        return "Keep the feet planted, control the descent, and let the knees track with the toes."
    if "hinge" in pattern_key or "deadlift" in pattern_key:
        return "Brace the trunk, hinge from the hips, and keep the spine neutral."
    if "pull" in pattern_key or "row" in pattern_key:
        return "Start each rep by setting the shoulder blades, then pull without shrugging."
    if "push" in pattern_key or "press" in pattern_key:
        return "Brace the core and press smoothly without losing shoulder control."
    if "core" in pattern_key:
        return "Move slowly enough to keep the ribs down and the pelvis controlled."
    return f"Use controlled tempo and clean range of motion on {name}."


def _why_this_exercise(*, name: str, focus: str, muscles: str, pattern: str, equipment: str) -> str:
    pieces = []
    if focus:
        pieces.append(f"fits the {focus} focus")
    if muscles:
        pieces.append(f"targets {muscles}")
    if pattern:
        pieces.append(f"uses a {pattern} pattern")
    if equipment:
        pieces.append(f"works with {equipment}")
    return f"{name} " + ", ".join(pieces) + "." if pieces else f"{name} supports today's training goal."


def _common_mistake_for_pattern(pattern: str) -> str:
    pattern_key = _normalize(pattern)
    if "squat" in pattern_key or "knee" in pattern_key:
        return "Rushing depth while the knees cave inward or the heels lift."
    if "hinge" in pattern_key or "deadlift" in pattern_key:
        return "Rounding the lower back instead of hinging from the hips."
    if "pull" in pattern_key or "row" in pattern_key:
        return "Shrugging the shoulders and pulling with momentum."
    if "push" in pattern_key or "press" in pattern_key:
        return "Flaring the elbows or losing rib and shoulder position."
    if "core" in pattern_key:
        return "Moving too fast and letting the lower back arch."
    return "Using momentum instead of controlled reps."


def _regression_for_exercise(*, name: str, equipment: list[str]) -> str:
    name_key = _normalize(name)
    if "push_up" in name_key:
        return "Elevate the hands or use a slower partial range."
    if "squat" in name_key:
        return "Use a box target or reduce depth while keeping control."
    if "press" in name_key:
        return "Use lighter load or a seated setup."
    if "row" in name_key or "pulldown" in name_key:
        return "Use a lighter load and pause briefly at the end range."
    if "bodyweight" in {_normalize(item) for item in equipment}:
        return "Reduce range of motion and slow the tempo."
    return "Use a lighter load and keep the movement range comfortable."


def _progression_for_exercise(*, name: str, difficulty: str, pattern: str) -> str:
    if _normalize(difficulty) == "beginner":
        return "Add reps within the target range before increasing load."
    if "conditioning" in _normalize(pattern):
        return "Add work time or reduce rest while keeping clean movement."
    return "Increase load gradually once all sets stay technically clean."


def _preference_score(
    *,
    exercise_difficulty: str,
    requested_difficulty: str | None,
    goal_tags: set[str],
    requested_goal: str | None,
    recommendation_tags: set[str],
    requested_recommendations: set[str],
) -> int:
    score = 0
    if requested_difficulty:
        score += _difficulty_preference_score(requested_difficulty, exercise_difficulty)
    if requested_goal and requested_goal in goal_tags:
        score += 20
    if requested_recommendations and requested_recommendations.intersection(recommendation_tags):
        score += 5
    return score


def _difficulty_preference_score(requested: str, exercise: str) -> int:
    if not exercise:
        return 0
    difficulty_rank = {"beginner": 0, "intermediate": 1, "advanced": 2}
    requested_rank = difficulty_rank.get(requested)
    exercise_rank = difficulty_rank.get(exercise)
    if requested_rank is None or exercise_rank is None:
        return 0
    if requested_rank == exercise_rank:
        return 30
    distance = abs(requested_rank - exercise_rank)
    if requested == "beginner":
        return 8 if exercise == "intermediate" else -20
    if requested == "advanced":
        return 20 if exercise == "intermediate" else 5
    return 15 if distance == 1 else -10


def build_video_resources(exercise_names: Iterable[str]) -> list[dict[str, str]]:
    """Return lightweight video resource objects for planned exercises."""

    from agent.tools.youtube_tool import search_youtube_video

    resources: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for name in exercise_names:
        exercise = get_exercise_by_name(name)
        display_name = str(exercise.get("name") if exercise else name).strip()
        if not display_name or _normalize(display_name) in seen_names:
            continue
        seen_names.add(_normalize(display_name))

        local_url = str((exercise or {}).get("youtube_url", "")).strip()
        media_url = str((exercise or {}).get("media_url", "")).strip()
        if local_url:
            resources.append(
                {
                    "exercise_name": display_name,
                    "title": f"{display_name} tutorial",
                    "url": local_url,
                    "source": "youtube",
                }
            )
            continue
        if media_url:
            resources.append(
                {
                    "exercise_name": display_name,
                    "title": f"{display_name} demo",
                    "url": media_url,
                    "source": str((exercise or {}).get("source", "exercise_media")),
                }
            )
            continue

        youtube_match = search_youtube_video(display_name)
        if youtube_match:
            resources.append(
                {
                    "exercise_name": display_name,
                    "title": youtube_match["title"],
                    "url": youtube_match["url"],
                    "source": youtube_match["source"],
                }
            )
            continue

        resources.append(
            {
                "exercise_name": display_name,
                "title": f"{display_name} exercise tutorial search",
                "url": f"https://www.youtube.com/results?search_query={quote_plus(display_name + ' exercise tutorial proper form')}",
                "source": "youtube_search",
            }
        )
    return resources
