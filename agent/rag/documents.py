"""Build retrievable documents from local fitness knowledge sources."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

EXERCISE_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "exercise_db.json"
FOOD_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "food_db.json"
EXERCISE_RAG_SEED_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge" / "exercise_rag_seed.json"
FOOD_RAG_SEED_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge" / "food_rag_seed.json"
KNOWLEDGE_RAG_SEED_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge" / "professional_knowledge_seed.json"
KNOWLEDGE_RAG_CORPUS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "knowledge" / "professional_knowledge_corpus.json"
)
WGER_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "external" / "wger_exercises.json"
LOCAL_EXERCISE_FALLBACK_LIMIT = 10
LOCAL_FOOD_FALLBACK_LIMIT = 5


def build_exercise_documents() -> list[dict[str, Any]]:
    """Return normalized exercise documents for local retrieval."""

    return _build_exercise_documents(load_exercise_documents_source())


def build_primary_exercise_documents() -> list[dict[str, Any]]:
    """Return normalized exercise documents for Milvus/main RAG retrieval."""

    return _build_exercise_documents(load_primary_exercise_documents_source())


def _build_exercise_documents(exercises: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for exercise in exercises:
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
                    "source": exercise.get("source") or "local",
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


def build_food_documents() -> list[dict[str, Any]]:
    """Return normalized food documents for nutrition retrieval."""

    return _build_food_documents(load_food_documents_source())


def build_primary_food_documents() -> list[dict[str, Any]]:
    """Return normalized food documents for Milvus/main RAG retrieval."""

    return _build_food_documents(load_primary_food_documents_source())


def _build_food_documents(foods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for food in foods:
        name = str(food.get("name", "")).strip()
        if not name:
            continue
        category = str(food.get("category", "")).strip()
        diet_tags = _list_text(food.get("diet_tags"))
        allergens = _list_text(food.get("allergens"))
        notes = str(food.get("notes", "")).strip()
        macro_line = (
            f"Calories {food.get('calories_per_100g', 0)} kcal, "
            f"protein {food.get('protein_g', 0)}g, carbs {food.get('carbs_g', 0)}g, "
            f"fat {food.get('fat_g', 0)}g, fiber {food.get('fiber_g', 0)}g per 100g"
        )

        text = "\n".join(
            piece
            for piece in [
                f"Food: {name}",
                f"Category: {category}" if category else "",
                macro_line,
                f"Diet tags: {diet_tags}" if diet_tags else "",
                f"Allergens: {allergens}" if allergens else "",
                f"Nutrition notes: {notes}" if notes else "",
            ]
            if piece
        )
        documents.append(
            {
                "id": f"food:{food.get('id') or _slug(name)}",
                "type": "food",
                "title": name,
                "text": text,
                "metadata": {
                    "source": food.get("source") or "local",
                    "food_id": food.get("id", ""),
                    "name": name,
                    "category": category,
                    "diet_tags": food.get("diet_tags") or [],
                    "allergens": food.get("allergens") or [],
                    "calories_per_100g": food.get("calories_per_100g", 0),
                    "protein_g": food.get("protein_g", 0),
                    "carbs_g": food.get("carbs_g", 0),
                    "fat_g": food.get("fat_g", 0),
                    "fiber_g": food.get("fiber_g", 0),
                    "notes": notes,
                },
                "raw": food,
            }
        )
    return documents


def build_knowledge_documents() -> list[dict[str, Any]]:
    """Return professional long-form knowledge documents for coaching QA."""

    return _build_knowledge_documents(load_knowledge_documents_source())


def build_primary_knowledge_documents() -> list[dict[str, Any]]:
    """Return professional long-form knowledge documents for Milvus/main RAG."""

    return _build_knowledge_documents(load_primary_knowledge_documents_source())


def _build_knowledge_documents(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for item in items:
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        chunks = item.get("chunks")
        chunk_texts = chunks if isinstance(chunks, list) else [item.get("text", "")]
        extra_metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        for chunk_index, chunk in enumerate(chunk_texts):
            chunk_text = str(chunk or "").strip()
            if not chunk_text:
                continue
            source = str(item.get("source") or "professional_knowledge").strip()
            source_url = str(item.get("source_url") or "").strip()
            topic = str(item.get("topic") or "").strip()
            section = str(item.get("section") or "").strip()
            doc_type = str(item.get("doc_type") or "").strip()
            evidence_type = str(item.get("evidence_type") or "").strip()
            goals = _listify(item.get("goal"))
            levels = _listify(item.get("level"))
            tags = _listify(item.get("tags"))
            doc_id_base = str(item.get("id") or _slug(title)).strip()

            text = "\n".join(
                piece
                for piece in [
                    f"Title: {title}",
                    f"Source: {source}" if source else "",
                    f"Topic: {topic}" if topic else "",
                    f"Section: {section}" if section else "",
                    f"Evidence type: {evidence_type}" if evidence_type else "",
                    f"Goals: {_list_text(goals)}" if goals else "",
                    f"Levels: {_list_text(levels)}" if levels else "",
                    f"Tags: {_list_text(tags)}" if tags else "",
                    chunk_text,
                ]
                if piece
            )
            document_id = f"{doc_id_base}:{chunk_index}"
            metadata = {
                "id": document_id,
                "source": source,
                "source_url": source_url,
                "doc_type": doc_type,
                "section": section,
                "topic": topic,
                "goal": goals,
                "level": levels,
                "tags": tags,
                "evidence_type": evidence_type,
                "chunk_index": chunk_index,
                "version": str(item.get("version") or "professional_seed_v1"),
                **extra_metadata,
            }
            documents.append(
                {
                    "id": f"knowledge:{document_id}",
                    "type": "knowledge",
                    "title": title,
                    "text": text,
                    "metadata": metadata,
                    "raw": {
                        "id": document_id,
                        "title": title,
                        "source": source,
                        "source_url": source_url,
                        "doc_type": doc_type,
                        "section": section,
                        "topic": topic,
                        "evidence_type": evidence_type,
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                    },
                }
            )
    return documents


@lru_cache(maxsize=1)
def load_exercise_documents_source() -> list[dict[str, Any]]:
    exercises = list(load_primary_exercise_documents_source())
    fallback = load_local_exercise_fallback_source()
    exercises.extend(fallback)
    return _dedupe_exercises(exercises)


@lru_cache(maxsize=1)
def load_food_documents_source() -> list[dict[str, Any]]:
    foods = list(load_primary_food_documents_source())
    foods.extend(load_local_food_fallback_source())
    return _dedupe_by_name(foods)


@lru_cache(maxsize=1)
def load_knowledge_documents_source() -> list[dict[str, Any]]:
    return load_primary_knowledge_documents_source()


@lru_cache(maxsize=1)
def load_primary_exercise_documents_source() -> list[dict[str, Any]]:
    """Professional/RAG exercise sources used before local fallback."""

    exercises: list[dict[str, Any]] = []
    if EXERCISE_RAG_SEED_PATH.exists():
        exercises.extend(_load_json_list(EXERCISE_RAG_SEED_PATH))
    if WGER_CACHE_PATH.exists():
        exercises.extend(
            item
            for item in (_clean_primary_exercise_source(item) for item in _load_json_list(WGER_CACHE_PATH))
            if item and not _is_low_quality_wger_exercise(item)
        )
    return _dedupe_exercises(exercises)


@lru_cache(maxsize=1)
def load_primary_food_documents_source() -> list[dict[str, Any]]:
    """Professional/RAG food sources used before local fallback."""

    if FOOD_RAG_SEED_PATH.exists():
        return _dedupe_by_name(_load_json_list(FOOD_RAG_SEED_PATH))
    return []


@lru_cache(maxsize=1)
def load_primary_knowledge_documents_source() -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    if KNOWLEDGE_RAG_CORPUS_PATH.exists():
        sources.extend(_load_json_list(KNOWLEDGE_RAG_CORPUS_PATH))
    if KNOWLEDGE_RAG_SEED_PATH.exists():
        sources.extend(_load_json_list(KNOWLEDGE_RAG_SEED_PATH))
    return _dedupe_knowledge_sources(sources)


@lru_cache(maxsize=1)
def load_local_exercise_fallback_source() -> list[dict[str, Any]]:
    """Small local safety net kept only for missing RAG/API coverage."""

    fallback = _load_json_list(EXERCISE_DB_PATH)
    fallback = sorted(fallback, key=_local_exercise_fallback_priority)
    return [_mark_local_fallback(item) for item in fallback[:LOCAL_EXERCISE_FALLBACK_LIMIT]]


@lru_cache(maxsize=1)
def load_full_legacy_exercise_source() -> list[dict[str, Any]]:
    """Full legacy local DB for exact lookups on old persisted plans only."""

    return _load_json_list(EXERCISE_DB_PATH)


@lru_cache(maxsize=1)
def load_local_food_fallback_source() -> list[dict[str, Any]]:
    """Small local nutrition safety net kept only for missing RAG coverage."""

    fallback = _load_json_list(FOOD_DB_PATH)
    fallback = sorted(fallback, key=_local_food_fallback_priority)
    return [_mark_local_fallback(item) for item in fallback[:LOCAL_FOOD_FALLBACK_LIMIT]]


@lru_cache(maxsize=1)
def load_full_legacy_food_source() -> list[dict[str, Any]]:
    """Full legacy local DB for exact lookups on old persisted meals only."""

    return _load_json_list(FOOD_DB_PATH)


def _dedupe_exercises(exercises: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for item in exercises:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        name_key = _canonical_exercise_key(name)
        if name_key in seen_names:
            continue
        seen_names.add(name_key)
        deduped.append(item)
    return deduped


def _dedupe_by_name(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for item in items:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        name_key = _slug(name)
        if name_key in seen_names:
            continue
        seen_names.add(name_key)
        deduped.append(item)
    return deduped


def _dedupe_knowledge_sources(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in items:
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        key_parts = [
            str(item.get("id") or ""),
            str(item.get("source_url") or ""),
            title,
            str(item.get("section") or ""),
        ]
        key = "|".join(_slug(part) for part in key_parts if part)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(item)
    return deduped


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _clean_primary_exercise_source(item: dict[str, Any]) -> dict[str, Any]:
    source = str(item.get("source") or "").strip()
    if source != "wger":
        return item

    cleaned = dict(item)
    name = str(cleaned.get("name", "")).strip()
    name = re.sub(r"\s+exercise$", "", name, flags=re.IGNORECASE).strip()
    cleaned["name"] = name

    notes = str(cleaned.get("notes", "")).strip()
    notes = re.sub(r"^[\s.:-]+", "", notes).strip()
    if not notes:
        notes = "Use controlled form and stop if the movement causes pain."
    cleaned["notes"] = notes

    equipment = _listify(cleaned.get("equipment"))
    if _is_generic_wger_equipment(equipment):
        inferred = _infer_equipment_from_name(name)
        if inferred:
            cleaned["equipment"] = inferred
    return cleaned


def _is_low_quality_wger_exercise(item: dict[str, Any]) -> bool:
    if str(item.get("source") or "") != "wger":
        return False
    name = str(item.get("name", "")).strip()
    notes = str(item.get("notes", "")).strip()
    if not name:
        return True
    if not name.isascii() or (notes and not notes.isascii()):
        return True
    if len(name.split()) > 7:
        return True
    if any(token in name.lower() for token in ["stretching", "rehab only"]):
        return True
    muscles = _listify(item.get("primary_muscles") or item.get("target_muscle"))
    focus = _listify(item.get("focus_tags"))
    return not muscles and not focus


def _is_generic_wger_equipment(equipment: list[str]) -> bool:
    normalized = {_slug(value) for value in equipment}
    return not normalized or normalized in [{"gym_mat"}, {"mat"}]


def _infer_equipment_from_name(name: str) -> list[str]:
    normalized = name.lower()
    if "cable" in normalized:
        return ["cable_machine"]
    if "machine" in normalized or "leg press" in normalized or "chest press" in normalized:
        return ["machine"]
    if "dumbbell" in normalized:
        return ["dumbbell"]
    if "barbell" in normalized:
        return ["barbell"]
    if "kettlebell" in normalized:
        return ["kettlebell"]
    return []


def _mark_local_fallback(item: dict[str, Any]) -> dict[str, Any]:
    return {**item, "source": "local_fallback"}


def _local_exercise_fallback_priority(exercise: dict[str, Any]) -> tuple[int, str]:
    name = _slug(str(exercise.get("name", "")))
    preferred_names = [
        "incline_push_up",
        "push_up",
        "machine_chest_press",
        "lat_pulldown",
        "seated_cable_row",
        "goblet_squat",
        "glute_bridge",
        "step_up",
        "dumbbell_lateral_raise",
        "face_pull",
        "plank",
        "dead_bug",
        "mountain_climber",
        "jump_rope",
    ]
    preferred_rank = {preferred_name: index for index, preferred_name in enumerate(preferred_names)}
    recommended = _list_text(exercise.get("recommended_for"))
    rank = preferred_rank.get(name, len(preferred_names) + 1)
    if "beginner_program" in recommended:
        rank -= 1
    return (rank, name)


def _local_food_fallback_priority(food: dict[str, Any]) -> tuple[int, str]:
    name = _slug(str(food.get("name", "")))
    preferred_names = [
        "chicken_breast",
        "egg",
        "firm_tofu",
        "brown_rice,_cooked",
        "white_rice,_cooked",
        "sweet_potato",
        "broccoli",
        "apple",
    ]
    preferred_rank = {preferred_name: index for index, preferred_name in enumerate(preferred_names)}
    return (preferred_rank.get(name, len(preferred_names) + 1), name)


def _list_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return ", ".join(str(item).strip() for item in value if str(item).strip())


def _listify(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _slug(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _canonical_exercise_key(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    tokens = [
        token
        for token in normalized.split()
        if token not in {"exercise", "exercises", "workout", "movement", "demo", "tutorial"}
    ]
    return " ".join(tokens)
