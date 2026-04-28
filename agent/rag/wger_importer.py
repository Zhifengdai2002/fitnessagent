"""Import and normalize wger exercises for the local RAG index."""

from __future__ import annotations

import argparse
import html
import json
import re
import ssl
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

WGER_BASE_URL = "https://wger.de/api/v2"
WGER_EXERCISEINFO_ENDPOINT = "/exerciseinfo/"
WGER_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "external" / "wger_exercises.json"


def import_wger_exercises(
    *,
    language: int = 2,
    limit: int = 200,
    max_pages: int = 3,
    base_url: str = WGER_BASE_URL,
    output_path: Path = WGER_CACHE_PATH,
    verify_ssl: bool = True,
) -> list[dict[str, Any]]:
    """Fetch wger exercises, normalize them, and persist the local cache."""

    raw_items = fetch_wger_exerciseinfo(
        language=language,
        limit=limit,
        max_pages=max_pages,
        base_url=base_url,
        verify_ssl=verify_ssl,
    )
    normalized = normalize_wger_exercises(raw_items)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def fetch_wger_exerciseinfo(
    *,
    language: int = 2,
    limit: int = 200,
    max_pages: int = 3,
    base_url: str = WGER_BASE_URL,
    verify_ssl: bool = True,
) -> list[dict[str, Any]]:
    """Fetch paginated exerciseinfo records from wger."""

    params = urllib.parse.urlencode({"language": language, "limit": limit, "offset": 0})
    next_url = f"{base_url.rstrip('/')}{WGER_EXERCISEINFO_ENDPOINT}?{params}"
    items: list[dict[str, Any]] = []
    for _ in range(max_pages):
        payload = fetch_json(next_url, verify_ssl=verify_ssl)
        results = payload.get("results", [])
        if isinstance(results, list):
            items.extend(item for item in results if isinstance(item, dict))
        next_value = payload.get("next")
        if not next_value:
            break
        next_url = str(next_value)
    return items


def fetch_json(url: str, *, verify_ssl: bool = True) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "FitnessAgent/0.1 local-rag-importer",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=30, context=ssl_context(verify_ssl)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def ssl_context(verify_ssl: bool) -> ssl.SSLContext:
    if not verify_ssl:
        return ssl._create_unverified_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def normalize_wger_exercises(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize wger records to the same schema as data/exercise_db.json."""

    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for item in items:
        name = first_translation_value(item, "name") or str(item.get("name") or "").strip()
        if not name:
            continue
        name_key = slug(name)
        if name_key in seen_names:
            continue
        seen_names.add(name_key)

        description = clean_html(first_translation_value(item, "description") or str(item.get("description") or ""))
        category = nested_name(item.get("category"))
        primary_muscles = names_from_list(item.get("muscles"))
        secondary_muscles = names_from_list(item.get("muscles_secondary"))
        equipment = names_from_list(item.get("equipment")) or ["bodyweight"]
        target_muscle = primary_muscles or secondary_muscles or muscle_hints_from_name(name)
        movement_pattern = infer_movement_pattern(name=name, category=category)
        movement_type = infer_movement_type(movement_pattern, category)
        focus_tags = infer_focus_tags(name=name, muscles=target_muscle + secondary_muscles, category=category)
        contraindications = infer_contraindications(name=name, movement_pattern=movement_pattern)
        image_url = first_media_url(item.get("images"), "image")
        video_url = first_media_url(item.get("videos"), "video") or ""

        normalized.append(
            {
                "id": f"wger_{item.get('id') or name_key}",
                "name": name,
                "target_muscle": target_muscle,
                "movement_type": movement_type,
                "difficulty": infer_difficulty(name=name, equipment=equipment, movement_pattern=movement_pattern),
                "equipment": equipment,
                "training_goal_tags": ["strength", "sculpting", "weight_loss"],
                "contraindications": contraindications,
                "joint_stress": infer_joint_stress(target_muscle, movement_pattern),
                "recommended_for": ["gym_program"] if any(eq != "bodyweight" for eq in equipment) else ["home_workout"],
                "focus_tags": focus_tags,
                "youtube_url": video_url,
                "media_url": image_url or video_url,
                "notes": description or f"Imported from wger. Use controlled form for {name}.",
                "movement_pattern": movement_pattern,
                "replacement_group": infer_replacement_group(name=name, movement_pattern=movement_pattern, muscles=target_muscle),
                "primary_muscles": primary_muscles or target_muscle,
                "secondary_muscles": secondary_muscles,
                "source": "wger",
                "source_id": item.get("id"),
                "category": category,
            }
        )
    return normalized


def load_wger_cache(path: Path = WGER_CACHE_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def first_translation_value(item: dict[str, Any], key: str) -> str:
    translations = item.get("translations")
    if isinstance(translations, list):
        for translation in translations:
            if (
                isinstance(translation, dict)
                and translation.get("language") == 2
                and str(translation.get(key) or "").strip()
            ):
                return str(translation[key]).strip()
        for translation in translations:
            if isinstance(translation, dict) and str(translation.get(key) or "").strip():
                return str(translation[key]).strip()
    return ""


def names_from_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for item in value:
        name = nested_name(item)
        if name:
            names.append(name)
    return names


def nested_name(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return ""
    for key in ("name_en", "name", "label"):
        if str(value.get(key) or "").strip():
            return str(value[key]).strip()
    return ""


def clean_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def first_media_url(value: Any, key_hint: str) -> str:
    if not isinstance(value, list):
        return ""
    for item in value:
        if not isinstance(item, dict):
            continue
        for key in (key_hint, "url", "image", "video"):
            if str(item.get(key) or "").strip():
                return str(item[key]).strip()
    return ""


def infer_focus_tags(*, name: str, muscles: list[str], category: str) -> list[str]:
    text = normalize_text(" ".join([name, category, *muscles]))
    tags: list[str] = []
    if any(term in text for term in ["chest", "pector", "triceps", "biceps", "arm"]):
        tags.append("upper_chest_arms")
    if any(term in text for term in ["shoulder", "deltoid", "delt"]):
        tags.append("upper_shoulders")
    if any(term in text for term in ["back", "lat", "trapezius", "rhomboid"]):
        tags.append("back_training")
    if any(term in text for term in ["quad", "glute", "hamstring", "calf", "leg"]):
        tags.append("lower_legs_glutes")
    if any(term in text for term in ["abs", "core", "abdominal", "oblique"]):
        tags.append("functional_core")
    if any(term in text for term in ["burpee", "jump", "conditioning", "mountain climber"]):
        tags.append("functional_conditioning")
    return tags or ["functional_conditioning"]


def infer_movement_pattern(*, name: str, category: str) -> str:
    text = normalize_text(f"{name} {category}")
    if any(term in text for term in ["bench press", "push up", "pushup", "chest press"]):
        return "horizontal_push"
    if any(term in text for term in ["overhead press", "shoulder press", "military press", "pike push"]):
        return "vertical_push"
    if any(term in text for term in ["row"]):
        return "horizontal_pull"
    if any(term in text for term in ["pulldown", "pull up", "pullup", "chin up", "chinup"]):
        return "vertical_pull"
    if any(term in text for term in ["squat", "lunge", "leg press", "step up"]):
        return "squat_pattern"
    if any(term in text for term in ["deadlift", "hinge", "good morning", "hip thrust", "bridge"]):
        return "hinge_pattern"
    if any(term in text for term in ["plank", "crunch", "sit up", "leg raise", "rotation"]):
        return "core"
    if any(term in text for term in ["raise", "curl", "extension", "fly"]):
        return "isolation"
    if any(term in text for term in ["jump", "burpee", "climber", "run"]):
        return "conditioning"
    return "general_strength"


def infer_movement_type(movement_pattern: str, category: str) -> str:
    if movement_pattern in {"horizontal_push", "vertical_push"}:
        return "push"
    if movement_pattern in {"horizontal_pull", "vertical_pull"}:
        return "pull"
    if movement_pattern == "core":
        return "core"
    if movement_pattern == "conditioning":
        return "conditioning"
    if "plyometrics" in normalize_text(category):
        return "power"
    return "strength"


def infer_replacement_group(*, name: str, movement_pattern: str, muscles: list[str]) -> str:
    text = normalize_text(" ".join([name, movement_pattern, *muscles]))
    if "shoulder" in text and "press" in text:
        return "shoulder_press"
    if "chest" in text and "press" in text:
        return "chest_press"
    if "row" in text:
        return "row_pattern"
    if "pulldown" in text or "pull up" in text:
        return "vertical_pull"
    if "squat" in text:
        return "squat_pattern"
    if "lunge" in text:
        return "lunge_pattern"
    if "bridge" in text or "thrust" in text or "deadlift" in text:
        return "hinge_glute"
    if "raise" in text and ("shoulder" in text or "delt" in text):
        return "shoulder_raise"
    return movement_pattern or "general_strength"


def infer_difficulty(*, name: str, equipment: list[str], movement_pattern: str) -> str:
    text = normalize_text(name)
    if any(term in text for term in ["machine", "seated", "assisted", "incline push"]):
        return "beginner"
    if any(term in text for term in ["barbell", "clean", "snatch", "pistol", "muscle up"]):
        return "advanced"
    if movement_pattern in {"conditioning", "core"} or equipment == ["bodyweight"]:
        return "beginner"
    return "intermediate"


def infer_contraindications(*, name: str, movement_pattern: str) -> list[str]:
    text = normalize_text(name)
    conditions: list[str] = []
    if movement_pattern in {"vertical_push", "horizontal_push"} or "shoulder" in text:
        conditions.append("acute shoulder pain")
    if movement_pattern in {"squat_pattern", "lunge_pattern"}:
        conditions.append("knee pain")
    if movement_pattern in {"hinge_pattern", "horizontal_pull"}:
        conditions.append("low back pain")
    if "wrist" in text or "push up" in text:
        conditions.append("wrist pain")
    return conditions


def infer_joint_stress(muscles: list[str], movement_pattern: str) -> list[str]:
    text = normalize_text(" ".join(muscles))
    joints: list[str] = []
    if "shoulder" in text or movement_pattern in {"vertical_push", "horizontal_push"}:
        joints.append("shoulder")
    if movement_pattern in {"squat_pattern", "lunge_pattern"}:
        joints.extend(["knee", "hip"])
    if movement_pattern == "hinge_pattern":
        joints.extend(["hip", "low back"])
    return joints


def muscle_hints_from_name(name: str) -> list[str]:
    text = normalize_text(name)
    if "shoulder" in text:
        return ["shoulders"]
    if "chest" in text:
        return ["chest"]
    if "back" in text or "row" in text:
        return ["back"]
    if "squat" in text or "lunge" in text:
        return ["quads", "glutes"]
    if "curl" in text:
        return ["biceps"]
    if "triceps" in text:
        return ["triceps"]
    return []


def normalize_text(value: str) -> str:
    return value.lower().replace("-", " ").replace("_", " ")


def slug(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import wger exercises into the local RAG cache.")
    parser.add_argument("--language", type=int, default=2)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--insecure", action="store_true", help="Disable SSL verification for local debugging.")
    args = parser.parse_args()
    exercises = import_wger_exercises(
        language=args.language,
        limit=args.limit,
        max_pages=args.max_pages,
        verify_ssl=not args.insecure,
    )
    print(f"Imported {len(exercises)} wger exercises into {WGER_CACHE_PATH}")


if __name__ == "__main__":
    main()
