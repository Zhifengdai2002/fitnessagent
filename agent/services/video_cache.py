"""Video resource cache with MySQL primary storage and JSON fallback."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.services.mysql_store import load_video_resource_from_mysql, save_video_resource_to_mysql
from agent.services.redis_store import load_cache_item, save_cache_item

VIDEO_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "video_cache.json"
VIDEO_CACHE_NAMESPACE = "video_resource"


def get_cached_video_resource(exercise_name: str, path: Path = VIDEO_CACHE_PATH) -> dict[str, Any] | None:
    """Return a cached video resource for an exercise."""

    exercise_key = video_cache_key(exercise_name)
    if not exercise_key:
        return None

    redis_resource = load_cache_item(VIDEO_CACHE_NAMESPACE, exercise_key)
    if redis_resource and redis_resource.get("url"):
        return {**redis_resource, "cache_status": "hit"}

    mysql_resource = load_video_resource_from_mysql(exercise_name)
    if mysql_resource:
        save_cache_item(VIDEO_CACHE_NAMESPACE, exercise_key, mysql_resource)
        return {**mysql_resource, "cache_status": "hit"}

    cache = load_video_cache(path)
    resource = cache.get(exercise_key)
    if isinstance(resource, dict) and resource.get("url"):
        save_cache_item(VIDEO_CACHE_NAMESPACE, exercise_key, resource)
        return {**resource, "cache_status": "hit"}
    return None


def save_cached_video_resource(
    exercise_name: str,
    resource: dict[str, Any],
    path: Path = VIDEO_CACHE_PATH,
) -> dict[str, Any]:
    """Persist a video resource and return its normalized payload."""

    exercise_key = video_cache_key(exercise_name)
    normalized = normalize_video_resource(exercise_name, resource)
    if not exercise_key or not normalized.get("url"):
        return normalized

    save_cache_item(VIDEO_CACHE_NAMESPACE, exercise_key, normalized)
    if save_video_resource_to_mysql(exercise_name, normalized):
        return normalized

    cache = load_video_cache(path)
    cache[exercise_key] = normalized
    save_video_cache(cache, path)
    return normalized


def normalize_video_resource(exercise_name: str, resource: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "exercise_name": str(resource.get("exercise_name") or exercise_name).strip(),
        "title": str(resource.get("title") or f"{exercise_name} tutorial").strip(),
        "url": str(resource.get("url") or "").strip(),
        "source": str(resource.get("source") or "video_cache").strip(),
        "provider": str(resource.get("provider") or resource.get("source") or "video_cache").strip(),
        "video_id": str(resource.get("video_id") or "").strip(),
        "channel_title": str(resource.get("channel_title") or "").strip(),
        "checked_at": str(resource.get("checked_at") or now).strip(),
        "cache_status": str(resource.get("cache_status") or "saved").strip(),
    }


def load_video_cache(path: Path = VIDEO_CACHE_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_video_cache(cache: dict[str, dict[str, Any]], path: Path = VIDEO_CACHE_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        return


def video_cache_key(exercise_name: str) -> str:
    return "_".join(str(exercise_name or "").strip().lower().replace("-", " ").split())
