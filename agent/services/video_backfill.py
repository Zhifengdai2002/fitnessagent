"""Resolve and backfill exercise video resources.

This service is the single path for turning an exercise name into a video.
Exercise knowledge can come from wger/RAG/Milvus, but tutorial videos are
resolved from YouTube API results only. Cached YouTube API results are reused
before doing another lookup.
"""

from __future__ import annotations

from typing import Any, Iterable
from urllib.parse import quote_plus

from agent.services.video_cache import get_cached_video_resource, save_cached_video_resource
from agent.tools.youtube_tool import search_youtube_video


def resolve_exercise_video_resource(
    exercise_name: str,
    *,
    local_resource: dict[str, Any] | None = None,
    allow_search_fallback: bool = False,
) -> dict[str, Any] | None:
    """Resolve one exercise video and persist YouTube API matches."""

    display_name = str(exercise_name or "").strip()
    if not display_name:
        return None

    cached_match = get_cached_video_resource(display_name)
    if _is_youtube_api_resource(cached_match):
        return {**cached_match, "cache_status": "hit"}

    youtube_match = search_youtube_video(display_name)
    if youtube_match:
        return save_cached_video_resource(display_name, youtube_match)

    if not allow_search_fallback:
        return None
    return youtube_search_fallback_resource(display_name)


def backfill_video_resources(
    exercise_names: Iterable[str],
    *,
    local_resources: dict[str, dict[str, Any]] | None = None,
    allow_search_fallback: bool = False,
) -> list[dict[str, Any]]:
    """Resolve YouTube API videos for a sequence of exercises while deduping names."""

    resources: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for name in exercise_names:
        display_name = str(name or "").strip()
        key = _normalize_key(display_name)
        if not display_name or key in seen_keys:
            continue
        seen_keys.add(key)
        resource = resolve_exercise_video_resource(
            display_name,
            local_resource=(local_resources or {}).get(key),
            allow_search_fallback=allow_search_fallback,
        )
        if resource:
            resources.append(resource)
    return resources


def youtube_search_fallback_resource(exercise_name: str) -> dict[str, str]:
    return {
        "exercise_name": exercise_name,
        "title": f"{exercise_name} exercise tutorial search",
        "url": f"https://www.youtube.com/results?search_query={quote_plus(exercise_name + ' exercise tutorial proper form')}",
        "source": "youtube_search",
        "provider": "youtube",
    }


def _is_youtube_api_resource(resource: dict[str, Any] | None) -> bool:
    if not resource:
        return False
    source = str(resource.get("source") or "").strip().lower()
    provider = str(resource.get("provider") or "").strip().lower()
    url = str(resource.get("url") or "").strip().lower()
    return source == "youtube_api" and (not provider or provider == "youtube") and (
        "youtube.com/watch" in url or "youtu.be/" in url
    )


def _normalize_key(value: str) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").split())
