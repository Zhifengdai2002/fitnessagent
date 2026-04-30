"""YouTube lookup helpers for exercise demos."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from agent.config import load_settings


@lru_cache(maxsize=256)
def search_youtube_video(exercise_name: str) -> dict[str, str] | None:
    """Return one YouTube tutorial result when API access is configured."""

    query_name = exercise_name.strip()
    if not query_name:
        return None

    settings = load_settings()
    if not settings.has_youtube_key:
        return None

    try:
        from googleapiclient.discovery import build
    except Exception:
        return None

    try:
        youtube = build("youtube", "v3", developerKey=settings.youtube_api_key, cache_discovery=False)
        response: dict[str, Any] = (
            youtube.search()
            .list(
                part="snippet",
                q=f"{query_name} exercise tutorial proper form",
                type="video",
                maxResults=1,
                safeSearch="strict",
                videoEmbeddable="true",
            )
            .execute()
        )
    except Exception:
        return None

    items = response.get("items") or []
    if not items:
        return None

    item = items[0]
    video_id = item.get("id", {}).get("videoId")
    title = item.get("snippet", {}).get("title") or f"{query_name} tutorial"
    if not video_id:
        return None
    return {
        "title": str(title),
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "source": "youtube_api",
        "provider": "youtube",
        "video_id": str(video_id),
        "channel_title": str(item.get("snippet", {}).get("channelTitle") or ""),
    }
