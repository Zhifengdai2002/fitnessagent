"""Runtime dependency diagnostics for the FastAPI backend."""

from __future__ import annotations

from typing import Any

from agent.config import load_settings
from agent.rag.milvus_store import exercise_collection_name, food_collection_name, milvus_enabled
from agent.services.mysql_store import database_url, is_mysql_configured
from agent.services.redis_store import is_redis_configured, redis_cache_ttl_seconds, redis_ttl_seconds, redis_url
from agent.services.video_cache import VIDEO_CACHE_PATH, load_video_cache


def dependency_health() -> dict[str, Any]:
    """Return non-secret health details for runtime integrations."""

    settings = load_settings()
    mysql = _mysql_health()
    redis = _redis_health()
    milvus = _milvus_health()
    youtube = _youtube_health(settings.has_youtube_key)
    video_cache = _video_cache_health(redis=redis, mysql=mysql)

    critical_checks = [
        _optional_ok(mysql),
        _optional_ok(redis),
        _optional_ok(milvus),
        _optional_ok(youtube),
    ]
    status = "ok" if all(critical_checks) else "degraded"

    return {
        "status": status,
        "services": {
            "mysql": mysql,
            "redis": redis,
            "milvus": milvus,
            "youtube": youtube,
            "video_cache": video_cache,
        },
        "rag": {
            "configured_backend": settings.rag_backend,
            "active_backend": _active_rag_backend(milvus),
            "exercise_collection": exercise_collection_name(),
            "food_collection": food_collection_name(),
        },
    }


def _mysql_health() -> dict[str, Any]:
    configured = is_mysql_configured()
    if not configured:
        return {
            "configured": False,
            "reachable": False,
            "role": "durable_state_and_structured_history",
            "detail": "DATABASE_URL or MYSQL_URL is not set; JSON fallback is active.",
        }

    try:
        from sqlalchemy import create_engine, text
    except Exception as exc:
        return {
            "configured": True,
            "reachable": False,
            "role": "durable_state_and_structured_history",
            "detail": f"sqlalchemy unavailable: {type(exc).__name__}",
        }

    try:
        engine = create_engine(database_url(), pool_pre_ping=True)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        return {
            "configured": True,
            "reachable": False,
            "role": "durable_state_and_structured_history",
            "detail": f"connection failed: {type(exc).__name__}",
        }

    return {
        "configured": True,
        "reachable": True,
        "role": "durable_state_and_structured_history",
        "detail": "connected",
    }


def _redis_health() -> dict[str, Any]:
    configured = is_redis_configured()
    if not configured:
        return {
            "configured": False,
            "reachable": False,
            "role": "short_lived_state_and_cache",
            "state_ttl_seconds": redis_ttl_seconds(),
            "cache_ttl_seconds": redis_cache_ttl_seconds(),
            "detail": "REDIS_URL is not set; cache falls back to MySQL/JSON where available.",
        }

    try:
        import redis
    except Exception as exc:
        return {
            "configured": True,
            "reachable": False,
            "role": "short_lived_state_and_cache",
            "state_ttl_seconds": redis_ttl_seconds(),
            "cache_ttl_seconds": redis_cache_ttl_seconds(),
            "detail": f"redis package unavailable: {type(exc).__name__}",
        }

    try:
        client = redis.Redis.from_url(redis_url(), decode_responses=True)
        client.ping()
    except Exception as exc:
        return {
            "configured": True,
            "reachable": False,
            "role": "short_lived_state_and_cache",
            "state_ttl_seconds": redis_ttl_seconds(),
            "cache_ttl_seconds": redis_cache_ttl_seconds(),
            "detail": f"connection failed: {type(exc).__name__}",
        }

    return {
        "configured": True,
        "reachable": True,
        "role": "short_lived_state_and_cache",
        "state_ttl_seconds": redis_ttl_seconds(),
        "cache_ttl_seconds": redis_cache_ttl_seconds(),
        "detail": "connected",
    }


def _milvus_health() -> dict[str, Any]:
    configured = milvus_enabled()
    if not configured:
        return {
            "configured": False,
            "reachable": False,
            "role": "vector_rag_primary_store",
            "collections": {},
            "detail": "RAG_BACKEND/MILVUS_URI do not enable Milvus; local vector fallback is active.",
        }

    try:
        from agent.rag.milvus_store import milvus_client

        client = milvus_client()
        collections = set(client.list_collections())
        exercise_ready = exercise_collection_name() in collections
        food_ready = food_collection_name() in collections
    except Exception as exc:
        return {
            "configured": True,
            "reachable": False,
            "role": "vector_rag_primary_store",
            "collections": {
                "exercise_ready": False,
                "food_ready": False,
            },
            "detail": f"connection failed: {type(exc).__name__}",
        }

    return {
        "configured": True,
        "reachable": True,
        "role": "vector_rag_primary_store",
        "collections": {
            "exercise_ready": exercise_ready,
            "food_ready": food_ready,
        },
        "detail": "connected" if exercise_ready and food_ready else "connected, but one or more collections need indexing",
    }


def _youtube_health(has_youtube_key: bool) -> dict[str, Any]:
    if not has_youtube_key:
        return {
            "configured": False,
            "reachable": False,
            "role": "exercise_video_backfill",
            "detail": "YOUTUBE_API_KEY is not set; video backfill will use search-link fallback.",
        }

    try:
        import googleapiclient.discovery  # noqa: F401
    except Exception as exc:
        return {
            "configured": True,
            "reachable": False,
            "role": "exercise_video_backfill",
            "detail": f"google api client unavailable: {type(exc).__name__}",
        }

    return {
        "configured": True,
        "reachable": None,
        "role": "exercise_video_backfill",
        "detail": "client available; live API calls are not made during health checks to avoid quota usage",
    }


def _video_cache_health(*, redis: dict[str, Any], mysql: dict[str, Any]) -> dict[str, Any]:
    try:
        json_entries = len(load_video_cache(VIDEO_CACHE_PATH))
    except Exception:
        json_entries = 0

    layers = []
    if redis.get("configured"):
        layers.append({"name": "redis", "reachable": redis.get("reachable")})
    if mysql.get("configured"):
        layers.append({"name": "mysql", "reachable": mysql.get("reachable")})
    layers.append({"name": "json_fallback", "reachable": True, "entries": json_entries})

    return {
        "configured": True,
        "reachable": True,
        "role": "video_resource_cache",
        "layers": layers,
        "json_cache_entries": json_entries,
    }


def _active_rag_backend(milvus: dict[str, Any]) -> str:
    if milvus.get("configured") and milvus.get("reachable"):
        collections = milvus.get("collections") if isinstance(milvus.get("collections"), dict) else {}
        if collections.get("exercise_ready") and collections.get("food_ready"):
            return "milvus"
        return "milvus_partial_with_local_fallback"
    return "local_vector_fallback"


def _optional_ok(service: dict[str, Any]) -> bool:
    if not service.get("configured"):
        return True
    reachable = service.get("reachable")
    return reachable is True or reachable is None
