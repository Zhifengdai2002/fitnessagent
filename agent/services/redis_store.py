"""Optional Redis cache for short-lived runtime state.

Redis is treated as an acceleration layer only. MySQL/JSON remain the durable
source of truth, so losing Redis should never lose user history.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

DEFAULT_TTL_SECONDS = 60 * 60 * 24 * 7
STATE_KEY_PREFIX = "fitness_agent:state"
CACHE_KEY_PREFIX = "fitness_agent:cache"


def redis_url() -> str:
    return os.getenv("REDIS_URL", "").strip()


def is_redis_configured() -> bool:
    return bool(redis_url())


def redis_ttl_seconds() -> int:
    raw_value = os.getenv("REDIS_STATE_TTL_SECONDS", "").strip()
    if not raw_value:
        return DEFAULT_TTL_SECONDS
    try:
        return max(0, int(raw_value))
    except ValueError:
        return DEFAULT_TTL_SECONDS


def redis_cache_ttl_seconds() -> int:
    raw_value = os.getenv("REDIS_CACHE_TTL_SECONDS", "").strip()
    if not raw_value:
        return redis_ttl_seconds()
    try:
        return max(0, int(raw_value))
    except ValueError:
        return redis_ttl_seconds()


def state_cache_key(user_id: str) -> str:
    clean_user = str(user_id or "demo-user").strip() or "demo-user"
    return f"{STATE_KEY_PREFIX}:{clean_user}"


def item_cache_key(namespace: str, key: str) -> str:
    clean_namespace = _clean_key_part(namespace or "default")
    clean_key = _clean_key_part(key or "item")
    return f"{CACHE_KEY_PREFIX}:{clean_namespace}:{clean_key}"


def load_state_from_redis(user_id: str) -> dict[str, Any]:
    if not is_redis_configured():
        return {}
    try:
        client = _redis_client()
        raw_payload = client.get(state_cache_key(user_id))
    except Exception:
        return {}
    if raw_payload is None:
        return {}
    if isinstance(raw_payload, bytes):
        raw_payload = raw_payload.decode("utf-8")
    try:
        payload = json.loads(str(raw_payload))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_state_to_redis(payload: dict[str, Any], user_id: str) -> bool:
    if not is_redis_configured():
        return False
    try:
        client = _redis_client()
        serialized = json.dumps(_json_safe(payload), ensure_ascii=False)
        ttl_seconds = redis_ttl_seconds()
        if ttl_seconds > 0:
            client.setex(state_cache_key(user_id), ttl_seconds, serialized)
        else:
            client.set(state_cache_key(user_id), serialized)
    except Exception:
        return False
    return True


def delete_state_from_redis(user_id: str) -> bool:
    if not is_redis_configured():
        return False
    try:
        _redis_client().delete(state_cache_key(user_id))
    except Exception:
        return False
    return True


def load_cache_item(namespace: str, key: str) -> dict[str, Any] | None:
    if not is_redis_configured():
        return None
    try:
        raw_payload = _redis_client().get(item_cache_key(namespace, key))
    except Exception:
        return None
    if raw_payload is None:
        return None
    if isinstance(raw_payload, bytes):
        raw_payload = raw_payload.decode("utf-8")
    try:
        payload = json.loads(str(raw_payload))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def save_cache_item(namespace: str, key: str, payload: dict[str, Any], *, ttl_seconds: int | None = None) -> bool:
    if not is_redis_configured() or not payload:
        return False
    try:
        serialized = json.dumps(_json_safe(payload), ensure_ascii=False)
        ttl = redis_cache_ttl_seconds() if ttl_seconds is None else max(0, int(ttl_seconds))
        client = _redis_client()
        if ttl > 0:
            client.setex(item_cache_key(namespace, key), ttl, serialized)
        else:
            client.set(item_cache_key(namespace, key), serialized)
    except Exception:
        return False
    return True


def _redis_client() -> Any:
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError("redis package is not installed") from exc
    return redis.Redis.from_url(redis_url(), decode_responses=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _clean_key_part(value: str) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").split()) or "item"
