"""Persistence helpers shared by Streamlit and FastAPI entrypoints."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from agent.services.mysql_store import (
    delete_state_from_mysql,
    is_mysql_configured,
    load_state_from_mysql,
    save_state_to_mysql,
)

APP_STATE_PATH = Path(__file__).resolve().parents[2] / "data" / "app_state.json"

PERSISTED_SESSION_KEYS = [
    "profile_inputs",
    "agent_result",
    "active_date",
    "completed_training_days",
    "week_history",
    "daily_history",
    "memory_store",
    "assistant_chat_messages",
    "last_feedback_summary",
    "last_action_message",
]


def load_app_state(path: Path = APP_STATE_PATH) -> dict[str, Any]:
    if is_mysql_configured():
        payload = load_state_from_mysql()
        if payload:
            return payload

    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_app_state(session_state: dict[str, Any], path: Path = APP_STATE_PATH) -> None:
    payload = {
        key: json_safe(session_state.get(key))
        for key in PERSISTED_SESSION_KEYS
    }
    payload["active_date"] = safe_iso_date(payload.get("active_date")) or date.today().isoformat()
    if save_state_to_mysql(payload):
        return

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        return


def delete_app_state(path: Path = APP_STATE_PATH) -> None:
    delete_state_from_mysql()
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


def json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def safe_iso_date(value: object) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if value is None:
        return ""
    raw_value = str(value).strip()
    if not raw_value:
        return ""
    try:
        return datetime.fromisoformat(raw_value).date().isoformat()
    except ValueError:
        pass
    compact_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw_value)
    if compact_match:
        year, month, day = compact_match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    chinese_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", raw_value)
    if chinese_match:
        year, month, day = chinese_match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    slash_match = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})", raw_value)
    if slash_match:
        year, month, day = slash_match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    return raw_value
