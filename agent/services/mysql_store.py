"""Single-user MySQL persistence for FitnessAgent runtime state."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

DEMO_USER_ID = os.getenv("FITNESS_AGENT_USER_ID", "demo-user")


def database_url() -> str:
    return os.getenv("DATABASE_URL") or os.getenv("MYSQL_URL") or ""


def is_mysql_configured() -> bool:
    return bool(database_url())


def load_state_from_mysql(user_id: str = DEMO_USER_ID) -> dict[str, Any]:
    if not is_mysql_configured():
        return {}

    try:
        from sqlalchemy import MetaData, Table, create_engine, select
    except ImportError:
        return {}

    try:
        engine = create_engine(database_url(), pool_pre_ping=True)
        metadata = MetaData()
        users, app_states = _ensure_schema(engine, metadata)
        with engine.begin() as connection:
            connection.execute(
                users.insert().prefix_with("IGNORE"),
                {"id": user_id, "created_at": datetime.utcnow(), "updated_at": datetime.utcnow()},
            )
            row = connection.execute(
                select(app_states.c.state_json).where(app_states.c.user_id == user_id)
            ).first()
    except Exception:
        return {}

    if not row:
        return {}
    try:
        payload = json.loads(row.state_json)
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_state_to_mysql(payload: dict[str, Any], user_id: str = DEMO_USER_ID) -> bool:
    if not is_mysql_configured():
        return False

    try:
        from sqlalchemy import MetaData, create_engine
        from sqlalchemy.dialects.mysql import insert as mysql_insert
    except ImportError:
        return False

    now = datetime.utcnow()
    state_json = json.dumps(payload, ensure_ascii=False)
    try:
        engine = create_engine(database_url(), pool_pre_ping=True)
        metadata = MetaData()
        users, app_states = _ensure_schema(engine, metadata)
        with engine.begin() as connection:
            user_insert = mysql_insert(users).values(id=user_id, created_at=now, updated_at=now)
            connection.execute(
                user_insert.on_duplicate_key_update(updated_at=now)
            )

            state_insert = mysql_insert(app_states).values(
                user_id=user_id,
                state_json=state_json,
                created_at=now,
                updated_at=now,
            )
            connection.execute(
                state_insert.on_duplicate_key_update(
                    state_json=state_json,
                    updated_at=now,
                )
            )
    except Exception:
        return False
    return True


def delete_state_from_mysql(user_id: str = DEMO_USER_ID) -> bool:
    if not is_mysql_configured():
        return False

    try:
        from sqlalchemy import MetaData, create_engine, delete
    except ImportError:
        return False

    try:
        engine = create_engine(database_url(), pool_pre_ping=True)
        metadata = MetaData()
        _, app_states = _ensure_schema(engine, metadata)
        with engine.begin() as connection:
            connection.execute(delete(app_states).where(app_states.c.user_id == user_id))
    except Exception:
        return False
    return True


def _ensure_schema(engine: Any, metadata: Any) -> tuple[Any, Any]:
    from sqlalchemy import Column, DateTime, ForeignKey, String, Table, Text

    users = Table(
        "users",
        metadata,
        Column("id", String(64), primary_key=True),
        Column("created_at", DateTime, nullable=False),
        Column("updated_at", DateTime, nullable=False),
        extend_existing=True,
    )
    app_states = Table(
        "app_states",
        metadata,
        Column("user_id", String(64), ForeignKey("users.id"), primary_key=True),
        Column("state_json", Text(length=16_777_215), nullable=False),
        Column("created_at", DateTime, nullable=False),
        Column("updated_at", DateTime, nullable=False),
        extend_existing=True,
    )
    metadata.create_all(engine)
    return users, app_states
