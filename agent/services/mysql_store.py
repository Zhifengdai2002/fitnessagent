"""Single-user MySQL persistence for FitnessAgent runtime state."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
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
        users, app_states, _, __ = _ensure_schema(engine, metadata)
        with engine.begin() as connection:
            connection.execute(
                users.insert().prefix_with("IGNORE"),
                {"id": user_id, "created_at": _utc_now(), "updated_at": _utc_now()},
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

    now = _utc_now()
    state_json = json.dumps(payload, ensure_ascii=False)
    try:
        engine = create_engine(database_url(), pool_pre_ping=True)
        metadata = MetaData()
        users, app_states, mirror_tables, _ = _ensure_schema(engine, metadata)
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
            _mirror_structured_state(connection, mirror_tables, payload, user_id, now)
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
        _, app_states, mirror_tables, __ = _ensure_schema(engine, metadata)
        with engine.begin() as connection:
            for table in mirror_tables.values():
                connection.execute(delete(table).where(table.c.user_id == user_id))
            connection.execute(delete(app_states).where(app_states.c.user_id == user_id))
    except Exception:
        return False
    return True


def load_video_resource_from_mysql(exercise_name: str, user_id: str = DEMO_USER_ID) -> dict[str, Any] | None:
    """Return a cached video resource for an exercise when MySQL is configured."""

    exercise_key = _video_exercise_key(exercise_name)
    if not exercise_key or not is_mysql_configured():
        return None

    try:
        from sqlalchemy import MetaData, create_engine, select
    except ImportError:
        return None

    try:
        engine = create_engine(database_url(), pool_pre_ping=True)
        metadata = MetaData()
        _, video_cache = _ensure_video_cache_schema(engine, metadata)
        with engine.begin() as connection:
            row = connection.execute(
                select(video_cache).where(
                    video_cache.c.user_id == user_id,
                    video_cache.c.exercise_key == exercise_key,
                )
            ).first()
    except Exception:
        return None

    if not row:
        return None
    payload = _json_load(getattr(row, "payload_json", "") or "")
    if isinstance(payload, dict) and payload:
        return payload
    return {
        "exercise_name": getattr(row, "exercise_name", exercise_name),
        "title": getattr(row, "title", ""),
        "url": getattr(row, "url", ""),
        "source": getattr(row, "source", ""),
        "provider": getattr(row, "provider", ""),
        "video_id": getattr(row, "video_id", ""),
        "channel_title": getattr(row, "channel_title", ""),
        "checked_at": getattr(row, "checked_at", ""),
    }


def save_video_resource_to_mysql(
    exercise_name: str,
    resource: dict[str, Any],
    user_id: str = DEMO_USER_ID,
) -> bool:
    """Upsert a cached video resource for an exercise."""

    exercise_key = _video_exercise_key(exercise_name)
    if not exercise_key or not resource or not is_mysql_configured():
        return False

    try:
        from sqlalchemy import MetaData, create_engine
        from sqlalchemy.dialects.mysql import insert as mysql_insert
    except ImportError:
        return False

    now = _utc_now()
    payload = {
        **resource,
        "exercise_name": str(resource.get("exercise_name") or exercise_name).strip(),
        "checked_at": str(resource.get("checked_at") or now.isoformat()),
    }
    try:
        engine = create_engine(database_url(), pool_pre_ping=True)
        metadata = MetaData()
        users, video_cache = _ensure_video_cache_schema(engine, metadata)
        with engine.begin() as connection:
            user_insert = mysql_insert(users).values(id=user_id, created_at=now, updated_at=now)
            connection.execute(user_insert.on_duplicate_key_update(updated_at=now))

            values = {
                "user_id": user_id,
                "exercise_key": exercise_key,
                "exercise_name": payload["exercise_name"][:255],
                "title": str(payload.get("title") or "")[:500],
                "url": str(payload.get("url") or ""),
                "source": str(payload.get("source") or "")[:64],
                "provider": str(payload.get("provider") or payload.get("source") or "")[:64],
                "video_id": str(payload.get("video_id") or "")[:128] or None,
                "channel_title": str(payload.get("channel_title") or "")[:255] or None,
                "checked_at": str(payload.get("checked_at") or "")[:40] or None,
                "payload_json": _json_dump(payload),
                "created_at": now,
                "updated_at": now,
            }
            cache_insert = mysql_insert(video_cache).values(**values)
            connection.execute(
                cache_insert.on_duplicate_key_update(
                    exercise_name=values["exercise_name"],
                    title=values["title"],
                    url=values["url"],
                    source=values["source"],
                    provider=values["provider"],
                    video_id=values["video_id"],
                    channel_title=values["channel_title"],
                    checked_at=values["checked_at"],
                    payload_json=values["payload_json"],
                    updated_at=now,
                )
            )
    except Exception:
        return False
    return True


def upsert_learned_preferences_to_mysql(
    learned_preferences: dict[str, Any],
    user_id: str = DEMO_USER_ID,
) -> bool:
    """Upsert Layer-3 derived preferences into user_learned_preferences table."""
    if not is_mysql_configured() or not learned_preferences:
        return False
    try:
        from sqlalchemy import MetaData, create_engine
        from sqlalchemy.dialects.mysql import insert as mysql_insert
    except ImportError:
        return False
    now = _utc_now()
    try:
        engine = create_engine(database_url(), pool_pre_ping=True)
        metadata = MetaData()
        users, _, _, prefs_table = _ensure_schema(engine, metadata)
        with engine.begin() as connection:
            user_insert = mysql_insert(users).values(id=user_id, created_at=now, updated_at=now)
            connection.execute(user_insert.on_duplicate_key_update(updated_at=now))
            stmt = mysql_insert(prefs_table).values(
                user_id=user_id,
                liked_exercises_json=_json_dump(learned_preferences.get("liked_exercises", [])),
                difficult_exercises_json=_json_dump(learned_preferences.get("difficult_exercises", [])),
                preferred_focuses_json=_json_dump(learned_preferences.get("preferred_focuses", [])),
                avoided_foods_json=_json_dump(learned_preferences.get("avoided_foods", [])),
                preferred_foods_json=_json_dump(learned_preferences.get("preferred_foods", [])),
                active_injury_areas_json=_json_dump(learned_preferences.get("active_injury_areas", [])),
                updated_at=now,
            )
            connection.execute(stmt.on_duplicate_key_update(
                liked_exercises_json=stmt.inserted.liked_exercises_json,
                difficult_exercises_json=stmt.inserted.difficult_exercises_json,
                preferred_focuses_json=stmt.inserted.preferred_focuses_json,
                avoided_foods_json=stmt.inserted.avoided_foods_json,
                preferred_foods_json=stmt.inserted.preferred_foods_json,
                active_injury_areas_json=stmt.inserted.active_injury_areas_json,
                updated_at=now,
            ))
    except Exception:
        return False
    return True


def load_recent_memory_from_mysql(
    user_id: str = DEMO_USER_ID,
    *,
    since_date: str | None = None,
    injury_window_days: int = 7,
    feedback_limit: int = 14,
    exercise_feedback_limit: int = 30,
) -> dict[str, Any]:
    """Query Layer-2 event tables and Layer-3 preferences for memory_context building.

    Returns a dict with keys matching legacy memory_context collections.
    Returns empty dict if MySQL not configured or on error.
    """
    if not is_mysql_configured():
        return {}
    try:
        from sqlalchemy import MetaData, create_engine, select, desc
    except ImportError:
        return {}

    from datetime import date, timedelta
    today = since_date or date.today().isoformat()

    try:
        engine = create_engine(database_url(), pool_pre_ping=True)
        metadata = MetaData()
        _, _, mirror_tables, prefs_table = _ensure_schema(engine, metadata)

        body_t = mirror_tables["body_metrics"]
        daily_t = mirror_tables["daily_feedback_records"]
        ex_t = mirror_tables["exercise_feedback_records"]
        mem_t = mirror_tables["memory_events"]

        with engine.connect() as connection:
            # body_metrics: last 14 records
            body_rows = connection.execute(
                select(body_t).where(body_t.c.user_id == user_id)
                .order_by(desc(body_t.c.record_date)).limit(14)
            ).fetchall()

            # daily_feedback: last feedback_limit records
            daily_rows = connection.execute(
                select(daily_t).where(daily_t.c.user_id == user_id)
                .order_by(desc(daily_t.c.record_date)).limit(feedback_limit)
            ).fetchall()

            # exercise_feedback: last exercise_feedback_limit records
            ex_rows = connection.execute(
                select(ex_t).where(ex_t.c.user_id == user_id)
                .order_by(desc(ex_t.c.record_date)).limit(exercise_feedback_limit)
            ).fetchall()

            # injury_events from memory_events (active, within window)
            injury_cutoff = (
                date.fromisoformat(today) - timedelta(days=injury_window_days)
            ).isoformat()
            injury_rows = connection.execute(
                select(mem_t).where(
                    mem_t.c.user_id == user_id,
                    mem_t.c.event_type == "injury_events",
                    mem_t.c.event_date >= injury_cutoff,
                    mem_t.c.status == "active",
                )
                .order_by(desc(mem_t.c.event_date))
            ).fetchall()

            # food/training preferences from memory_events
            food_rows = connection.execute(
                select(mem_t).where(
                    mem_t.c.user_id == user_id,
                    mem_t.c.event_type == "food_preferences",
                ).order_by(desc(mem_t.c.event_date)).limit(10)
            ).fetchall()

            training_rows = connection.execute(
                select(mem_t).where(
                    mem_t.c.user_id == user_id,
                    mem_t.c.event_type == "training_preferences",
                ).order_by(desc(mem_t.c.event_date)).limit(10)
            ).fetchall()

            plan_log_rows = connection.execute(
                select(mirror_tables["plan_modification_logs"])
                .where(mirror_tables["plan_modification_logs"].c.user_id == user_id)
                .order_by(desc(mirror_tables["plan_modification_logs"].c.event_date))
                .limit(12)
            ).fetchall()

            # Layer-3 learned preferences
            prefs_row = connection.execute(
                select(prefs_table).where(prefs_table.c.user_id == user_id)
            ).first()

    except Exception:
        return {}

    def _row_payload(row: Any) -> dict[str, Any]:
        payload = _json_load(getattr(row, "payload_json", "") or "")
        return payload if isinstance(payload, dict) else {}

    def _plan_log_dict(row: Any) -> dict[str, Any]:
        return {
            "date": getattr(row, "event_date", ""),
            "action_type": getattr(row, "action_type", ""),
            "summary": getattr(row, "summary", ""),
        }

    active_injuries = [_row_payload(r) for r in injury_rows]

    learned_preferences: dict[str, Any] = {}
    if prefs_row:
        learned_preferences = {
            "liked_exercises":    _json_load(getattr(prefs_row, "liked_exercises_json", "[]") or "[]"),
            "difficult_exercises": _json_load(getattr(prefs_row, "difficult_exercises_json", "[]") or "[]"),
            "preferred_focuses":  _json_load(getattr(prefs_row, "preferred_focuses_json", "[]") or "[]"),
            "avoided_foods":      _json_load(getattr(prefs_row, "avoided_foods_json", "[]") or "[]"),
            "preferred_foods":    _json_load(getattr(prefs_row, "preferred_foods_json", "[]") or "[]"),
            "active_injury_areas": [
                str(inj.get("area") or inj.get("injury_area") or "")
                for inj in active_injuries
                if isinstance(inj, dict) and (inj.get("area") or inj.get("injury_area"))
            ],
        }

    return {
        "active_injuries": active_injuries,
        "recent_body_metrics": [
            {
                "date": getattr(r, "record_date", ""),
                "weight_kg": getattr(r, "weight_kg", None),
                "body_fat_pct": getattr(r, "body_fat_pct", None),
            }
            for r in reversed(body_rows)
        ],
        "recent_daily_feedback": [
            {
                "date": getattr(r, "record_date", ""),
                "status": getattr(r, "workout_status", ""),
                "plan_focus": getattr(r, "focus", ""),
                "feeling": getattr(r, "workout_feeling", ""),
                "emoji": getattr(r, "feeling_emoji", ""),
                "completed_actions": _json_load(getattr(r, "completed_actions_json", "[]") or "[]"),
            }
            for r in reversed(daily_rows)
        ],
        "recent_exercise_feedback": [
            {
                "date": getattr(r, "record_date", ""),
                "exercise_name": getattr(r, "exercise_name", ""),
                "focus": getattr(r, "focus", ""),
                "status": getattr(r, "workout_status", ""),
                "feeling_emoji": getattr(r, "feeling_emoji", ""),
                "workout_feeling": getattr(r, "workout_feeling", ""),
            }
            for r in reversed(ex_rows)
        ],
        "recent_food_preferences":     [_row_payload(r) for r in reversed(food_rows)],
        "recent_training_preferences": [_row_payload(r) for r in reversed(training_rows)],
        "recent_plan_modifications":   [_plan_log_dict(r) for r in reversed(plan_log_rows)],
        "learned_preferences":         learned_preferences,
    }


def _ensure_video_cache_schema(engine: Any, metadata: Any) -> tuple[Any, Any]:
    from sqlalchemy import Column, DateTime, ForeignKey, String, Table, Text

    users = Table(
        "users",
        metadata,
        Column("id", String(64), primary_key=True),
        Column("created_at", DateTime, nullable=False),
        Column("updated_at", DateTime, nullable=False),
        extend_existing=True,
    )
    video_cache = Table(
        "video_cache",
        metadata,
        Column("user_id", String(64), ForeignKey("users.id"), primary_key=True),
        Column("exercise_key", String(255), primary_key=True),
        Column("exercise_name", String(255), nullable=False),
        Column("title", String(500), nullable=True),
        Column("url", Text, nullable=False),
        Column("source", String(64), nullable=False),
        Column("provider", String(64), nullable=True),
        Column("video_id", String(128), nullable=True),
        Column("channel_title", String(255), nullable=True),
        Column("checked_at", String(40), nullable=True),
        Column("payload_json", Text(length=16_777_215), nullable=False),
        Column("created_at", DateTime, nullable=False),
        Column("updated_at", DateTime, nullable=False),
        extend_existing=True,
    )
    metadata.create_all(engine)
    return users, video_cache


def _ensure_schema(engine: Any, metadata: Any) -> tuple[Any, Any, dict[str, Any], Any]:
    from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Table, Text

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
    body_metrics = Table(
        "body_metrics",
        metadata,
        Column("user_id", String(64), ForeignKey("users.id"), primary_key=True),
        Column("record_date", String(10), primary_key=True),
        Column("weight_kg", Float, nullable=True),
        Column("body_fat_pct", Float, nullable=True),
        Column("source", String(32), nullable=False),
        Column("created_at", DateTime, nullable=False),
        Column("updated_at", DateTime, nullable=False),
        extend_existing=True,
    )
    daily_feedback_records = Table(
        "daily_feedback_records",
        metadata,
        Column("user_id", String(64), ForeignKey("users.id"), primary_key=True),
        Column("record_date", String(10), primary_key=True),
        Column("cycle_number", Integer, nullable=True),
        Column("feeling_emoji", String(16), nullable=True),
        Column("workout_feeling", Text, nullable=True),
        Column("workout_status", String(32), nullable=False),
        Column("focus", String(255), nullable=True),
        Column("completed_actions_json", Text(length=16_777_215), nullable=False),
        Column("created_at", DateTime, nullable=False),
        Column("updated_at", DateTime, nullable=False),
        extend_existing=True,
    )
    exercise_feedback_records = Table(
        "exercise_feedback_records",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("user_id", String(64), ForeignKey("users.id"), nullable=False, index=True),
        Column("record_date", String(10), nullable=True, index=True),
        Column("cycle_number", Integer, nullable=True),
        Column("sequence_index", Integer, nullable=False),
        Column("exercise_name", String(255), nullable=True),
        Column("focus", String(255), nullable=True),
        Column("sets_count", Integer, nullable=True),
        Column("reps", String(64), nullable=True),
        Column("workout_status", String(32), nullable=False),
        Column("feeling_emoji", String(16), nullable=True),
        Column("workout_feeling", Text, nullable=True),
        Column("injury_areas_json", Text, nullable=False),
        Column("source", String(32), nullable=False),
        Column("created_at", DateTime, nullable=False),
        extend_existing=True,
    )
    chat_messages = Table(
        "chat_messages",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("user_id", String(64), ForeignKey("users.id"), nullable=False, index=True),
        Column("sequence_index", Integer, nullable=False),
        Column("role", String(32), nullable=False),
        Column("content", Text, nullable=False),
        Column("created_at", DateTime, nullable=False),
        extend_existing=True,
    )
    plan_modification_logs = Table(
        "plan_modification_logs",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("user_id", String(64), ForeignKey("users.id"), nullable=False, index=True),
        Column("event_date", String(10), nullable=True),
        Column("action_type", String(64), nullable=False),
        Column("summary", Text, nullable=True),
        Column("injury_areas_json", Text, nullable=False),
        Column("recorded_at", String(40), nullable=True),
        Column("created_at", DateTime, nullable=False),
        extend_existing=True,
    )
    memory_events = Table(
        "memory_events",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("user_id", String(64), ForeignKey("users.id"), nullable=False, index=True),
        Column("event_type", String(64), nullable=False),
        Column("event_date", String(10), nullable=True),
        Column("event_key", String(128), nullable=True),
        Column("status", String(32), nullable=True),
        Column("payload_json", Text(length=16_777_215), nullable=False),
        Column("created_at", DateTime, nullable=False),
        extend_existing=True,
    )
    user_learned_preferences = Table(
        "user_learned_preferences",
        metadata,
        Column("user_id", String(64), ForeignKey("users.id"), primary_key=True),
        Column("liked_exercises_json", Text, nullable=False, default="[]"),
        Column("difficult_exercises_json", Text, nullable=False, default="[]"),
        Column("preferred_focuses_json", Text, nullable=False, default="[]"),
        Column("avoided_foods_json", Text, nullable=False, default="[]"),
        Column("preferred_foods_json", Text, nullable=False, default="[]"),
        Column("active_injury_areas_json", Text, nullable=False, default="[]"),
        Column("updated_at", DateTime, nullable=False),
        extend_existing=True,
    )
    metadata.create_all(engine)
    return users, app_states, {
        "body_metrics": body_metrics,
        "daily_feedback_records": daily_feedback_records,
        "exercise_feedback_records": exercise_feedback_records,
        "chat_messages": chat_messages,
        "plan_modification_logs": plan_modification_logs,
        "memory_events": memory_events,
    }, user_learned_preferences


def _mirror_structured_state(
    connection: Any,
    tables: dict[str, Any],
    payload: dict[str, Any],
    user_id: str,
    now: datetime,
) -> None:
    from sqlalchemy.dialects.mysql import insert as mysql_insert

    rows_by_table = _structured_rows_from_payload(payload, user_id, now)

    # Tables with natural composite PK → upsert (ON DUPLICATE KEY UPDATE)
    for tname in ("body_metrics", "daily_feedback_records"):
        table = tables.get(tname)
        if table is None:
            continue
        for row in rows_by_table.get(tname, []):
            stmt = mysql_insert(table).values(**row)
            # build update dict from all non-PK columns
            update_cols = {c.name: getattr(stmt.inserted, c.name)
                          for c in table.columns
                          if c.name not in ("user_id", "record_date")}
            connection.execute(stmt.on_duplicate_key_update(**update_cols))

    # Append-only tables (auto-increment PK) → insert only rows not already present
    for tname in ("exercise_feedback_records", "plan_modification_logs",
                  "chat_messages", "memory_events"):
        table = tables.get(tname)
        if table is None:
            continue
        new_rows = rows_by_table.get(tname, [])
        if not new_rows:
            continue
        _insert_new_rows_only(connection, table, new_rows, user_id, tname)


def _insert_new_rows_only(
    connection: Any,
    table: Any,
    rows: list[dict[str, Any]],
    user_id: str,
    table_name: str,
) -> None:
    """Insert only rows that don't already exist, using table-specific dedup keys."""
    from sqlalchemy import select, and_

    dedup_cols: dict[str, list[str]] = {
        "exercise_feedback_records": ["record_date", "sequence_index"],
        "plan_modification_logs":    ["event_date", "action_type", "recorded_at"],
        "chat_messages":             ["sequence_index"],
        "memory_events":             ["event_type", "event_key"],
    }
    key_cols = dedup_cols.get(table_name, [])
    if not key_cols:
        if rows:
            connection.execute(table.insert(), rows)
        return

    # Fetch existing dedup keys for this user
    existing_rows = connection.execute(
        select(*[table.c[c] for c in key_cols]).where(table.c.user_id == user_id)
    ).fetchall()
    existing_keys = {tuple(str(getattr(r, c) or "") for c in key_cols) for r in existing_rows}

    new_rows = [
        r for r in rows
        if tuple(str(r.get(c) or "") for c in key_cols) not in existing_keys
    ]
    if new_rows:
        connection.execute(table.insert(), new_rows)


def _structured_rows_from_payload(
    payload: dict[str, Any],
    user_id: str,
    now: datetime,
) -> dict[str, list[dict[str, Any]]]:
    memory_store = _as_dict(payload.get("memory_store"))
    agent_result = _as_dict(payload.get("agent_result"))
    result_memory = _as_dict(agent_result.get("memory_store"))
    memory_store = {**result_memory, **memory_store}

    daily_history = _coerce_list(payload.get("daily_history"))
    daily_history.extend(_coerce_list(agent_result.get("daily_history")))
    daily_feedback_memory = _coerce_list(memory_store.get("daily_feedback_records"))
    daily_history.extend(daily_feedback_memory)
    exercise_feedback_items = _coerce_list(memory_store.get("exercise_feedback_records"))
    exercise_feedback_items.extend(_exercise_feedback_items_from_daily_history(daily_history))

    body_metric_items = _coerce_list(memory_store.get("body_metrics"))
    body_metric_items.extend(daily_history)
    body_metric_items.extend(_coerce_list(agent_result.get("state_history")))

    return {
        "body_metrics": _body_metric_rows(body_metric_items, user_id, now),
        "daily_feedback_records": _daily_feedback_rows(daily_history, user_id, now),
        "exercise_feedback_records": _exercise_feedback_rows(exercise_feedback_items, user_id, now),
        "chat_messages": _chat_message_rows(
            _coerce_list(payload.get("assistant_chat_messages")),
            user_id,
            now,
        ),
        "plan_modification_logs": _plan_log_rows(
            _coerce_list(memory_store.get("plan_modification_logs")),
            user_id,
            now,
        ),
        "memory_events": _memory_event_rows(memory_store, user_id, now),
    }


def _body_metric_rows(items: list[Any], user_id: str, now: datetime) -> list[dict[str, Any]]:
    by_date: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        record_date = _safe_date_text(_first_present(item, ["date", "record_date", "feedback_date"]))
        if not record_date:
            continue
        weight = _safe_float(_first_present(item, ["weight_kg", "current_weight_kg", "weight"]))
        body_fat = _safe_float(_first_present(item, ["body_fat_pct", "current_body_fat_pct", "body_fat"]))
        if weight is None and body_fat is None:
            continue
        by_date[record_date] = {
            "user_id": user_id,
            "record_date": record_date,
            "weight_kg": weight,
            "body_fat_pct": body_fat,
            "source": str(item.get("source") or "daily_feedback")[:32],
            "created_at": now,
            "updated_at": now,
        }
    return list(by_date.values())


def _daily_feedback_rows(items: list[Any], user_id: str, now: datetime) -> list[dict[str, Any]]:
    by_date: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        record_date = _safe_date_text(_first_present(item, ["date", "record_date", "feedback_date"]))
        if not record_date:
            continue
        feedback = _as_dict(item.get("feedback"))
        completed_actions = _first_present(
            item,
            ["completed_actions", "actions", "completed_workouts"],
            fallback=[],
        )
        focus = str(_first_present(item, ["plan_focus", "focus", "workout_focus", "title"], fallback="") or "")
        by_date[record_date] = {
            "user_id": user_id,
            "record_date": record_date,
            "cycle_number": _safe_int(item.get("cycle_number")),
            "feeling_emoji": str(
                _first_present(feedback, ["emoji", "feeling_emoji"], fallback=item.get("feeling_emoji") or "")
            )[:16] or None,
            "workout_feeling": str(
                _first_present(
                    feedback,
                    ["workout_feeling", "notes", "summary"],
                    fallback=_first_present(item, ["workout_feeling", "notes", "summary"], fallback=""),
                )
                or ""
            ),
            "workout_status": str(item.get("status") or _infer_workout_status(item, focus))[:32],
            "focus": focus[:255] or None,
            "completed_actions_json": _json_dump(completed_actions),
            "created_at": now,
            "updated_at": now,
        }
    return list(by_date.values())


def _exercise_feedback_rows(items: list[Any], user_id: str, now: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        record_date = _safe_date_text(_first_present(item, ["date", "record_date", "feedback_date"]))
        exercise_name = str(
            _first_present(item, ["exercise_name", "name", "action"], fallback="")
            or ""
        ).strip()
        status = str(item.get("status") or item.get("workout_status") or "completed")[:32]
        dedupe_key = (record_date or "", exercise_name, status)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        rows.append(
            {
                "user_id": user_id,
                "record_date": record_date,
                "cycle_number": _safe_int(item.get("cycle_number")),
                "sequence_index": _safe_int(item.get("sequence_index")) or index,
                "exercise_name": exercise_name[:255] or None,
                "focus": str(item.get("focus") or item.get("plan_focus") or "")[:255] or None,
                "sets_count": _safe_int(item.get("sets") or item.get("sets_count")),
                "reps": str(item.get("reps") or "")[:64] or None,
                "workout_status": status,
                "feeling_emoji": str(item.get("feeling_emoji") or item.get("emoji") or "")[:16] or None,
                "workout_feeling": str(item.get("workout_feeling") or item.get("feeling") or item.get("notes") or ""),
                "injury_areas_json": _json_dump(item.get("injury_areas") or []),
                "source": str(item.get("source") or "daily_feedback")[:32],
                "created_at": now,
            }
        )
    return rows


def _exercise_feedback_items_from_daily_history(items: list[Any]) -> list[dict[str, Any]]:
    feedback_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        record_date = _safe_date_text(_first_present(item, ["date", "record_date", "feedback_date"]))
        if not record_date:
            continue
        feedback = _as_dict(item.get("feedback"))
        completed_plan = _as_dict(item.get("completed_plan"))
        exercises = _coerce_list(completed_plan.get("exercises"))
        completed_actions = _coerce_list(
            _first_present(item, ["completed_actions", "actions", "completed_workouts"], fallback=[])
        )
        if not exercises and completed_actions:
            exercises = [
                action if isinstance(action, dict) else {"name": action}
                for action in completed_actions
            ]
        status = str(item.get("status") or _infer_workout_status(item, str(item.get("plan_focus") or "")))
        if not exercises and status in {"cancelled", "no_scheduled"}:
            exercises = [{"name": "", "sets": None, "reps": ""}]
        for index, exercise in enumerate(exercises):
            if not isinstance(exercise, dict):
                continue
            feedback_items.append(
                {
                    "date": record_date,
                    "cycle_number": item.get("cycle_number") or completed_plan.get("cycle_number"),
                    "sequence_index": index,
                    "exercise_name": exercise.get("name") or "",
                    "focus": item.get("plan_focus") or completed_plan.get("focus") or "",
                    "sets": exercise.get("sets"),
                    "reps": exercise.get("reps"),
                    "status": "cancelled" if status == "cancelled" else status,
                    "feeling_emoji": _first_present(feedback, ["emoji", "feeling_emoji"], fallback=item.get("feeling_emoji") or ""),
                    "workout_feeling": _first_present(
                        feedback,
                        ["workout_feeling", "notes", "summary"],
                        fallback=_first_present(item, ["workout_feeling", "notes", "summary"], fallback=""),
                    ),
                    "injury_areas": feedback.get("injury_areas") or item.get("injury_areas") or [],
                    "source": "daily_history",
                }
            )
    return feedback_items


def _chat_message_rows(messages: list[Any], user_id: str, now: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        rows.append(
            {
                "user_id": user_id,
                "sequence_index": index,
                "role": str(message.get("role") or "assistant")[:32],
                "content": content,
                "created_at": now,
            }
        )
    return rows


def _plan_log_rows(items: list[Any], user_id: str, now: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary") or item.get("message") or item.get("user_message") or "").strip()
        rows.append(
            {
                "user_id": user_id,
                "event_date": _safe_date_text(_first_present(item, ["date", "event_date", "active_date"])),
                "action_type": str(item.get("action_type") or item.get("tool_name") or "plan_update")[:64],
                "summary": summary,
                "injury_areas_json": _json_dump(item.get("injury_areas") or item.get("injury_areas_json") or []),
                "recorded_at": str(item.get("recorded_at") or item.get("created_at") or "")[:40] or None,
                "created_at": now,
            }
        )
    return rows


def _memory_event_rows(memory_store: dict[str, Any], user_id: str, now: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mirrored_collections = {
        "body_metrics",
        "daily_feedback_records",
        "exercise_feedback_records",
        "plan_modification_logs",
    }
    for collection, items in memory_store.items():
        if collection in mirrored_collections:
            continue
        for index, item in enumerate(_coerce_list(items)):
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "user_id": user_id,
                    "event_type": str(collection)[:64],
                    "event_date": _safe_date_text(_first_present(item, ["date", "event_date", "record_date"])),
                    "event_key": str(
                        _first_present(item, ["id", "area", "food", "preference", "key"], fallback=index)
                    )[:128],
                    "status": str(item.get("status") or "")[:32] or None,
                    "payload_json": _json_dump(item),
                    "created_at": now,
                }
            )
    return rows


def _infer_workout_status(item: dict[str, Any], focus: str) -> str:
    completed_plan = _as_dict(item.get("completed_plan"))
    if item.get("is_cancelled") or completed_plan.get("is_cancelled"):
        return "cancelled"
    if not completed_plan and "no scheduled" in focus.lower():
        return "no_scheduled"
    return "completed"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _first_present(mapping: dict[str, Any], keys: list[str], fallback: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return fallback


def _safe_date_text(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).date().isoformat()
    except ValueError:
        pass
    return raw[:10]


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_dump(value: Any) -> str:
    return json.dumps(value if value is not None else [], ensure_ascii=False)


def _json_load(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _video_exercise_key(exercise_name: str) -> str:
    return "_".join(str(exercise_name or "").strip().lower().replace("-", " ").split())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)
