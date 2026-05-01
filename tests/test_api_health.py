from __future__ import annotations

from api import diagnostics
from api.main import health


class _FakeSettings:
    rag_backend = "auto"
    has_youtube_key = False


def test_health_reports_local_fallback_when_optional_services_are_unconfigured(monkeypatch) -> None:
    monkeypatch.setattr(diagnostics, "load_settings", lambda: _FakeSettings())
    monkeypatch.setattr(diagnostics, "is_mysql_configured", lambda: False)
    monkeypatch.setattr(diagnostics, "is_redis_configured", lambda: False)
    monkeypatch.setattr(diagnostics, "milvus_enabled", lambda: False)
    monkeypatch.setattr(diagnostics, "load_video_cache", lambda path: {})

    payload = health()

    assert payload["status"] == "ok"
    assert payload["rag"]["active_backend"] == "local_vector_fallback"
    assert payload["services"]["mysql"]["configured"] is False
    assert payload["services"]["redis"]["configured"] is False
    assert payload["services"]["youtube"]["configured"] is False


def test_health_marks_configured_mysql_as_degraded_when_connection_fails(monkeypatch) -> None:
    monkeypatch.setattr(diagnostics, "load_settings", lambda: _FakeSettings())
    monkeypatch.setattr(diagnostics, "is_mysql_configured", lambda: True)
    monkeypatch.setattr(diagnostics, "database_url", lambda: "mysql+pymysql://user:pass@127.0.0.1/db")
    monkeypatch.setattr(diagnostics, "is_redis_configured", lambda: False)
    monkeypatch.setattr(diagnostics, "milvus_enabled", lambda: False)
    monkeypatch.setattr(diagnostics, "load_video_cache", lambda path: {})

    def fail_engine(*args, **kwargs):
        raise RuntimeError("no database")

    monkeypatch.setattr("sqlalchemy.create_engine", fail_engine)

    payload = health()

    assert payload["status"] == "degraded"
    assert payload["services"]["mysql"]["configured"] is True
    assert payload["services"]["mysql"]["reachable"] is False


def test_youtube_health_does_not_make_live_api_calls(monkeypatch) -> None:
    monkeypatch.setattr(diagnostics, "load_settings", lambda: type("S", (), {"rag_backend": "auto", "has_youtube_key": True})())
    monkeypatch.setattr(diagnostics, "is_mysql_configured", lambda: False)
    monkeypatch.setattr(diagnostics, "is_redis_configured", lambda: False)
    monkeypatch.setattr(diagnostics, "milvus_enabled", lambda: False)
    monkeypatch.setattr(diagnostics, "load_video_cache", lambda path: {})

    payload = health()

    assert payload["services"]["youtube"]["configured"] is True
    assert payload["services"]["youtube"]["reachable"] is None
    assert "quota" in payload["services"]["youtube"]["detail"]
