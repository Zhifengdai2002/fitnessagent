from __future__ import annotations

from agent.services import persistence
from agent.services import video_cache
from agent.rag import retriever


def test_load_app_state_prefers_redis(monkeypatch, tmp_path) -> None:
    path = tmp_path / "app_state.json"
    path.write_text('{"active_date": "2026-04-30"}', encoding="utf-8")

    monkeypatch.setattr(persistence, "load_state_from_redis", lambda user_id: {"active_date": "2026-05-01"})
    monkeypatch.setattr(persistence, "is_mysql_configured", lambda: True)

    def fail_mysql_load() -> None:
        raise AssertionError("MySQL should not be read when Redis has the active state")

    monkeypatch.setattr(persistence, "load_state_from_mysql", fail_mysql_load)

    assert persistence.load_app_state(path)["active_date"] == "2026-05-01"


def test_load_app_state_warms_redis_from_mysql(monkeypatch, tmp_path) -> None:
    warmed: dict[str, object] = {}
    payload = {"active_date": "2026-05-02"}

    monkeypatch.setattr(persistence, "load_state_from_redis", lambda user_id: {})
    monkeypatch.setattr(persistence, "is_mysql_configured", lambda: True)
    monkeypatch.setattr(persistence, "load_state_from_mysql", lambda: payload)
    monkeypatch.setattr(
        persistence,
        "save_state_to_redis",
        lambda saved_payload, user_id: warmed.update({"payload": saved_payload, "user_id": user_id}) or True,
    )

    assert persistence.load_app_state(tmp_path / "missing.json") == payload
    assert warmed["payload"] == payload


def test_save_and_delete_app_state_updates_redis(monkeypatch, tmp_path) -> None:
    saved: dict[str, object] = {}
    deleted: list[str] = []

    monkeypatch.setattr(
        persistence,
        "save_state_to_redis",
        lambda payload, user_id: saved.update({"payload": payload, "user_id": user_id}) or True,
    )
    monkeypatch.setattr(persistence, "save_state_to_mysql", lambda payload: True)
    monkeypatch.setattr(persistence, "delete_state_from_redis", lambda user_id: deleted.append(user_id) or True)
    monkeypatch.setattr(persistence, "delete_state_from_mysql", lambda: True)

    persistence.save_app_state({"active_date": "2026-05-03"}, tmp_path / "app_state.json")
    persistence.delete_app_state(tmp_path / "app_state.json")

    assert saved["payload"]["active_date"] == "2026-05-03"
    assert deleted == [persistence.DEMO_USER_ID]


def test_video_cache_prefers_redis_and_warms_from_mysql(monkeypatch, tmp_path) -> None:
    warmed: dict[str, object] = {}
    monkeypatch.setattr(
        video_cache,
        "load_cache_item",
        lambda namespace, key: {"title": "Redis video", "url": "https://example.com/redis", "source": "redis"},
    )

    def fail_mysql_load(exercise_name: str) -> None:
        raise AssertionError("MySQL should not be read when Redis has the video cache")

    monkeypatch.setattr(video_cache, "load_video_resource_from_mysql", fail_mysql_load)

    resource = video_cache.get_cached_video_resource("Incline Push Up", tmp_path / "video_cache.json")

    assert resource["url"] == "https://example.com/redis"
    assert resource["cache_status"] == "hit"

    monkeypatch.setattr(video_cache, "load_cache_item", lambda namespace, key: None)
    monkeypatch.setattr(
        video_cache,
        "load_video_resource_from_mysql",
        lambda exercise_name: {"title": "MySQL video", "url": "https://example.com/mysql", "source": "youtube_api"},
    )
    monkeypatch.setattr(
        video_cache,
        "save_cache_item",
        lambda namespace, key, payload: warmed.update({"namespace": namespace, "key": key, "payload": payload}) or True,
    )

    resource = video_cache.get_cached_video_resource("Incline Push Up", tmp_path / "video_cache.json")

    assert resource["url"] == "https://example.com/mysql"
    assert warmed["namespace"] == video_cache.VIDEO_CACHE_NAMESPACE


def test_rag_search_prefers_redis_cache(monkeypatch) -> None:
    cached_result = {
        "score": 0.9,
        "document": {
            "metadata": {"name": "Cached Row"},
            "raw": {"name": "Cached Row", "source": "wger"},
        },
    }

    monkeypatch.setattr(retriever, "milvus_enabled", lambda: False)
    monkeypatch.setattr(
        retriever,
        "load_cache_item",
        lambda namespace, key: {"results": [cached_result]},
    )

    def fail_search(*args, **kwargs) -> None:
        raise AssertionError("Local vector search should not run when Redis has the RAG result")

    monkeypatch.setattr(retriever, "search_index", fail_search)

    results = retriever._search_exercise_documents("row", limit=5)

    assert results == [cached_result]


def test_rag_search_warms_redis_cache(monkeypatch) -> None:
    warmed: dict[str, object] = {}
    local_result = {
        "score": 0.8,
        "document": {
            "metadata": {"name": "Local Row"},
            "raw": {"name": "Local Row", "source": "local"},
        },
    }

    monkeypatch.setattr(retriever, "milvus_enabled", lambda: False)
    monkeypatch.setattr(retriever, "load_cache_item", lambda namespace, key: None)
    monkeypatch.setattr(retriever, "exercise_index", lambda: {"documents": []})
    monkeypatch.setattr(retriever, "search_index", lambda query, index, limit: [local_result])
    monkeypatch.setattr(retriever, "redis_cache_ttl_seconds", lambda: 123)
    monkeypatch.setattr(
        retriever,
        "save_cache_item",
        lambda namespace, key, payload, ttl_seconds=None: warmed.update(
            {"namespace": namespace, "key": key, "payload": payload, "ttl_seconds": ttl_seconds}
        )
        or True,
    )

    results = retriever._search_exercise_documents("row", limit=5)

    assert results == [local_result]
    assert warmed["namespace"] == retriever.RAG_SEARCH_CACHE_NAMESPACE
    assert warmed["payload"] == {"results": [local_result]}
    assert warmed["ttl_seconds"] == 123


def test_milvus_rag_search_cache_skips_collection_warmup(monkeypatch) -> None:
    cached_result = {
        "score": 0.95,
        "document": {
            "metadata": {"name": "Milvus Cable Row"},
            "raw": {"name": "Milvus Cable Row", "source": "wger"},
        },
    }

    monkeypatch.setattr(retriever, "milvus_enabled", lambda: True)
    monkeypatch.setattr(retriever, "load_cache_item", lambda namespace, key: {"results": [cached_result]})

    def fail_warmup(*args, **kwargs) -> None:
        raise AssertionError("Milvus collection warmup should not run when Redis has the RAG result")

    monkeypatch.setattr(retriever, "build_exercise_documents", fail_warmup)
    monkeypatch.setattr(retriever, "ensure_milvus_collection", fail_warmup)
    monkeypatch.setattr(retriever, "search_milvus_collection", fail_warmup)

    results = retriever._search_exercise_documents("cable row", limit=5)

    assert results == [cached_result]
