from __future__ import annotations

from agent.services import video_cache
from agent.services.video_backfill import backfill_video_resources, resolve_exercise_video_resource
from agent.tools import exercise_tool


def test_video_cache_json_fallback_roundtrip(monkeypatch, tmp_path) -> None:
    cache_path = tmp_path / "video_cache.json"
    monkeypatch.setattr(video_cache, "load_video_resource_from_mysql", lambda exercise_name: None)
    monkeypatch.setattr(video_cache, "save_video_resource_to_mysql", lambda exercise_name, resource: False)

    saved = video_cache.save_cached_video_resource(
        "Incline Push-Up",
        {
            "title": "Incline Push-Up proper form",
            "url": "https://www.youtube.com/watch?v=abc123",
            "source": "youtube_api",
            "provider": "youtube",
            "video_id": "abc123",
        },
        path=cache_path,
    )
    loaded = video_cache.get_cached_video_resource("incline push up", path=cache_path)

    assert saved["exercise_name"] == "Incline Push-Up"
    assert loaded
    assert loaded["url"] == "https://www.youtube.com/watch?v=abc123"
    assert loaded["video_id"] == "abc123"
    assert loaded["cache_status"] == "hit"


def test_build_video_resources_uses_cache_before_youtube(monkeypatch) -> None:
    monkeypatch.setattr(exercise_tool, "load_all_exercise_db", lambda: [{"name": "No Local Media"}])
    monkeypatch.setattr(
        "agent.services.video_backfill.get_cached_video_resource",
        lambda exercise_name: {
            "exercise_name": exercise_name,
            "title": "Cached demo",
            "url": "https://www.youtube.com/watch?v=cached",
            "source": "youtube_api",
        },
    )

    def fail_youtube_lookup(exercise_name: str) -> None:
        raise AssertionError("YouTube lookup should not run when cache is present")

    monkeypatch.setattr("agent.services.video_backfill.search_youtube_video", fail_youtube_lookup)

    resources = exercise_tool.build_video_resources(["No Local Media"])

    assert resources == [
        {
            "exercise_name": "No Local Media",
            "title": "Cached demo",
            "url": "https://www.youtube.com/watch?v=cached",
            "source": "youtube_api",
            "cache_status": "hit",
        }
    ]


def test_video_backfill_searches_youtube_and_saves_cache(monkeypatch) -> None:
    saved: dict[str, object] = {}

    monkeypatch.setattr("agent.services.video_backfill.get_cached_video_resource", lambda exercise_name: None)
    monkeypatch.setattr(
        "agent.services.video_backfill.search_youtube_video",
        lambda exercise_name: {
            "title": f"{exercise_name} proper form",
            "url": "https://www.youtube.com/watch?v=yt123",
            "source": "youtube_api",
            "provider": "youtube",
            "video_id": "yt123",
        },
    )

    def fake_save(exercise_name: str, resource: dict) -> dict:
        saved["exercise_name"] = exercise_name
        saved["resource"] = resource
        return {**resource, "exercise_name": exercise_name}

    monkeypatch.setattr("agent.services.video_backfill.save_cached_video_resource", fake_save)

    resource = resolve_exercise_video_resource("Machine Chest Press")

    assert resource
    assert resource["url"] == "https://www.youtube.com/watch?v=yt123"
    assert saved["exercise_name"] == "Machine Chest Press"


def test_backfill_video_resources_ignores_local_media_and_uses_youtube_api(monkeypatch) -> None:
    looked_up: list[str] = []

    def fake_youtube_lookup(exercise_name: str) -> dict[str, str]:
        looked_up.append(exercise_name)
        return {
            "title": f"{exercise_name} proper form",
            "url": "https://www.youtube.com/watch?v=ytlocalignored",
            "source": "youtube_api",
            "provider": "youtube",
            "video_id": "ytlocalignored",
        }

    monkeypatch.setattr("agent.services.video_backfill.get_cached_video_resource", lambda exercise_name: None)
    monkeypatch.setattr("agent.services.video_backfill.search_youtube_video", fake_youtube_lookup)
    monkeypatch.setattr(
        "agent.services.video_backfill.save_cached_video_resource",
        lambda exercise_name, resource: {**resource, "exercise_name": exercise_name},
    )

    resources = backfill_video_resources(
        ["Imported Shoulder Raise"],
        local_resources={
            "imported_shoulder_raise": {
                "name": "Imported Shoulder Raise",
                "media_url": "https://example.com/shoulder.mp4",
                "source": "wger",
            }
        },
    )

    assert looked_up == ["Imported Shoulder Raise"]
    assert resources[0]["url"] == "https://www.youtube.com/watch?v=ytlocalignored"
    assert resources[0]["source"] == "youtube_api"


def test_non_youtube_api_cache_is_ignored(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.services.video_backfill.get_cached_video_resource",
        lambda exercise_name: {
            "exercise_name": exercise_name,
            "title": "Old wger demo",
            "url": "https://example.com/wger.mp4",
            "source": "wger",
            "provider": "wger",
        },
    )
    monkeypatch.setattr(
        "agent.services.video_backfill.search_youtube_video",
        lambda exercise_name: {
            "title": f"{exercise_name} proper form",
            "url": "https://www.youtube.com/watch?v=freshapi",
            "source": "youtube_api",
            "provider": "youtube",
            "video_id": "freshapi",
        },
    )
    monkeypatch.setattr(
        "agent.services.video_backfill.save_cached_video_resource",
        lambda exercise_name, resource: {**resource, "exercise_name": exercise_name},
    )

    resource = resolve_exercise_video_resource("Reverse Lunge")

    assert resource
    assert resource["source"] == "youtube_api"
    assert resource["url"] == "https://www.youtube.com/watch?v=freshapi"
