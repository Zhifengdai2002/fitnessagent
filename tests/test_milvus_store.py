from __future__ import annotations

import json

from agent.rag import milvus_store
from agent.rag.retriever import retrieve_exercises


class _FakeSettings:
    rag_backend = "milvus"
    milvus_uri = "http://localhost:19530"
    milvus_token = ""
    milvus_exercise_collection = "fitness_exercises_test"
    milvus_food_collection = "fitness_foods_test"

    @property
    def has_milvus(self) -> bool:
        return True


class _FakeMilvusClient:
    def __init__(self) -> None:
        self.collections: dict[str, list[dict]] = {}

    def create_schema(self, **kwargs):
        class _Schema:
            def __init__(self) -> None:
                self.fields = []

            def add_field(self, **field_kwargs) -> None:
                self.fields.append(field_kwargs)

        return _Schema()

    def prepare_index_params(self):
        class _IndexParams:
            def __init__(self) -> None:
                self.indexes = []

            def add_index(self, **index_kwargs) -> None:
                self.indexes.append(index_kwargs)

        return _IndexParams()

    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, **kwargs) -> None:
        self.collections[collection_name] = []

    def drop_collection(self, collection_name: str) -> None:
        self.collections.pop(collection_name, None)

    def insert(self, collection_name: str, data: list[dict]) -> None:
        self.collections.setdefault(collection_name, []).extend(data)

    def search(self, collection_name: str, **kwargs) -> list[list[dict]]:
        rows = self.collections.get(collection_name, [])
        hits = [
            {
                "distance": 0.9,
                "entity": {"document_json": row["document_json"]},
            }
            for row in rows[: kwargs.get("limit", 10)]
        ]
        return [hits]


def test_milvus_build_and_search_with_fake_client(monkeypatch) -> None:
    fake_client = _FakeMilvusClient()
    document = {
        "title": "Machine Chest Press",
        "text": "beginner chest press machine horizontal push",
        "metadata": {"name": "Machine Chest Press", "source": "curated_rag"},
        "raw": {"name": "Machine Chest Press", "source": "curated_rag"},
    }

    monkeypatch.setattr(milvus_store, "load_settings", lambda: _FakeSettings())
    monkeypatch.setattr(milvus_store, "milvus_client", lambda: fake_client)

    assert milvus_store.milvus_enabled()
    assert milvus_store.build_milvus_collection(
        [document],
        collection_name="fitness_exercises_test",
    )

    results = milvus_store.search_milvus_collection(
        "beginner chest press",
        collection_name="fitness_exercises_test",
        limit=3,
    )

    assert results[0]["score"] == 0.9
    assert results[0]["document"]["metadata"]["name"] == "Machine Chest Press"
    assert json.loads(fake_client.collections["fitness_exercises_test"][0]["document_json"])["title"] == "Machine Chest Press"


def test_retriever_prefers_milvus_results_when_available(monkeypatch) -> None:
    monkeypatch.setattr("agent.rag.retriever.milvus_enabled", lambda: True)
    monkeypatch.setattr("agent.rag.retriever.ensure_milvus_collection", lambda documents, collection_name: True)
    monkeypatch.setattr("agent.rag.retriever.exercise_collection_name", lambda: "fitness_exercises_test")
    monkeypatch.setattr(
        "agent.rag.retriever.search_milvus_collection",
        lambda query, collection_name, limit: [
            {
                "score": 0.8,
                "document": {
                    "metadata": {
                        "name": "Milvus Shoulder Press",
                        "source": "curated_rag",
                        "focus_tags": ["upper_shoulders"],
                        "difficulty": "beginner",
                    },
                    "raw": {
                        "name": "Milvus Shoulder Press",
                        "source": "curated_rag",
                        "focus_tags": ["upper_shoulders"],
                        "difficulty": "beginner",
                    },
                },
            }
        ],
    )

    results = retrieve_exercises(
        query="beginner shoulder press",
        focus="upper_shoulders",
        level="beginner",
        limit=1,
    )

    assert results[0]["name"] == "Milvus Shoulder Press"


def test_retriever_indexes_only_primary_documents_in_milvus(monkeypatch) -> None:
    captured_sources: set[str] = set()

    monkeypatch.setattr("agent.rag.retriever.milvus_enabled", lambda: True)
    monkeypatch.setattr("agent.rag.retriever.exercise_collection_name", lambda: "fitness_exercises_test")

    def capture_documents(documents, collection_name):
        captured_sources.update(
            str(document.get("metadata", {}).get("source") or "")
            for document in documents
        )
        return True

    monkeypatch.setattr("agent.rag.retriever.ensure_milvus_collection", capture_documents)
    monkeypatch.setattr(
        "agent.rag.retriever.search_milvus_collection",
        lambda query, collection_name, limit: [
            {
                "score": 0.8,
                "document": {
                    "metadata": {
                        "name": "Milvus Row",
                        "source": "wger",
                        "focus_tags": ["back_training"],
                        "difficulty": "beginner",
                    },
                    "raw": {
                        "name": "Milvus Row",
                        "source": "wger",
                        "focus_tags": ["back_training"],
                        "difficulty": "beginner",
                    },
                },
            }
        ],
    )

    results = retrieve_exercises(
        query="beginner back row",
        focus="back_training",
        level="beginner",
        limit=1,
    )

    assert results[0]["name"] == "Milvus Row"
    assert "local_fallback" not in captured_sources
    assert {"curated_rag", "wger"}.intersection(captured_sources)
