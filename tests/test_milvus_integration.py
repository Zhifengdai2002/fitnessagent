from __future__ import annotations

import os

import pytest

from agent.rag.milvus_store import build_milvus_collection, milvus_enabled, milvus_client, search_milvus_collection


@pytest.mark.integration
def test_real_milvus_roundtrip_when_enabled() -> None:
    if os.getenv("RUN_MILVUS_INTEGRATION") != "1":
        pytest.skip("Set RUN_MILVUS_INTEGRATION=1 to run real Milvus integration tests.")
    if not milvus_enabled():
        pytest.skip("Milvus is not configured.")

    collection_name = os.getenv("MILVUS_TEST_COLLECTION", "fitness_agent_test_exercises")
    document = {
        "title": "Integration Test Chest Press",
        "text": "beginner chest press machine horizontal push tutorial",
        "metadata": {
            "name": "Integration Test Chest Press",
            "source": "integration_test",
            "focus_tags": ["upper_chest_arms"],
            "difficulty": "beginner",
        },
        "raw": {
            "name": "Integration Test Chest Press",
            "source": "integration_test",
            "focus_tags": ["upper_chest_arms"],
            "difficulty": "beginner",
        },
    }

    assert build_milvus_collection([document], collection_name=collection_name, recreate=True)
    results = search_milvus_collection(
        "beginner chest machine press",
        collection_name=collection_name,
        limit=3,
    )

    assert results
    assert results[0]["document"]["metadata"]["name"] == "Integration Test Chest Press"

    client = milvus_client()
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
