import json

from agent.rag.documents import build_primary_knowledge_documents
from agent.rag.retriever import rebuild_knowledge_index, retrieve_knowledge
from agent.services.coach_tools import coach_tool_registry
from agent.tools.knowledge_tool import query_knowledge_base


def test_build_professional_knowledge_sources():
    docs = build_primary_knowledge_documents()
    sources = {str(doc.get("metadata", {}).get("source")) for doc in docs}
    assert {"exrx", "dietary_guidelines_for_americans", "pmc"}.issubset(sources)
    assert all(doc.get("type") == "knowledge" and doc.get("text") for doc in docs)


def test_retrieve_nutrition_knowledge():
    rebuild_knowledge_index()
    results = retrieve_knowledge(
        query="protein resistance training muscle recovery",
        topic="nutrition",
        limit=3,
    )
    assert results
    joined = " ".join(str(item.get("text", "")) for item in results).lower()
    assert "protein" in joined


def test_knowledge_tool_returns_compact_results():
    results = query_knowledge_base(
        "Should I train hard after poor sleep?",
        topic="recovery",
        limit=2,
    )
    assert results
    assert {"title", "summary", "source", "source_url"}.issubset(results[0])


def test_coach_registry_has_knowledge_tool():
    handler = coach_tool_registry()["query_knowledge_base"]["handler"]
    payload = handler(
        {
            "user_message": "How much protein should I eat?",
            "previous_result": {},
            "session_state": {},
            "profile_inputs": {},
        },
        {"query": "protein intake resistance training", "topic": "nutrition", "limit": 2},
    )
    decoded = json.loads(payload)
    assert decoded["knowledge_results"]
