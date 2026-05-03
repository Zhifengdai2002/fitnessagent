"""Professional knowledge retrieval helpers for AI Coach QA."""

from __future__ import annotations

from typing import Any

from agent.rag.retriever import retrieve_knowledge

MAX_SUMMARY_CHARS = 1200


def query_knowledge_base(
    query: str,
    topic: str = "",
    goal: str = "",
    level: str = "",
    injury_areas: list[str] | None = None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Return compact professional knowledge snippets for a coaching answer."""

    query_text = str(query or "").strip()
    if not query_text:
        return []

    results = retrieve_knowledge(
        query=query_text,
        topic=topic or None,
        goal=goal or None,
        level=level or None,
        injury_areas=injury_areas or [],
        limit=max(1, min(int(limit or 4), 6)),
    )

    compact: list[dict[str, Any]] = []
    for result in results:
        compact.append(
            {
                "title": result.get("title", ""),
                "summary": _truncate(str(result.get("text") or ""), MAX_SUMMARY_CHARS),
                "source": result.get("source", ""),
                "source_url": result.get("source_url", ""),
                "topic": result.get("topic", ""),
                "section": result.get("section", ""),
                "evidence_type": result.get("evidence_type", ""),
            }
        )
    return compact


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "..."
