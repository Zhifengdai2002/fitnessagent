import json
from pathlib import Path

from agent.rag import knowledge_ingester


def test_split_text_uses_overlapping_chunks():
    paragraph = "Resistance training improves strength and lean mass when progressed gradually. "
    text = paragraph * 80

    chunks = knowledge_ingester.split_text(text, chunk_size=260, chunk_overlap=40)

    assert len(chunks) > 3
    assert all("Resistance training" in chunk for chunk in chunks[:2])


def test_ingest_professional_knowledge_writes_real_chunks(monkeypatch, tmp_path):
    source_config = {
        "html_sources": [
            {
                "id": "sample_training_page",
                "title": "Sample Training Page",
                "source": "sample_source",
                "url": "https://example.com/training",
                "topic": "training",
                "doc_type": "article",
                "evidence_type": "curated_reference",
                "tags": ["progression"],
            }
        ],
        "pubmed_queries": [
            {
                "id": "sample_pubmed_query",
                "term": "resistance training recovery english[Language]",
                "source": "pmc",
                "topic": "recovery",
                "doc_type": "abstract",
                "evidence_type": "peer_reviewed_abstract",
            }
        ],
    }
    sources_path = tmp_path / "sources.json"
    output_path = tmp_path / "corpus.json"
    sources_path.write_text(json.dumps(source_config), encoding="utf-8")

    html_text = """
    <html><head><title>Training Principles</title></head><body>
      <main>
        <h1>Training Principles</h1>
        <p>Progressive overload should be applied gradually with recovery. Beginners should learn
        exercise technique before increasing load, and weekly volume should be matched to current
        readiness, training age, and tolerance.</p>
        <p>Training plans should consider frequency, intensity, time, and exercise selection. When
        fatigue is high, coaches can preserve the movement pattern while reducing total work.</p>
      </main>
    </body></html>
    """

    monkeypatch.setattr(knowledge_ingester, "_fetch_text", lambda url: html_text)
    monkeypatch.setattr(knowledge_ingester, "_pubmed_search", lambda term, retmax: ["12345"])
    monkeypatch.setattr(
        knowledge_ingester,
        "_pubmed_fetch",
        lambda pmids: [
            {
                "pmid": "12345",
                "title": "Sleep and resistance training recovery",
                "abstract": "Poor sleep can reduce readiness and recovery after exercise. " * 20,
                "journal": "Example Journal",
                "year": "2025",
            }
        ],
    )
    monkeypatch.setattr(knowledge_ingester, "_write_raw_cache", lambda *args, **kwargs: None)

    stats = knowledge_ingester.ingest_professional_knowledge(
        sources_path=sources_path,
        output_path=output_path,
        pubmed_retmax=5,
        request_delay_seconds=0,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert stats.source_records == 2
    assert stats.chunks == sum(len(record["chunks"]) for record in payload)
    assert {record["source"] for record in payload} == {"sample_source", "pmc"}
    assert all(record["metadata"]["language"] == "en" for record in payload)
    assert any(record["source_url"] == "https://pubmed.ncbi.nlm.nih.gov/12345/" for record in payload)
