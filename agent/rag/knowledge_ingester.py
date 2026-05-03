"""Ingest English professional fitness/nutrition sources into RAG chunks.

The generated corpus is intentionally plain JSON so it can feed both the local
test vector store and the Milvus backend without changing retrieval code.
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES_PATH = PROJECT_ROOT / "data" / "knowledge" / "professional_knowledge_sources.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "knowledge" / "professional_knowledge_corpus.json"
RAW_CACHE_DIR = PROJECT_ROOT / "data" / "external" / "knowledge_raw"
USER_AGENT = "FitnessAgentRAG/0.1 (local educational research; contact: local)"
NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass(frozen=True)
class IngestionStats:
    source_records: int
    chunks: int
    failed_sources: int


class MainTextExtractor(HTMLParser):
    """Small HTML-to-text extractor tuned for reference pages."""

    BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "blockquote", "td", "th"}
    SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._active_block: str | None = None
        self._buffer: list[str] = []
        self.blocks: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK_TAGS:
            self._flush_block()
            self._active_block = tag

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCK_TAGS:
            self._flush_block()
            self._active_block = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        cleaned = _normalize_whitespace(data)
        if not cleaned:
            return
        if self._in_title:
            self.title = _normalize_whitespace(f"{self.title} {cleaned}")
            return
        if self._active_block:
            self._buffer.append(cleaned)

    def close(self) -> None:
        self._flush_block()
        super().close()

    def _flush_block(self) -> None:
        text = _normalize_whitespace(" ".join(self._buffer))
        self._buffer.clear()
        if len(text) >= 30 and not _looks_like_navigation(text):
            self.blocks.append(text)


def load_source_config(path: Path = DEFAULT_SOURCES_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Source config must be an object: {path}")
    return payload


def ingest_professional_knowledge(
    *,
    sources_path: Path = DEFAULT_SOURCES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    pubmed_retmax: int = 200,
    request_delay_seconds: float = 0.34,
    rebuild_index: bool = False,
) -> IngestionStats:
    """Fetch configured sources, chunk them, and write the corpus JSON file."""

    config = load_source_config(sources_path)
    ingested_at = datetime.now(UTC).isoformat()
    records: list[dict[str, Any]] = []
    failed_sources = 0

    for source in config.get("html_sources", []):
        if not isinstance(source, dict):
            continue
        try:
            records.append(_ingest_html_source(source, ingested_at=ingested_at))
        except Exception as exc:  # noqa: BLE001 - ingestion should keep going.
            failed_sources += 1
            records.append(_failure_record(source, exc, ingested_at=ingested_at))
        time.sleep(request_delay_seconds)

    for source in config.get("pdf_sources", []):
        if not isinstance(source, dict):
            continue
        try:
            records.append(_ingest_pdf_source(source, ingested_at=ingested_at))
        except Exception as exc:  # noqa: BLE001 - ingestion should keep going.
            failed_sources += 1
            records.append(_failure_record(source, exc, ingested_at=ingested_at))
        time.sleep(request_delay_seconds)

    seen_pmids: set[str] = set()
    for query in config.get("pubmed_queries", []):
        if not isinstance(query, dict):
            continue
        try:
            pmids = _pubmed_search(str(query.get("term") or ""), retmax=pubmed_retmax)
            time.sleep(request_delay_seconds)
            for article in _pubmed_fetch(pmids):
                pmid = str(article.get("pmid") or "").strip()
                if not pmid or pmid in seen_pmids:
                    continue
                seen_pmids.add(pmid)
                records.append(_pubmed_record(query, article, ingested_at=ingested_at))
        except Exception as exc:  # noqa: BLE001
            failed_sources += 1
            records.append(_failure_record(query, exc, ingested_at=ingested_at))
        time.sleep(request_delay_seconds)

    valid_records = [record for record in records if record.get("chunks")]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(valid_records, ensure_ascii=False, indent=2), encoding="utf-8")

    if rebuild_index:
        from agent.rag.retriever import rebuild_knowledge_index

        rebuild_knowledge_index()

    return IngestionStats(
        source_records=len(valid_records),
        chunks=sum(len(record.get("chunks") or []) for record in valid_records),
        failed_sources=failed_sources,
    )


def _ingest_html_source(source: dict[str, Any], *, ingested_at: str) -> dict[str, Any]:
    url = str(source.get("url") or "").strip()
    if not url:
        raise ValueError("HTML source is missing url")
    html = _fetch_text(url)
    _write_raw_cache(_slug(str(source.get("id") or source.get("title") or url)), html, suffix=".html")

    parser = MainTextExtractor()
    parser.feed(html)
    parser.close()
    title = str(source.get("title") or parser.title or url).strip()
    text = _paragraphs_to_text(parser.blocks)
    if len(text) < 200:
        raise ValueError(f"not enough extracted text from {url}")

    return _base_record(
        source,
        title=title,
        source_url=url,
        text=text,
        ingested_at=ingested_at,
        metadata={"language": "en"},
    )


def _ingest_pdf_source(source: dict[str, Any], *, ingested_at: str) -> dict[str, Any]:
    url = str(source.get("url") or "").strip()
    if not url:
        raise ValueError("PDF source is missing url")

    pdf_bytes = _fetch_bytes(url)
    _write_raw_cache(_slug(str(source.get("id") or source.get("title") or url)), pdf_bytes, suffix=".pdf")
    text = _extract_pdf_text(pdf_bytes)
    if len(text) < 500:
        raise ValueError(f"not enough extracted PDF text from {url}")

    return _base_record(
        source,
        title=str(source.get("title") or url).strip(),
        source_url=url,
        text=text,
        ingested_at=ingested_at,
        metadata={"language": "en", "file_type": "pdf"},
    )


def _pubmed_search(term: str, *, retmax: int) -> list[str]:
    if not term:
        return []
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": str(max(1, retmax)),
        "retmode": "json",
        "sort": "relevance",
    }
    payload = json.loads(_fetch_text(f"{NCBI_ESEARCH_URL}?{urllib.parse.urlencode(params)}"))
    ids = payload.get("esearchresult", {}).get("idlist", [])
    return [str(pmid) for pmid in ids if str(pmid).strip()]


def _pubmed_fetch(pmids: list[str]) -> list[dict[str, Any]]:
    articles: list[dict[str, Any]] = []
    for batch in _batched(pmids, 100):
        if not batch:
            continue
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
        }
        xml_text = _fetch_text(f"{NCBI_EFETCH_URL}?{urllib.parse.urlencode(params)}")
        _write_raw_cache(f"pubmed_{batch[0]}_{batch[-1]}", xml_text, suffix=".xml")
        root = ET.fromstring(xml_text)
        for article_node in root.findall(".//PubmedArticle"):
            parsed = _parse_pubmed_article(article_node)
            if parsed.get("abstract"):
                articles.append(parsed)
        time.sleep(0.34)
    return articles


def _parse_pubmed_article(article_node: ET.Element) -> dict[str, Any]:
    pmid = _node_text(article_node.find(".//PMID"))
    title = _node_text(article_node.find(".//ArticleTitle"))
    abstract_parts = []
    for abstract_text in article_node.findall(".//AbstractText"):
        label = abstract_text.attrib.get("Label") or abstract_text.attrib.get("NlmCategory") or ""
        part = _normalize_whitespace(" ".join(abstract_text.itertext()))
        if not part:
            continue
        abstract_parts.append(f"{label}: {part}" if label else part)
    journal = _node_text(article_node.find(".//Journal/Title"))
    year = _node_text(article_node.find(".//PubDate/Year"))
    return {
        "pmid": pmid,
        "title": title,
        "abstract": "\n\n".join(abstract_parts),
        "journal": journal,
        "year": year,
    }


def _pubmed_record(query: dict[str, Any], article: dict[str, Any], *, ingested_at: str) -> dict[str, Any]:
    pmid = str(article.get("pmid") or "").strip()
    title = str(article.get("title") or f"PubMed article {pmid}").strip()
    text = "\n".join(
        part
        for part in [
            f"Journal: {article.get('journal')}" if article.get("journal") else "",
            f"Year: {article.get('year')}" if article.get("year") else "",
            str(article.get("abstract") or "").strip(),
        ]
        if part
    )
    return _base_record(
        query,
        title=title,
        source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        text=text,
        ingested_at=ingested_at,
        metadata={
            "language": "en",
            "pmid": pmid,
            "journal": article.get("journal") or "",
            "year": article.get("year") or "",
            "query_id": query.get("id") or "",
        },
    )


def _base_record(
    source: dict[str, Any],
    *,
    title: str,
    source_url: str,
    text: str,
    ingested_at: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    chunks = split_text(text)
    return {
        "id": str(source.get("id") or _slug(title)),
        "title": title,
        "source": str(source.get("source") or "professional_knowledge"),
        "source_url": source_url,
        "topic": str(source.get("topic") or ""),
        "section": str(source.get("section") or ""),
        "doc_type": str(source.get("doc_type") or ""),
        "evidence_type": str(source.get("evidence_type") or ""),
        "goal": _listify(source.get("goal")),
        "level": _listify(source.get("level")),
        "tags": _listify(source.get("tags")),
        "version": f"ingested_{ingested_at[:10]}",
        "metadata": {
            **metadata,
            "ingested_at": ingested_at,
            "chunk_size": 900,
            "chunk_overlap": 120,
        },
        "chunks": chunks,
    }


def split_text(text: str, *, chunk_size: int = 900, chunk_overlap: int = 120) -> list[str]:
    """Split text with LangChain when available, otherwise use a local fallback."""

    cleaned = _normalize_long_text(text)
    if not cleaned:
        return []
    try:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " "],
        )
        chunks = splitter.split_text(cleaned)
    except Exception:  # noqa: BLE001 - fallback keeps ingestion dependency-light.
        chunks = _fallback_split(cleaned, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 120]


def _failure_record(source: dict[str, Any], exc: Exception, *, ingested_at: str) -> dict[str, Any]:
    source_id = str(source.get("id") or source.get("term") or source.get("url") or "unknown_source")
    return {
        "id": f"failed_{_slug(source_id)}",
        "title": f"Failed ingestion: {source_id}",
        "source": str(source.get("source") or "unknown"),
        "source_url": str(source.get("url") or ""),
        "topic": str(source.get("topic") or ""),
        "doc_type": "ingestion_error",
        "evidence_type": "error",
        "version": f"ingested_{ingested_at[:10]}",
        "metadata": {"error": str(exc), "ingested_at": ingested_at},
        "chunks": [],
    }


def _fetch_text(url: str, *, timeout_seconds: int = 30) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds, context=_ssl_context()) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc.reason}") from exc


def _fetch_bytes(url: str, *, timeout_seconds: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds, context=_ssl_context()) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc.reason}") from exc


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required to ingest PDF sources") from exc

    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[str] = []
    for index, page in enumerate(reader.pages):
        text = _normalize_long_text(page.extract_text() or "")
        if text:
            pages.append(f"Page {index + 1}\n{text}")
    return "\n\n".join(pages)


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 - fallback to system certificates.
        return ssl.create_default_context()


def _write_raw_cache(name: str, content: str | bytes, *, suffix: str) -> None:
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_CACHE_DIR / f"{_slug(name)[:120]}{suffix}"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8", errors="replace")


def _paragraphs_to_text(blocks: Iterable[str]) -> str:
    seen: set[str] = set()
    paragraphs: list[str] = []
    for block in blocks:
        normalized = _normalize_whitespace(block)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        paragraphs.append(normalized)
    return "\n\n".join(paragraphs)


def _fallback_split(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def _normalize_long_text(text: str) -> str:
    lines = [_normalize_whitespace(line) for line in text.splitlines()]
    paragraphs = [line for line in lines if line]
    return "\n\n".join(paragraphs)


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _looks_like_navigation(text: str) -> bool:
    lowered = text.lower()
    if lowered in {"menu", "search", "skip to content", "share", "print"}:
        return True
    return len(text.split()) <= 3 and any(token in lowered for token in ["cookie", "login", "subscribe"])


def _node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return _normalize_whitespace(" ".join(node.itertext()))


def _batched(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _listify(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return normalized.strip("_") or "item"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest professional English knowledge sources into RAG JSON.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--pubmed-retmax", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.34)
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = ingest_professional_knowledge(
        sources_path=args.sources,
        output_path=args.output,
        pubmed_retmax=args.pubmed_retmax,
        request_delay_seconds=args.delay,
        rebuild_index=args.rebuild_index,
    )
    print(
        "Ingested professional knowledge: "
        f"{stats.source_records} records, {stats.chunks} chunks, {stats.failed_sources} failed sources"
    )


if __name__ == "__main__":
    main()
