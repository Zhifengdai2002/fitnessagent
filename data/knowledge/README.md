# Knowledge Base

This directory contains source configuration and generated corpora for the
FitnessAgent RAG layer.

## Professional Knowledge Ingestion

`professional_knowledge_sources.json` lists English long-form sources such as
ExRx, WHO, Dietary Guidelines for Americans, The Fitness Wiki, and PubMed query
terms. It is only a source manifest, not the knowledge base itself.

Run ingestion to fetch real source content, split it into chunks, and write the
generated corpus:

```bash
.venv/bin/python -m agent.rag.knowledge_ingester --pubmed-retmax 300 --rebuild-index
```

The generated `professional_knowledge_corpus.json` is loaded before the smaller
manual seed file. Each record keeps chunk text plus metadata such as `source`,
`source_url`, `topic`, `section`, `doc_type`, `evidence_type`, and language.

If Milvus is enabled through environment variables, the rebuild step also
refreshes the Milvus knowledge collection. Otherwise it refreshes the local
test vector index.

## Embeddings

The RAG layer can run in two embedding modes:

- `EMBEDDING_PROVIDER=hash`: deterministic local vectors for offline tests.
- `EMBEDDING_PROVIDER=zhipu`: OpenAI-compatible Zhipu `embedding-3` vectors for
  production retrieval.

When switching provider or vector dimensions, recreate Milvus collections:

```bash
.venv/bin/python -m agent.rag.milvus_indexer --recreate
```

The generated local JSON index also records the embedding provider and
dimension, so stale indexes are rebuilt automatically.
