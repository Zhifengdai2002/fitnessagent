# FitnessAgent

FitnessAgent is an AI fitness coaching system built on LangGraph + ReAct architecture. It integrates GLM vector embeddings and a Milvus vector store for dual-path RAG (exercise retrieval and knowledge Q&A), uses a three-tier Redis/MySQL memory architecture to persist user preferences and injury history, and drives a plan-generation → evaluation feedback loop via a LangGraph DAG. A FastAPI backend and Next.js frontend expose these capabilities as a personalized, conversationally adjustable training and nutrition plan.

---

## Architecture Overview

```text
Next.js frontend (fetch → JSON)
  └─ FastAPI backend
       ├─ AI Coach service  (ReAct loop, hard-safety gate, 8 native tools)
       ├─ LangGraph workflow (planner node → evaluator node → conditional edge)
       ├─ Three-tier memory  (Redis hot state / MySQL events / MySQL preferences)
       └─ Dual RAG pipeline  (exercise candidates + professional knowledge)
```

---

## LangGraph vs LangChain

LangChain is a linear pipeline — a chain of sequential operations where execution moves step to step. LangGraph is a stateful graph with nodes, edges, conditional routing, and loops, designed for complex systems that require memory, supervision, and human interaction.

FitnessAgent uses LangGraph because the planning workflow is stateful and dynamic. The graph continuously maintains and updates: chat history, user profile, injury records, preferences, current training cycle, and daily plan — all of which directly affect decisions at each node.

**Example:** The planner node generates a week's plan given a reported ankle injury. If the generated plan includes ankle-sensitive movements, the evaluator node detects this, attaches additional safety constraints, and routes execution back to the planner for revision.

---

## AI Coach — ReAct Architecture

The AI Coach uses a ReAct (Reasoning → Acting → Observing) loop because user requests are not single-turn tasks; they involve reasoning, tool usage, and memory retrieval across multiple steps.

**Safety gate first:** Before any LLM call, the system checks for hard-safety phrases (injury reports, chest pain, dizziness). Dangerous requests are stopped directly by rule-based logic, returning safety guidance without calling the LLM — preventing unsafe hallucinated recommendations.

**Loop:** After passing the safety gate, the system builds a system prompt combining memory context and injects it to the LLM (GLM). The model follows the reasoning-acting-observing loop to decide whether to call a tool or return a final response (max 4 steps).

**Example:** User says "I don't sleep well, can you decrease the intensity?" The agent reasons about the request, retrieves sleep-improvement advice from the knowledge RAG, calls the plan-adjustment tool to reduce intensity, observes whether all constraints are satisfied, and iterates until the plan is safe.

**8 native tools:** `cancel_workout`, `adjust_workout_volume`, `replace_exercise`, `replace_food`, `update_today_plan`, `update_cycle_plan`, `query_knowledge_base`, `write_memory`.

---

## Three-Tier Memory

| Layer | Storage | Contents | TTL / Scope |
|-------|---------|----------|-------------|
| Hot state | Redis | Recent conversations, current-week context, active injury status | 7-day TTL |
| Event log | MySQL (append-only) | Injury history, weight tracking, daily feedback, plan modification logs | Permanent |
| Derived preferences | MySQL (upserted) | Liked / avoided exercises, food preferences, preference evolution | Permanent |

When the user triggers an action such as "make tomorrow's plan," the system pulls from all three layers, consolidates them into a structured **memory context**, and injects it into the planner prompt.

**Sliding window:** The conversation history keeps the last 12 messages. When the total exceeds 12, older messages are compressed by an LLM summarizer into a summary capped at ~500 words, preserving key facts without blowing up the context window.

---

## RAG Pipeline

**Data sources:** US government health websites, fitness knowledge pages, and research paper abstracts from PubMed.

**Text splitting:** `RecursiveCharacterTextSplitter` — paragraph → sentence → word → character cascade. Chunk size: 900 characters, overlap: 120 characters. Chunks shorter than 120 characters are filtered out.

**Embeddings:** GLM `embedding-3`, 1024-dimensional vectors. Chosen for strong bilingual Chinese-English capability and good cross-lingual semantic alignment.

**Vector store (Milvus):** Each chunk stores an embedding vector, document JSON, metadata, and title. Document IDs are SHA-256-based hashes to prevent duplicate insertion across re-ingests.

**Retrieval:** Top-20 candidates retrieved by cosine similarity → metadata reranking by topic and fitness goals → top-4 chunks returned to the LLM.

**Dual-path RAG:**
- *Exercise RAG* — template-based query built from focus, target muscles, level, and goal; preference-scored rerank using learned user preferences.
- *Knowledge RAG* — the Coach LLM writes a free-form English query via function calling; results ground the answer to professional references.

---

## Evaluation — Recall@k + RAGAS

20 realistic fitness-related queries with defined keywords and reference answers were used for evaluation.

| Metric | Score | Description |
|--------|-------|-------------|
| Recall@4 | ~90% | Whether top-4 retrieved chunks contain the key information |
| Context Precision (RAGAS) | ~70% | Fraction of retrieved chunks that are actually relevant |
| Context Recall (RAGAS) | ~70% | Fraction of needed information that was retrieved |
| Faithfulness (RAGAS) | ~60% | Answer claims are grounded in retrieved context, not hallucinated |
| Answer Relevance (RAGAS) | ~65% | Cosine similarity between the answer and the original query |

RAGAS evaluation uses Qwen as the LLM judge. Scores reflect the current knowledge corpus size; re-ingesting more documents is expected to improve Context Recall.

---

## Core Features

- Generates personalized 7-day training cycles from user profile, fitness level, goals, available training days, body metrics, and diet preferences.
- AI Coach chat can directly modify today's plan: change focus, adjust intensity/sets/reps, replace exercises or food, cancel workout, handle injury reports.
- Records daily feedback (feeling note, emoji, weight, body fat) and uses it as a training signal for the next plan.
- Advances the plan automatically with `Make Tomorrow's Plan`; generates a new 7-day cycle when the current one ends.
- Persists all state through the three-tier memory architecture (Redis + MySQL); falls back to local JSON when neither is configured.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js, React, TypeScript |
| Backend | FastAPI, Pydantic |
| Agent workflow | LangGraph |
| AI Coach loop | ReAct (function calling) |
| LLM | GLM (Zhipu / z.ai compatible API) |
| Embeddings | GLM `embedding-3`, 1024 dims |
| Vector store | Milvus (AUTOINDEX, COSINE) |
| Hot memory | Redis (TTL 7d) |
| Long-term memory | MySQL (append-only events + upserted preferences) |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| RAG evaluation | Recall@k + RAGAS |

---

## Quick Start

**Backend:**
```bash
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**Frontend:**
```bash
cd frontend
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Open `http://127.0.0.1:3000`.

**Optional — Milvus vector store:**
```bash
docker compose up -d milvus-etcd milvus-minio milvus-standalone
```

Set in `.env`:
```env
RAG_BACKEND=milvus
MILVUS_URI=http://127.0.0.1:19530
MILVUS_EXERCISE_COLLECTION=fitness_exercises
MILVUS_FOOD_COLLECTION=fitness_foods
```

**Optional — GLM embeddings:**
```env
EMBEDDING_PROVIDER=zhipu
EMBEDDING_MODEL_NAME=embedding-3
EMBEDDING_API_KEY=your_zhipu_or_zai_key
EMBEDDING_BASE_URL=https://api.z.ai/api/paas/v4/
EMBEDDING_DIMENSIONS=1024
```

**Rebuild vector collections after changing embedding config:**
```bash
.venv/bin/python -m agent.rag.milvus_indexer --recreate
```

**Rebuild knowledge corpus:**
```bash
.venv/bin/python -m agent.rag.knowledge_ingester --pubmed-retmax 50 --rebuild-index
```

**Run Milvus integration test:**
```bash
RUN_MILVUS_INTEGRATION=1 .venv/bin/python -m pytest tests/test_milvus_integration.py -q
```

---

## Disclaimer

FitnessAgent is a fitness planning and coaching demo. It is not medical advice. Stop training and seek qualified medical guidance when you experience injury symptoms, sharp pain, chest pain, dizziness, or other red flags.
