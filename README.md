# FitnessAgent

FitnessAgent is an AI fitness coaching app that generates personalized training and nutrition plans, remembers your feedback, and lets you adjust the plan through natural conversation with an AI Coach.

---

## Features

**Personalized Plan Generation**
- Creates a 7-day training cycle based on your fitness level, goals, available training days, body metrics, and diet preferences.
- Displays today's workout, today's nutrition, the full weekly cycle, and past history.

**AI Coach Chat**
- A floating chat interface that can directly modify your plan mid-conversation.
- Supported adjustments:
  - Change today's training focus (e.g. switch from chest to back)
  - Increase or decrease workout intensity
  - Add or reduce sets and reps
  - Replace a specific exercise with another
  - Swap out a food item in today's nutrition
  - Cancel today's workout
  - Report an injury or pain — the coach will cancel or adapt the plan safely
- Answers professional questions about training, nutrition, recovery, and exercise form using a knowledge base.

**Daily Feedback**
- Log how today's workout felt with an emoji, a short note, current weight, and body fat.
- Feedback is used as a training signal when generating the next plan.

**Plan Advancement**
- `Make Tomorrow's Plan` — advances the cycle by one day using today's feedback.
- Automatically starts a new 7-day cycle when the current one ends.

**Memory**
- Remembers your preferences, injury history, and past feedback across sessions.
- The AI Coach references this memory when suggesting plan changes or answering questions.

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

---

## Optional Setup

**Milvus vector store (for richer exercise retrieval and knowledge Q&A):**
```bash
docker compose up -d milvus-etcd milvus-minio milvus-standalone
```

Add to `.env`:
```env
RAG_BACKEND=milvus
MILVUS_URI=http://127.0.0.1:19530
MILVUS_EXERCISE_COLLECTION=fitness_exercises
MILVUS_FOOD_COLLECTION=fitness_foods
EMBEDDING_PROVIDER=zhipu
EMBEDDING_MODEL_NAME=embedding-3
EMBEDDING_API_KEY=your_key
EMBEDDING_BASE_URL=https://api.z.ai/api/paas/v4/
EMBEDDING_DIMENSIONS=1024
```

Rebuild vector collections:
```bash
.venv/bin/python -m agent.rag.milvus_indexer --recreate
```

Rebuild knowledge corpus:
```bash
.venv/bin/python -m agent.rag.knowledge_ingester --pubmed-retmax 50 --rebuild-index
```

---

## Disclaimer

FitnessAgent is a fitness planning and coaching demo. It is not medical advice. Stop training and seek qualified medical guidance when you experience injury symptoms, sharp pain, chest pain, dizziness, or other red flags.
