# FitnessAgent

FitnessAgent is an AI fitness coaching workspace with a Next.js frontend, a FastAPI backend, a LangGraph multi-agent planning workflow, and a Zhipu-compatible LLM API. It generates cycle-based workout and nutrition plans, lets an AI Coach modify Today's Plan through chat, records daily feedback, and persists runtime memory across restarts.

## Features

- Profile-driven training and nutrition planning using age, sex, height, weight, body fat, fitness level, goal, available days, diet preferences, and notes.
- 7-day cycle planning with Today's Plan, Training Cycle, Daily Feedback, and cycle-grouped History.
- Floating AI Coach chat that can update Today's Plan directly.
- Multi-agent AI Coach routing:
  - Coordinator routes chat messages to the correct specialist path.
  - Safety logic handles injury, pain, cancellation, and recovery risk.
  - Planner logic updates workouts, nutrition, intensity, exercises, and same-day changes.
  - Memory logic records structured facts that should influence future planning.
- Controlled plan editing instead of free-form LLM rewrites:
  - cancel today's workout
  - protect related future sessions when injury is reported
  - replace same-focus exercises
  - replace today's food items
  - change today's focus
  - adjust reps, notes, exercise count, or sets based on the user's request
- Intensity rules:
  - baseline beginner: 2 exercises
  - baseline intermediate: 3 exercises
  - baseline advanced: 4 exercises
  - default sets: 4 sets per exercise
  - explicit set changes are allowed from 3 to 5 sets
  - beginner baseline reps: 6-10
  - intermediate/advanced baseline reps: 10-15
  - vague higher-intensity requests raise reps/notes and may add one exercise
  - explicit "add sets" changes sets only and does not add exercises
  - vague lower-intensity requests reduce reps/notes and may reduce exercises for intermediate/advanced users
- Make Tomorrow's Plan records the final Today's Plan plus daily feedback, then advances to the next calendar day.
- Automatic next-cycle generation when the current 7-day cycle ends.
- MySQL persistence when `DATABASE_URL` is configured, with `data/app_state.json` as local fallback.
- Reset button clears saved app state and starts fresh.
- Local exercise, food, and lightweight RAG data are bundled under `data/`.

## Architecture

```text
Next.js UI
   |
   v
FastAPI backend
   |
   v
Service layer
   |
   +--> LangGraph planner/evaluator workflow
   +--> AI Coach chat service
   +--> Memory service
   +--> MySQL or JSON persistence
   +--> Exercise, food, video, and local RAG tools
```

## Project Structure

```text
.
├── api/
│   ├── main.py                    # FastAPI routes
│   ├── schemas.py                 # API request/response schemas
│   └── services.py                # Backend service entry points
├── frontend/
│   ├── app/                       # Next.js app router UI
│   └── lib/api.ts                 # Frontend API client
├── agent/
│   ├── graph.py                   # LangGraph workflow assembly
│   ├── state.py                   # Shared typed state
│   ├── nodes/
│   │   ├── planner.py             # Plan generation/change logic
│   │   ├── evaluator.py           # Feedback evaluation
│   │   └── coach.py               # Coach node support
│   ├── services/
│   │   ├── coach_chat_service.py  # AI Coach chat/tools logic
│   │   ├── feedback_service.py    # Daily feedback and next-day flow
│   │   ├── memory.py              # Structured memory helpers
│   │   ├── mysql_store.py         # MySQL persistence backend
│   │   ├── persistence.py         # MySQL-first, JSON-fallback storage
│   │   ├── planning_helpers.py    # Date/session helper logic
│   │   └── state_builders.py      # Initial state construction
│   ├── prompts/                   # LLM prompts for planner, evaluator, coordinator, safety, memory
│   └── tools/                     # Exercise, food, video, and RAG helpers
├── data/
│   ├── exercise_db.json           # Local exercise knowledge base
│   ├── food_db.json               # Local food knowledge base
│   ├── app_state.json             # JSON fallback runtime state, ignored by git
│   └── knowledge/                 # Local knowledge snippets
├── requirements.txt
└── .env.example
```

## Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install --cache .npm-cache
cd ..
```

Create your local environment file:

```bash
cp .env.example .env
```

Configure the LLM API in `.env`:

```bash
MODEL_API_KEY=your_model_api_key_here
MODEL_BASE_URL=https://api.z.ai/api/paas/v4/
MODEL_NAME=glm-4.5-air
MODEL_THINKING_TYPE=enabled
YOUTUBE_API_KEY=your_youtube_api_key_optional
```

`MODEL_API_KEY` or `ZAI_API_KEY` can be used for the Zhipu-compatible API key.

## Optional MySQL Persistence

FitnessAgent can store the single-user runtime state in MySQL:

```bash
DATABASE_URL=mysql+pymysql://fitness_user:fitness_password@127.0.0.1:3306/fitness_agent?charset=utf8mb4
FITNESS_AGENT_USER_ID=demo-user
```

Create the database before starting the backend:

```sql
CREATE DATABASE fitness_agent CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

The app creates the `users` and `app_states` tables automatically. If `DATABASE_URL` is not set or MySQL is unavailable, FitnessAgent falls back to:

```text
data/app_state.json
```

## Run

Start the FastAPI backend:

```bash
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal, start the Next.js frontend:

```bash
cd frontend
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Open:

```text
http://127.0.0.1:3000
```

Backend health check:

```text
http://127.0.0.1:8000/health
```

## Typical Workflow

1. Fill out the Profile panel.
2. Click `Run FitnessAgent` to generate a 7-day cycle.
3. Review Today's Plan, Today's Nutrition, and Training Cycle.
4. Ask AI Coach to adjust today's plan if needed.
5. Fill Daily Feedback at the end of the day.
6. Click `Make Tomorrow's Plan`.
7. Repeat daily. When the current cycle ends, the next cycle is generated automatically.

## AI Coach Behavior

The chat flow is controlled by prompts plus deterministic service logic:

- `coach_coordinator_prompt.txt` classifies the message and routes it.
- `coach_safety_prompt.txt` handles injury, pain, cancellation, and safety issues.
- `coach_planner_prompt.txt` selects the planner action.
- `change_request_prompt.txt` normalizes natural-language requests into structured fields.
- `planner_prompt.txt` controls generation and cycle-level planning rules.
- `coach_chat_service.py` applies the actual state changes.

This means the LLM interprets the request, but code performs bounded updates to avoid accidental plan rewrites.

Examples:

- "add sets today" increases sets only, bounded to 5, and does not add an exercise.
- "make it 5 sets" sets exercises to 5 sets.
- "reduce sets" lowers sets, bounded to 3.
- "I feel strong and want more" uses higher-intensity logic.
- "I slept badly" uses lower-intensity logic.
- "replace broccoli" changes today's nutrition while preserving the workout.
- "cancel today's plan" cancels today's workout only.
- "my back is injured" cancels today's workout and protects same-cycle sessions that clearly stress the back.

## Persistence and Memory

FitnessAgent keeps two related kinds of state:

- Runtime app state: current date, profile, plan, daily history, chat messages, and UI-facing data.
- Structured memory: injury events, plan modification logs, food preferences, training signals, and feedback-derived facts.

With MySQL enabled, the state is stored under:

```text
users.id = demo-user
app_states.user_id = demo-user
```

Without MySQL, the same payload is stored in `data/app_state.json`.

Use the Reset button in the UI to start over. It deletes the saved app state for the configured user and clears the JSON fallback file.

## API Endpoints

```text
GET  /health
GET  /state
POST /generate_plan
POST /chat
POST /make_tomorrow_plan
POST /reset
```

## Notes

- FitnessAgent is a coaching and planning tool, not medical advice.
- If pain, injury, dizziness, chest pain, or other red flags are reported, the AI Coach should prioritize stopping training and recommend qualified medical guidance.
- The local knowledge base is intentionally small for development. It can later be replaced or expanded with external exercise/nutrition APIs, embeddings, or a vector database.
