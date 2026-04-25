# FitnessAgent

FitnessAgent is a Streamlit fitness coaching app backed by a LangGraph workflow and a Zhipu-compatible LLM API. It generates cycle-based training and nutrition plans, lets an AI Coach modify today's plan through chat, records daily feedback, and carries the latest app state across restarts.

## Features

- Profile-driven plan generation with fitness level, goal, available days, diet preferences, and body metrics.
- Cycle-based training plans with a visible Today's Plan and Training Cycle.
- AI Coach floating chat box for plan questions and direct plan edits.
- AI Coach can adjust today's workout intensity, cancel today's workout, replace same-type exercises, update nutrition items, and preserve previous edits as the next baseline.
- Hard training rules are enforced: beginner plans use 2 exercises, intermediate plans use 3, advanced plans use 4, and every exercise uses 4 sets.
- Rep rules: beginner baseline is 6-10 reps; intermediate and advanced baseline is 10-15 reps.
- Higher intensity adds one exercise and raises reps/notes; lower intensity reduces reps/notes and can reduce exercise count without going below 2 exercises.
- Daily Feedback records the final Today's Plan, completion status, current weight, body fat, feeling note, and emoji.
- History groups daily records by cycle.
- When the current cycle ends, Make Tomorrow's Plan automatically generates the next cycle and updates Today's Plan.
- Local persistence saves the current app state to `data/app_state.json`, so reopening the app resumes from the previous plan.
- Reset App clears local saved state and starts fresh.

## Project Structure

```text
.
├── app.py                         # Streamlit frontend and app state handling
├── agent/
│   ├── graph.py                   # LangGraph workflow assembly
│   ├── nodes/
│   │   ├── planner.py             # Plan generation and cycle/change logic
│   │   └── evaluator.py           # Feedback evaluation logic
│   ├── prompts/                   # LLM prompts
│   └── tools/                     # Exercise, food, video, and local RAG helpers
├── data/
│   ├── exercise_db.json           # Local exercise knowledge base
│   └── food_db.json               # Local food knowledge base
├── requirements.txt
└── .env.example
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your environment file:

```bash
cp .env.example .env
```

Set your model configuration in `.env`:

```bash
MODEL_API_KEY=your_model_api_key
MODEL_BASE_URL=https://api.z.ai/api/paas/v4/
MODEL_NAME=glm-4.5-air
MODEL_THINKING_TYPE=enabled
YOUTUBE_API_KEY=your_youtube_api_key_optional
```

`MODEL_API_KEY` or `ZAI_API_KEY` can be used for the Zhipu-compatible API key.

## Run

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

If you are using the local virtual environment directly:

```bash
.venv/bin/streamlit run app.py
```

## Typical Workflow

1. Fill out the sidebar profile.
2. Click `Run FitnessAgent` to generate the first cycle.
3. Review Today's Plan and Training Cycle.
4. Ask AI Coach to adjust today's workout or nutrition if needed.
5. Fill Daily Feedback after the day is complete.
6. Click `Make Tomorrow's Plan`.
7. Repeat daily. When the current cycle ends, the app generates the next cycle automatically.

## Local Persistence

FitnessAgent saves runtime state to:

```text
data/app_state.json
```

This includes the current plan, current date, profile inputs, daily history, feedback history, and AI Coach chat messages. The file is ignored by git because it contains personal runtime data.

To start over, use the `Reset App` button in the sidebar. This clears the current Streamlit state and deletes `data/app_state.json`.

## Notes

- The app is a coaching and planning tool, not medical advice.
- If pain, injury, dizziness, chest pain, or other red flags are reported, the AI Coach should cancel training for the day and recommend appropriate caution.
- Local RAG is currently based on the bundled exercise and food JSON databases. Larger external exercise or nutrition databases can be added later as a richer retrieval layer.
