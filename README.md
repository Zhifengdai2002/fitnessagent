# FitnessAgent

FitnessAgent is an AI fitness coaching workspace that creates personalized training and nutrition plans, remembers user feedback, and lets an AI Coach adjust the plan through natural conversation.

The goal is not just to generate a static workout schedule, but to build a coach-like system that can react to daily conditions such as fatigue, injury, food preferences, schedule changes, and user motivation.

## Core Features

- Generates personalized 7-day training cycles from user profile, fitness level, goals, available training days, body metrics, and diet preferences.
- Shows a clear Today's Plan, Today's Nutrition, Training Cycle, Daily Feedback, and cycle-based History.
- Provides a floating AI Coach chat that can directly update Today's Plan.
- Supports same-day plan changes such as:
  - changing today's training focus
  - increasing or decreasing intensity
  - adding or reducing sets
  - replacing same-focus exercises
  - replacing food items
  - cancelling today's workout
  - handling injury or pain reports
- Records daily feedback including final plan, current weight, body fat, feeling note, and emoji.
- Automatically advances the plan with `Make Tomorrow's Plan`.
- Generates a new cycle when the current 7-day cycle ends.
- Persists current state and memory with MySQL, with local JSON fallback.

## AI Coach Behavior

The AI Coach combines LLM reasoning with controlled tool-like actions. The LLM interprets what the user wants, while backend logic applies bounded updates to avoid accidental plan rewrites.

Examples:

- If the user says "add sets", the coach changes sets only, without adding exercises.
- If the user says "I feel strong and want more", the coach can increase reps, notes, or add an exercise.
- If the user says "I slept badly", the coach lowers intensity.
- If the user says "replace broccoli", nutrition changes while the workout stays unchanged.
- If the user reports injury or pain, safety rules can cancel today's workout and protect related future sessions.

## Multi-Agent Design

FitnessAgent uses a multi-agent workflow around planning and coaching:

- **Coordinator Agent** decides whether the user message needs no action, safety handling, or plan modification.
- **Safety Agent** handles injury, pain, cancellation, and recovery risk.
- **Planner Agent** creates and updates workouts, nutrition, intensity, and cycle plans.
- **Evaluator / Feedback Logic** processes daily feedback and prepares future planning context.
- **Memory Layer** stores structured facts such as injuries, food preferences, training signals, and plan modification logs.

This structure makes the app closer to a real coaching system than a one-shot plan generator.

## Tech Stack

- **Frontend:** Next.js, React, TypeScript
- **Backend:** FastAPI, Pydantic
- **Agent Workflow:** LangGraph
- **LLM:** Zhipu-compatible chat API
- **Persistence:** MySQL with JSON fallback
- **Knowledge Tools:** local exercise database, food database, lightweight RAG helpers, YouTube resource lookup
- **Language:** Python, TypeScript

## Current Architecture

```text
Next.js frontend
   -> FastAPI backend
      -> AI Coach service
      -> LangGraph planning workflow
      -> Memory and persistence layer
      -> Exercise, food, video, and RAG tools
```

## Future Improvements

- Replace the local JSON exercise and food databases with larger professional data sources.
- Add vector search for exercise substitutions, food alternatives, and coaching knowledge.
- Expand memory from simple structured storage into a richer long-term user profile.
- Add multi-user authentication and user-specific database records.
- Move from single-user MySQL persistence to production-ready user/session management.
- Add explicit function calling / tool calling for all AI Coach actions.
- Improve plan evaluation with progress trends across multiple cycles.
- Add richer UI visualizations for body metrics, training consistency, and plan changes.
- Add testing around AI Coach plan modifications to prevent regressions.

## Quick Start

Backend:

```bash
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Open:

```text
http://127.0.0.1:3000
```

## Disclaimer

FitnessAgent is a fitness planning and coaching demo. It is not medical advice. Users should stop training and seek qualified medical guidance when they experience injury symptoms, sharp pain, chest pain, dizziness, or other red flags.
