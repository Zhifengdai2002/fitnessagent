"""FastAPI backend for FitnessAgent."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import services
from api.schemas import ApiResponse, ChatRequest, ChatResponse, DailyFeedbackRequest, GeneratePlanRequest

app = FastAPI(title="FitnessAgent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/state", response_model=ApiResponse)
def state() -> ApiResponse:
    return ApiResponse(state=services.get_state())


@app.post("/generate_plan", response_model=ApiResponse)
def generate_plan(request: GeneratePlanRequest) -> ApiResponse:
    try:
        state_payload = services.generate_plan(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ApiResponse(message="Plan generated.", state=state_payload)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        reply, state_payload = services.chat(request.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(message="AI Coach replied.", reply=reply, state=state_payload)


@app.post("/make_tomorrow_plan", response_model=ApiResponse)
def make_tomorrow_plan(request: DailyFeedbackRequest) -> ApiResponse:
    try:
        state_payload = services.make_tomorrow_plan(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ApiResponse(message="Tomorrow's plan is ready.", state=state_payload)


@app.post("/reset", response_model=ApiResponse)
def reset() -> ApiResponse:
    return ApiResponse(message="State reset.", state=services.reset_state())
