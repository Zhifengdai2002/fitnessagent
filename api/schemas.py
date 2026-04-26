"""Pydantic schemas for the FitnessAgent API."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class GeneratePlanRequest(BaseModel):
    age: int = 30
    sex: Literal["male", "female", "other", "prefer_not_to_say"] = "male"
    height_cm: float = 175.0
    weight_kg: float = 78.0
    body_fat_pct: float = 24.0
    fitness_level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    activity_level: str = "lightly_active"
    primary_goal: str = "fat_loss"
    timeline_weeks: int = 4
    target_weight_kg: float = 72.0
    target_body_fat_pct: float = 18.0
    sessions_per_week: int = 4
    minutes_per_session: int = 60
    available_days: list[str] = Field(default_factory=lambda: ["Monday", "Wednesday", "Saturday"])
    start_date: str = Field(default_factory=lambda: date.today().isoformat())
    allergies_text: str = ""
    dietary_preferences: list[str] = Field(default_factory=list)
    profile_notes: str = ""


class ChatRequest(BaseModel):
    message: str


class DailyFeedbackRequest(BaseModel):
    current_weight_kg: float = 78.0
    current_body_fat_pct: float = 24.0
    workout_feeling: str = ""
    feeling_emoji: Literal["😊", "😐", "😫"] = "😊"


class ApiResponse(BaseModel):
    ok: bool = True
    message: str = ""
    state: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(ApiResponse):
    reply: str = ""
