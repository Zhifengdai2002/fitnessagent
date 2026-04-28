"""Shared LangGraph state for the FitnessAgent workflow.

This file mirrors the problem formulation:

- u: user profile
- c: user constraints
- g: fitness goals
- p_t: current plan at time t
- f_t: latest feedback at time t
- s_t: current user state snapshot at time t
"""

from typing import Literal

from typing_extensions import TypedDict


class UserProfile(TypedDict, total=False):
    """Static profile captured during cold start."""

    user_id: str
    age: int
    sex: Literal["male", "female", "other", "prefer_not_to_say"]
    height_cm: float
    weight_kg: float
    body_fat_pct: float
    fitness_level: Literal["beginner", "intermediate", "advanced"]
    activity_level: str
    training_background: str


class UserConstraints(TypedDict, total=False):
    """Physical, dietary, and scheduling constraints."""

    sessions_per_week: int
    minutes_per_session: int
    available_days: list[str]
    program_start_date: str
    injuries: list[str]
    pain_sensitive_areas: list[str]
    food_allergies: list[str]
    dietary_preferences: list[str]
    equipment_access: list[str]
    excluded_exercises: list[str]


class FitnessGoals(TypedDict, total=False):
    """User goals and success criteria."""

    primary_goal: str
    secondary_goals: list[str]
    target_weight_kg: float
    target_body_fat_pct: float
    target_muscle_gain_kg: float
    timeline_weeks: int
    priority_order: list[str]


class UserStateSnapshot(TypedDict, total=False):
    """State s_t at a given time step."""

    date: str
    weight_kg: float
    body_fat_pct: float
    resting_heart_rate: int
    readiness_score: float
    stress_level: int
    recovery_score: float
    sleep_hours: float
    notes: str


class ManualUserLog(TypedDict, total=False):
    """Daily manual logs provided directly by the user."""

    date: str
    sleep_hours: float
    weight_kg: float
    body_fat_pct: float
    water_intake_liters: float
    calories_consumed: int
    steps: int
    notes: str
    feeling_emoji: str


class ActivityTelemetry(TypedDict, total=False):
    """Objective activity data from tools such as Strava."""

    source: str
    activity_type: str
    duration_minutes: int
    distance_km: float
    avg_heart_rate: int
    max_heart_rate: int
    pace: str
    calories_burned: int
    performed_at: str


class UserFeedback(TypedDict, total=False):
    """Feedback f_t used to update the next plan."""

    date: str
    completed_workouts: list[str]
    completed_actions: list[str]
    feeling_emoji: str
    adherence_score: float
    fatigue_level: int
    pain_level: int
    pain_points: list[str]
    soreness_areas: list[str]
    motivation_level: int
    performance_notes: str
    manual_log: ManualUserLog
    activity_telemetry: list[ActivityTelemetry]


class ExercisePlanItem(TypedDict, total=False):
    """One movement inside a workout session."""

    name: str
    target_muscle: str
    sets: int
    reps: str
    equipment: str
    notes: str
    primary_muscles: list[str]
    secondary_muscles: list[str]
    coaching_cue: str
    why_this_exercise: str
    common_mistake: str
    regression: str
    progression: str
    knowledge_source: str


class WorkoutSession(TypedDict, total=False):
    """One planned training session."""

    day: str
    scheduled_date: str
    cycle_number: int
    cycle_session_index: int
    is_ad_hoc: bool
    is_cancelled: bool
    focus: str
    warmup: list[str]
    exercises: list[ExercisePlanItem]
    cooldown: list[str]
    safety_notes: list[str]


class NutritionTargets(TypedDict, total=False):
    """Daily nutrition targets for the current plan."""

    daily_calories: int
    protein_g: int
    carbs_g: int
    fat_g: int
    hydration_liters: float


class MealSuggestion(TypedDict, total=False):
    """Suggested food item from the local food database."""

    food_name: str
    serving_size: str
    calories: int
    protein_g: float
    carbs_g: float
    fat_g: float
    meal_slot: str


class FitnessPlan(TypedDict, total=False):
    """Plan p_t produced by the planner."""

    plan_id: str
    generated_at: str
    cycle_number: int
    cycle_start_date: str
    cycle_end_date: str
    summary: str
    objective_alignment: str
    workout_sessions: list[WorkoutSession]
    nutrition_targets: NutritionTargets
    meal_suggestions: list[MealSuggestion]
    recovery_actions: list[str]
    coaching_focus: list[str]


class DailyHistoryEntry(TypedDict, total=False):
    """Saved daily record containing the final plan and user feedback."""

    date: str
    weight_kg: float
    body_fat_pct: float
    completed_actions: list[str]
    completed_plan: WorkoutSession
    feedback: dict


class VideoResource(TypedDict, total=False):
    """Tutorial videos attached to exercises."""

    exercise_name: str
    title: str
    url: str
    source: str


class KnowledgeReference(TypedDict, total=False):
    """External knowledge retrieved for reasoning or explanation."""

    source_type: Literal["pubmed", "usda", "nutritionix", "wger", "exrx", "internal"]
    title: str
    url: str
    snippet: str
    reason: str


class EvaluationResult(TypedDict, total=False):
    """Hybrid evaluation output used by the feedback loop."""

    objective_score: float
    llm_score: float
    safety_risk: Literal["low", "medium", "high"]
    engagement_risk: Literal["low", "medium", "high"]
    should_revise: bool
    reasons: list[str]
    summary: str


class NormalizedChangeRequest(TypedDict, total=False):
    """Structured interpretation of a same-day change request."""

    request_type: str
    scope: str
    focus_category: str
    injury_reported: bool
    injury_areas: list[str]
    cancel_today: bool
    intensity_adjustment: str
    set_adjustment: str
    set_target: int
    duration_adjustment: str
    temporary_food_avoidances: list[str]
    permanent_food_preferences: list[str]
    summary: str
    confidence: float


class FitnessAgentState(TypedDict, total=False):
    """Top-level state shared across LangGraph nodes."""

    thread_id: str
    current_date: str
    profile_notes: str
    plan_change_request: str
    normalized_change_request: NormalizedChangeRequest

    # Problem formulation variables.
    user_profile: UserProfile
    constraints: UserConstraints
    goals: FitnessGoals
    current_state: UserStateSnapshot
    latest_feedback: UserFeedback
    current_plan: FitnessPlan
    memory_context: dict

    # Historical memory.
    state_history: list[UserStateSnapshot]
    feedback_history: list[UserFeedback]
    plan_history: list[FitnessPlan]
    daily_history: list[DailyHistoryEntry]

    # Tool and retrieval outputs.
    youtube_resources: list[VideoResource]
    retrieved_knowledge: list[KnowledgeReference]

    # Control flow and final outputs.
    evaluation_result: EvaluationResult
    coaching_message: str
    needs_revision: bool
    revision_reason: str


def create_initial_state() -> FitnessAgentState:
    """Return a clean initial state for a new user thread."""

    return {
        "thread_id": "",
        "current_date": "",
        "profile_notes": "",
        "plan_change_request": "",
        "normalized_change_request": {},
        "user_profile": {},
        "constraints": {},
        "goals": {},
        "current_state": {},
        "latest_feedback": {},
        "current_plan": {},
        "memory_context": {},
        "state_history": [],
        "feedback_history": [],
        "plan_history": [],
        "daily_history": [],
        "youtube_resources": [],
        "retrieved_knowledge": [],
        "evaluation_result": {},
        "coaching_message": "",
        "needs_revision": False,
        "revision_reason": "",
    }
