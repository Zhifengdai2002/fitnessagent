"""Evaluator node for judging whether the current plan needs revision."""

from __future__ import annotations

import json

from agent.llm import call_model_json, load_prompt
from agent.state import EvaluationResult, FitnessAgentState


def feedback_evaluation_node(state: FitnessAgentState) -> FitnessAgentState:
    """Evaluate latest feedback and decide whether the next plan should change."""

    latest_feedback = state.get("latest_feedback", {})
    current_state = state.get("current_state", {})
    current_plan = state.get("current_plan", {})
    goals = state.get("goals", {})

    if not _has_substantive_feedback(latest_feedback):
        return {
            "evaluation_result": {},
            "needs_revision": False,
            "revision_reason": "",
            "feedback_history": list(state.get("feedback_history", [])),
            "state_history": list(state.get("state_history", [])),
        }

    fallback_safety_risk = _assess_safety_risk(latest_feedback, current_state)
    fallback_engagement_risk = _assess_engagement_risk(latest_feedback)
    fallback_objective_score = _compute_objective_score(
        latest_feedback=latest_feedback,
        current_state=current_state,
        goals=goals,
        state_history=state.get("state_history", []),
    )
    fallback_llm_score = _compute_alignment_score(latest_feedback, current_plan)
    fallback_weight_change = _compute_weight_change(current_state, state.get("state_history", []))
    fallback_body_fat_change = _compute_body_fat_change(current_state, state.get("state_history", []))
    fallback_reasons = _collect_revision_reasons(
        latest_feedback,
        current_state,
        goals,
        fallback_safety_risk,
        fallback_engagement_risk,
        fallback_weight_change,
        fallback_body_fat_change,
    )
    fallback_should_revise = _should_revise(
        fallback_safety_risk,
        fallback_engagement_risk,
        latest_feedback,
        fallback_reasons,
    )

    model_evaluation = call_model_json(
        system_prompt=load_prompt("evaluator_prompt.txt"),
        user_prompt=json.dumps(
            {
                "latest_feedback": latest_feedback,
                "current_state": current_state,
                "current_plan": current_plan,
                "goals": goals,
                "plan_change_request": state.get("plan_change_request", ""),
                "weight_change_kg": fallback_weight_change,
                "body_fat_change_pct": fallback_body_fat_change,
                "feedback_history_tail": state.get("feedback_history", [])[-3:],
                "state_history_tail": state.get("state_history", [])[-3:],
                "fallback_reference": {
                    "objective_score": fallback_objective_score,
                    "llm_score": fallback_llm_score,
                    "safety_risk": fallback_safety_risk,
                    "engagement_risk": fallback_engagement_risk,
                },
            },
            ensure_ascii=True,
            indent=2,
        ),
        temperature=0.1,
        max_tokens=1800,
    )

    safety_risk = _sanitize_risk(model_evaluation.get("safety_risk"), fallback_safety_risk)
    engagement_risk = _sanitize_risk(model_evaluation.get("engagement_risk"), fallback_engagement_risk)
    objective_score = _sanitize_score(model_evaluation.get("objective_score"), fallback_objective_score)
    llm_score = _sanitize_score(model_evaluation.get("llm_score"), fallback_llm_score)
    reasons = _sanitize_reasons(model_evaluation.get("reasons"), fallback_reasons)
    should_revise = bool(model_evaluation.get("should_revise", fallback_should_revise))
    if safety_risk == "high" or engagement_risk == "high":
        should_revise = True
    summary = str(
        model_evaluation.get("summary")
        or _build_summary(
            objective_score=objective_score,
            llm_score=llm_score,
            safety_risk=safety_risk,
            engagement_risk=engagement_risk,
            should_revise=should_revise,
            reasons=reasons,
        )
    )

    evaluation_result: EvaluationResult = {
        "objective_score": objective_score,
        "llm_score": llm_score,
        "safety_risk": safety_risk,
        "engagement_risk": engagement_risk,
        "should_revise": should_revise,
        "reasons": reasons,
        "summary": summary,
    }

    updated_feedback_history = _append_history_item(
        history=state.get("feedback_history", []),
        item=latest_feedback,
        date_key="date",
    )
    updated_state_history = _append_history_item(
        history=state.get("state_history", []),
        item=current_state,
        date_key="date",
    )

    return {
        "evaluation_result": evaluation_result,
        "feedback_history": updated_feedback_history,
        "state_history": updated_state_history,
        "needs_revision": should_revise,
        "revision_reason": "; ".join(reasons[:3]) if reasons else "",
    }


def _assess_safety_risk(latest_feedback: dict, current_state: dict) -> str:
    fatigue_level = int(latest_feedback.get("fatigue_level", 0))
    pain_level = int(latest_feedback.get("pain_level", 0))
    pain_points = [str(item).lower() for item in latest_feedback.get("pain_points", [])]
    soreness_areas = [str(item).lower() for item in latest_feedback.get("soreness_areas", [])]
    sleep_hours = float(current_state.get("sleep_hours", 7.0))
    recovery_score = float(current_state.get("recovery_score", 1.0))

    high_risk_keywords = ("sharp", "acute", "injury", "swollen", "unstable")
    if (
        fatigue_level >= 9
        or pain_level >= 7
        or sleep_hours < 5.0
        or recovery_score < 0.35
        or any(any(keyword in point for keyword in high_risk_keywords) for point in pain_points)
    ):
        return "high"

    if (
        fatigue_level >= 7
        or pain_level >= 4
        or sleep_hours < 6.0
        or recovery_score < 0.55
        or pain_points
        or soreness_areas
    ):
        return "medium"

    return "low"


def _assess_engagement_risk(latest_feedback: dict) -> str:
    adherence_score = float(latest_feedback.get("adherence_score", 1.0))
    motivation_level = int(latest_feedback.get("motivation_level", 7))
    completed_workouts = latest_feedback.get("completed_workouts", [])

    if adherence_score < 0.4 or motivation_level <= 3:
        return "high"
    if adherence_score < 0.7 or motivation_level <= 5:
        return "medium"
    if not completed_workouts and adherence_score < 0.85:
        return "medium"
    return "low"


def _compute_objective_score(
    *,
    latest_feedback: dict,
    current_state: dict,
    goals: dict,
    state_history: list[dict],
) -> float:
    adherence_score = float(latest_feedback.get("adherence_score", 1.0))
    fatigue_level = int(latest_feedback.get("fatigue_level", 0))
    pain_level = int(latest_feedback.get("pain_level", 0))
    sleep_hours = float(current_state.get("sleep_hours", 7.0))
    recovery_score = float(current_state.get("recovery_score", 0.7))
    primary_goal = str(goals.get("primary_goal", "weight_loss")).lower()
    weight_change_kg = _compute_weight_change(current_state, state_history)
    body_fat_change_pct = _compute_body_fat_change(current_state, state_history)

    score = 100.0
    score -= (1.0 - adherence_score) * 35.0
    score -= max(0, fatigue_level - 5) * 4.0
    score -= pain_level * 4.0
    if sleep_hours < 7.0:
        score -= (7.0 - sleep_hours) * 3.0
    if recovery_score < 0.7:
        score -= (0.7 - recovery_score) * 25.0
    if "weight" in primary_goal and weight_change_kg > 0:
        score -= min(18.0, weight_change_kg * 6.0)
    elif "weight" in primary_goal and weight_change_kg < 0:
        score += min(8.0, abs(weight_change_kg) * 4.0)
    if "sculpt" in primary_goal and body_fat_change_pct > 0:
        score -= min(18.0, body_fat_change_pct * 7.0)
    elif "sculpt" in primary_goal and body_fat_change_pct < 0:
        score += min(10.0, abs(body_fat_change_pct) * 5.0)
    return round(max(0.0, min(score, 100.0)), 1)


def _compute_alignment_score(latest_feedback: dict, current_plan: dict) -> float:
    adherence_score = float(latest_feedback.get("adherence_score", 1.0))
    fatigue_level = int(latest_feedback.get("fatigue_level", 0))
    pain_level = int(latest_feedback.get("pain_level", 0))
    session_count = len(current_plan.get("workout_sessions", []))

    score = 85.0
    if session_count >= 4 and fatigue_level >= 7:
        score -= 15.0
    if pain_level >= 4:
        score -= 12.0
    if adherence_score < 0.7:
        score -= (0.7 - adherence_score) * 40.0
    return round(max(0.0, min(score, 100.0)), 1)


def _collect_revision_reasons(
    latest_feedback: dict,
    current_state: dict,
    goals: dict,
    safety_risk: str,
    engagement_risk: str,
    weight_change_kg: float,
    body_fat_change_pct: float,
) -> list[str]:
    reasons: list[str] = []

    fatigue_level = int(latest_feedback.get("fatigue_level", 0))
    pain_level = int(latest_feedback.get("pain_level", 0))
    adherence_score = float(latest_feedback.get("adherence_score", 1.0))
    motivation_level = int(latest_feedback.get("motivation_level", 7))
    sleep_hours = float(current_state.get("sleep_hours", 7.0))
    primary_goal = str(goals.get("primary_goal", "weight_loss")).lower()

    if safety_risk == "high":
        reasons.append("Safety risk is high and the plan should be reduced or modified.")
    elif safety_risk == "medium":
        reasons.append("Recovery signals are elevated, so the plan may need a lighter version.")

    if pain_level >= 4 or latest_feedback.get("pain_points"):
        pain_points = ", ".join(latest_feedback.get("pain_points", [])) or "reported pain"
        reasons.append(f"Pain feedback was reported around: {pain_points}.")

    if fatigue_level >= 7:
        reasons.append("Fatigue is high enough to justify lowering volume or intensity.")

    if adherence_score < 0.7:
        reasons.append("Adherence dropped below the target threshold, so plan simplicity should increase.")

    if engagement_risk == "high" or motivation_level <= 3:
        reasons.append("Motivation is low, so coaching and workload should be adjusted.")

    if sleep_hours < 6.0:
        reasons.append("Sleep is insufficient for the current training demand.")

    if weight_change_kg >= 0.8 and ("fat" in primary_goal or "weight" in primary_goal):
        reasons.append(
            f"Weight increased by {weight_change_kg:.1f}kg since the last check-in, so nutrition adherence and recovery should be reviewed supportively."
        )
    if body_fat_change_pct >= 0.4 and "sculpt" in primary_goal:
        reasons.append(
            f"Body-fat percentage increased by {body_fat_change_pct:.1f} points, so the sculpting plan should tighten nutrition and conditioning support."
        )

    return reasons


def _should_revise(
    safety_risk: str,
    engagement_risk: str,
    latest_feedback: dict,
    reasons: list[str],
) -> bool:
    fatigue_level = int(latest_feedback.get("fatigue_level", 0))
    pain_level = int(latest_feedback.get("pain_level", 0))
    adherence_score = float(latest_feedback.get("adherence_score", 1.0))

    if safety_risk == "high" or engagement_risk == "high":
        return True
    if pain_level >= 4 or fatigue_level >= 8:
        return True
    if adherence_score < 0.6:
        return True
    return bool(reasons and (safety_risk == "medium" or engagement_risk == "medium"))


def _build_summary(
    *,
    objective_score: float,
    llm_score: float,
    safety_risk: str,
    engagement_risk: str,
    should_revise: bool,
    reasons: list[str],
) -> str:
    action = "Revision recommended." if should_revise else "Current plan can continue."
    reason_text = reasons[0] if reasons else "No major issues detected."
    return (
        f"Objective score: {objective_score}/100. "
        f"Plan fit score: {llm_score}/100. "
        f"Safety risk: {safety_risk}. "
        f"Engagement risk: {engagement_risk}. "
        f"{action} {reason_text}"
    )


def _append_history_item(*, history: list[dict], item: dict, date_key: str) -> list[dict]:
    if not item:
        return list(history)

    updated_history = list(history)
    item_date = item.get(date_key)
    if updated_history and item_date and updated_history[-1].get(date_key) == item_date:
        updated_history[-1] = item
        return updated_history

    updated_history.append(item)
    return updated_history


def _compute_weight_change(current_state: dict, state_history: list[dict]) -> float:
    try:
        current_weight = float(current_state.get("weight_kg"))
    except (TypeError, ValueError):
        return 0.0

    if not state_history:
        return 0.0

    previous_state = state_history[-1]
    try:
        previous_weight = float(previous_state.get("weight_kg"))
    except (TypeError, ValueError):
        return 0.0
    return round(current_weight - previous_weight, 2)


def _compute_body_fat_change(current_state: dict, state_history: list[dict]) -> float:
    try:
        current_body_fat = float(current_state.get("body_fat_pct"))
    except (TypeError, ValueError):
        return 0.0

    if not state_history:
        return 0.0

    previous_state = state_history[-1]
    try:
        previous_body_fat = float(previous_state.get("body_fat_pct"))
    except (TypeError, ValueError):
        return 0.0
    return round(current_body_fat - previous_body_fat, 2)


def _has_substantive_feedback(latest_feedback: dict) -> bool:
    if not latest_feedback:
        return False
    if latest_feedback.get("completed_workouts"):
        return True
    if latest_feedback.get("pain_points") or latest_feedback.get("soreness_areas"):
        return True
    notes = str(latest_feedback.get("performance_notes", "")).strip()
    manual_notes = str(latest_feedback.get("manual_log", {}).get("notes", "")).strip()
    return bool(notes or manual_notes)


def _sanitize_risk(value: object, fallback: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"low", "medium", "high"}:
        return normalized
    return fallback


def _sanitize_score(value: object, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return round(max(0.0, min(numeric, 100.0)), 1)


def _sanitize_reasons(value: object, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return fallback
    reasons = [str(item) for item in value if str(item).strip()]
    return reasons or fallback
