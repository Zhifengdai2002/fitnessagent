"""Evaluator node — currently a graph hook for future RL reward back-fill.

Plan decisions and level upgrades are handled in feedback_service.py at
cycle rollover time. This node keeps the planner → evaluator → END edge
alive so reward logic can be added here later without restructuring the graph.
"""

from __future__ import annotations

from agent.state import FitnessAgentState


def feedback_evaluation_node(state: FitnessAgentState) -> FitnessAgentState:
    return {
        "evaluation_result": {},
        "needs_revision": False,
        "revision_reason": "",
        "feedback_history": list(state.get("feedback_history", [])),
        "state_history": list(state.get("state_history", [])),
    }
