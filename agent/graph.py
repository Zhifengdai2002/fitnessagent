"""LangGraph workflow assembly for FitnessAgent."""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from agent.nodes import feedback_evaluation_node, plan_generation_node
from agent.state import FitnessAgentState, create_initial_state


def route_after_evaluation(state: FitnessAgentState) -> Literal["planner", "end"]:
    """Decide whether the workflow should loop back to the planner."""

    revision_attempts = len(state.get("plan_history", []))
    if state.get("needs_revision") and revision_attempts < 1:
        return "planner"
    return "end"


def build_workflow() -> StateGraph:
    """Create the uncompiled workflow graph."""

    workflow = StateGraph(FitnessAgentState)
    workflow.add_node("planner", plan_generation_node)
    workflow.add_node("evaluator", feedback_evaluation_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "evaluator")
    workflow.add_conditional_edges(
        "evaluator",
        route_after_evaluation,
        {
            "planner": "planner",
            "end": END,
        },
    )
    return workflow


def compile_workflow(checkpointer: InMemorySaver | None = None):
    """Compile the workflow into an executable LangGraph app."""

    workflow = build_workflow()
    return workflow.compile(checkpointer=checkpointer or InMemorySaver())


def create_default_app():
    """Return a compiled app with in-memory checkpointing."""

    return compile_workflow()


def run_agent(state: FitnessAgentState | None = None):
    """Run the compiled workflow once for a provided state."""

    app = create_default_app()
    initial_state = state or create_initial_state()
    thread_id = initial_state.get("thread_id") or "fitness-agent-demo"
    return app.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )
