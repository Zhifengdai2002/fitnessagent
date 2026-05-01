"""Coach ReAct Agent node.

Replaces the 5-LLM-call pipeline in coach_chat_service with a single
tool-calling loop: understand → call tools → respond.
"""

from __future__ import annotations

import json
from typing import Any

from agent.llm import get_model_client, load_prompt, load_settings
from agent.services.coach_tools import (
    COACH_NATIVE_TOOLS,
    build_chat_context,
    execute_coach_tool_call,
    sanitize_tool_arguments,
)

_MAX_STEPS = 4

_HARD_SAFETY_PHRASES = [
    "chest pain", "heart attack", "can't breathe", "cannot breathe",
    "difficulty breathing", "fainting", "unconscious",
    "胸痛", "心脏病", "无法呼吸", "呼吸困难", "晕倒",
]


def coach_react_node(user_message: str, session_state: dict[str, Any]) -> str:
    """ReAct Coach entry point. Hard safety check runs before any LLM call."""

    msg_lower = user_message.lower()
    for phrase in _HARD_SAFETY_PHRASES:
        if phrase in msg_lower:
            return (
                "It sounds like you may be experiencing a medical emergency. "
                "Please stop all activity immediately and consult a qualified healthcare professional. "
                "Do not resume exercising until you have been cleared by a doctor."
            )

    result = session_state.get("agent_result") or {}
    system_prompt = _build_system_prompt(result, session_state)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in session_state.get("assistant_chat_messages", [])[-6:]
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_message},
    ]

    client = get_model_client()
    settings = load_settings()

    tool_context: dict[str, Any] = {
        "previous_result": result,
        "session_state": session_state,
        "user_message": user_message,
        "profile_inputs": session_state.get("profile_inputs") or {},
    }

    for _ in range(_MAX_STEPS):
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=messages,
            tools=COACH_NATIVE_TOOLS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=800,
            extra_body={"thinking": {"type": "disabled"}},
        )
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            return _extract_text(msg) or "Done. Let me know if you need anything else."

        # Add the full assistant message with ALL tool calls at once (required by API format)
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in tool_calls
            ],
        })

        # Execute every tool call and append its result; refresh previous_result after each
        # so the next tool operates on the already-mutated plan, not the stale original.
        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                raw_args = json.loads(tc.function.arguments or "{}")
            except Exception:
                raw_args = {}
            arguments = sanitize_tool_arguments(tool_name, raw_args)
            observation = execute_coach_tool_call(
                {"tool_name": tool_name, "arguments": arguments}, tool_context
            ) or "Done."
            tool_context["previous_result"] = session_state.get("agent_result") or tool_context["previous_result"]
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": observation,
            })

    # Max steps reached — ask for a plain-text wrap-up without further tool calls.
    messages.append({
        "role": "user",
        "content": "Please give me a brief summary of what was changed.",
    })
    response = client.chat.completions.create(
        model=settings.model_name,
        messages=messages,
        temperature=0.4,
        max_tokens=400,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return _extract_text(response.choices[0].message) or "Your plan has been updated."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_system_prompt(result: dict[str, Any], session_state: dict[str, Any]) -> str:
    parts = [load_prompt("coach_react_prompt.txt")]
    parts.append(f"\n--- App State ---\n{build_chat_context(result, session_state)}")
    summary = str(session_state.get("conversation_summary") or "").strip()
    if summary:
        parts.append(f"\n--- Conversation Summary ---\n{summary}")
    return "\n".join(parts)


def _extract_text(msg: Any) -> str:
    content = getattr(msg, "content", "") or ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
            else:
                text = getattr(item, "text", "") or getattr(item, "content", "") or ""
            if text:
                parts.append(str(text))
        return " ".join(parts).strip()
    return str(content).strip()


