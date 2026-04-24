"""Model client helpers for JSON-based planner and evaluator calls."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent.config import load_settings


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""

    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def get_model_client() -> OpenAI:
    """Create a cached OpenAI-compatible client for the configured provider."""

    settings = load_settings()
    if not settings.has_model_api_key:
        raise RuntimeError(
            "Missing model API key. Set MODEL_API_KEY in .env to call GLM-4.5-Air."
        )
    return OpenAI(
        api_key=settings.model_api_key,
        base_url=settings.model_base_url,
    )


def call_model_json(
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 3000,
) -> dict[str, Any]:
    """Call the configured model and parse a JSON object from the response."""

    settings = load_settings()
    client = get_model_client()

    response = client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={
            "thinking": {
                # JSON mode is far more reliable when visible thinking is disabled.
                "type": "disabled",
            }
        },
    )
    message = response.choices[0].message
    content = _coerce_message_content(message.content)
    if not content.strip():
        reasoning = getattr(message, "reasoning_content", "") or ""
        content = _extract_possible_json_from_reasoning(reasoning)
    parsed = _extract_json(content)
    if not isinstance(parsed, dict):
        raise ValueError("Model response did not contain a JSON object.")
    return parsed


def _coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _extract_json(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[index:])
            return parsed
        except json.JSONDecodeError:
            continue
    preview = stripped[:400]
    raise ValueError(f"Unable to parse JSON from model response: {preview}")


def _extract_possible_json_from_reasoning(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    for marker in ("{", "["):
        index = stripped.find(marker)
        if index != -1:
            return stripped[index:]
    return stripped
