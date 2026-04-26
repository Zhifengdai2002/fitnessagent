"""Runtime helpers that let the API use app services without Streamlit sessions."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import app as streamlit_app


class BackendStreamlitAdapter:
    """Small adapter for app functions that expect `st.session_state`."""

    def __init__(self, session_state: dict[str, Any]) -> None:
        self.session_state = session_state

    def error(self, message: object) -> None:
        raise RuntimeError(str(message))

    def warning(self, message: object) -> None:
        self.session_state["last_warning"] = str(message)

    def info(self, message: object) -> None:
        self.session_state["last_info"] = str(message)

    def success(self, message: object) -> None:
        self.session_state["last_success"] = str(message)


@contextmanager
def backend_streamlit_context(session_state: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Temporarily replace app.st with a backend-safe adapter."""

    runtime_state: dict[str, Any] = session_state if session_state is not None else {}
    original_st = streamlit_app.st
    streamlit_app.st = BackendStreamlitAdapter(runtime_state)  # type: ignore[assignment]
    try:
        yield runtime_state
    finally:
        streamlit_app.st = original_st
