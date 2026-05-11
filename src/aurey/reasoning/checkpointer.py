"""LangGraph checkpoint helpers (in-memory session/thread identity)."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver


def make_memory_checkpointer() -> BaseCheckpointSaver:
    """Return a fresh in-memory saver; key runs with ``config['configurable']['thread_id']``."""

    return MemorySaver()


def thread_config(session_id: str, **extra: Any) -> dict[str, Any]:
    """Build ``invoke`` / ``ainvoke`` config with a stable thread id plus optional fields."""

    return {"configurable": {"thread_id": session_id, **extra}}


__all__ = ["make_memory_checkpointer", "thread_config"]
