"""Deep agent harness, checkpointer helpers, and compiled graph factory."""

from __future__ import annotations

from typing import Any

from aurey.reasoning.checkpointer import make_memory_checkpointer, thread_config
from aurey.reasoning.harness import (
    AUREY_DEEP_HARNESS_BASE,
    ensure_aurey_wallet_harness,
    resolve_harness_model_spec,
)

__all__ = [
    "AUREY_DEEP_HARNESS_BASE",
    "AUREY_DEEP_USER_PROMPT",
    "create_aurey_deep_agent",
    "ensure_aurey_wallet_harness",
    "make_memory_checkpointer",
    "resolve_harness_model_spec",
    "thread_config",
]


def __getattr__(name: str) -> Any:
    """Lazy-load deep agent to avoid import cycles (``invoke`` → ``runtime`` → ``graphs``)."""

    if name == "create_aurey_deep_agent":
        from aurey.reasoning.deep_agent import create_aurey_deep_agent

        return create_aurey_deep_agent
    if name == "AUREY_DEEP_USER_PROMPT":
        from aurey.reasoning.deep_agent import AUREY_DEEP_USER_PROMPT

        return AUREY_DEEP_USER_PROMPT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
