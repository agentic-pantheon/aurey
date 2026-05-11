"""Factory for the compiled Deep Agents graph (optional ``deepagents`` dependency surface)."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from aurey.reasoning.harness import ensure_aurey_wallet_harness, resolve_harness_model_spec
from aurey.runtime import AureyRuntime
from aurey.tools.agent_tools import build_aurey_subgraph_tools

try:
    from deepagents import create_deep_agent as _create_deep_agent_impl
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _create_deep_agent_impl = None


AUREY_DEEP_USER_PROMPT = (
    "You are Aurey's planner for on-chain and Alchemy-backed reads, swap preparation, and "
    "transaction execution.\n"
    "Rules:\n"
    "- Call tools with structured arguments only (no opaque JSON blobs).\n"
    "- Never ask the user to paste private keys or raw RPC URLs; paths are resolved server-side.\n"
    "- After a successful `tx_prepare_*` (`ok` true), calling `tx_execute` REQUIRES the argument "
    "`envelope` set to `result.envelope` from that prepare output, unchanged. Do not call "
    "`tx_execute` without `envelope`.\n"
    "- Use **request_user_input** only when required fields are missing."
)


def _import_deepagents_create_agent():
    """Load Deep Agents entrypoints; single choke point for optional dependency / API drift."""

    if _create_deep_agent_impl is None:
        raise RuntimeError(
            "The 'deepagents' package is required to build an Aurey deep agent. "
            "Install project dependencies (see pyproject.toml)."
        )
    try:
        from deepagents.middleware.filesystem import FilesystemPermission
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "deepagents middleware (filesystem permissions) is unavailable. "
            "Upgrade or reinstall the 'deepagents' package."
        ) from exc
    return _create_deep_agent_impl, FilesystemPermission


def create_aurey_deep_agent(
    runtime: AureyRuntime,
    *,
    model: str | BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    extra_system_prompt: str | None = None,
    name: str = "aurey_deep_agent",
) -> CompiledStateGraph[Any, Any, Any]:
    """Compile Deep Agents with subgraph-backed tools and optional MemorySaver checkpointer."""

    create_deep_agent, FilesystemPermission = _import_deepagents_create_agent()

    harness_spec = resolve_harness_model_spec(model)
    ensure_aurey_wallet_harness(harness_spec)

    deny_all_fs = FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny")

    tools = build_aurey_subgraph_tools(runtime)
    user_sys = AUREY_DEEP_USER_PROMPT.strip()
    if extra_system_prompt and extra_system_prompt.strip():
        user_sys = f"{user_sys}\n\n{extra_system_prompt.strip()}"

    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=user_sys,
        checkpointer=checkpointer,
        permissions=[deny_all_fs],
        name=name,
    )


__all__ = ["AUREY_DEEP_USER_PROMPT", "create_aurey_deep_agent"]
