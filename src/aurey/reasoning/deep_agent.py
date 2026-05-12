"""Factory for the compiled Deep Agents graph (optional ``deepagents`` dependency surface)."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from aurey.graphs.evm_codec import normalize_evm_address
from aurey.reasoning.harness import ensure_aurey_wallet_harness, resolve_harness_model_spec
from aurey.runtime import AureyRuntime
from aurey.settings import AureySettings
from aurey.tools.agent_tools import build_aurey_subgraph_tools

_log = logging.getLogger(__name__)

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
    "- Prefer **checksum** ``0x`` token contract addresses in ``swap_prepare`` "
    "(LiFi ``fromToken`` / ``toToken``). If a quote errors or goes stale, call ``swap_prepare`` "
    "again; optionally raise ``slippage`` (decimal, e.g. ``0.01``) or set ``order`` to "
    "``FASTEST`` / ``CHEAPEST`` per LiFi docs.\n"
    "- For LiFi **token swaps**, if `swap_prepare` includes `allowance`, you MUST broadcast an "
    "ERC-20 approval first: `tx_prepare_erc20_approval` on the same `chain` using "
    "`token_address=allowance.token_address`, `spender_address=allowance.spender_address`, "
    "`amount_wei=int(allowance.amount_raw)` (or larger), then `tx_execute` that envelope. "
    "If `allowance` is omitted, either LiFi did not require one or on-chain allowance was "
    "already sufficient—do not approve again in that case. "
    "After it confirms, call `tx_execute(prepared_id=result['prepared_id'])` from the original "
    "`swap_prepare` output. Do not copy LiFi calldata through the model. "
    "Skipping approval when `allowance` is set usually makes simulation revert. Do not pass raw "
    "LiFi `transaction_request` JSON alone to `tx_execute`.\n"
    "- For LiFi swaps, call `tx_execute` with `prepared_id` from `swap_prepare` or "
    "`tx_prepare_lifi_swap`; this keeps calldata server-side. For non-LiFi prepared txs, set "
    "`envelope` to `result['envelope']` from the prepare output, unchanged, and optionally "
    "`idempotency_key`. Never call `tx_execute` with only `idempotency_key`.\n"
    "- For ERC-20 `tx_prepare_*`, parameter `amount_wei` means **raw token units** for that "
    "token's decimals (USDC on Base = **6**: 0.01 USDC → `10000`, not `10**16`). "
    "Call **evm_get_erc20_decimals** for the token contract when decimals are not certain.\n"
    "- For **ENS names** (`vitalik.eth`, other ENS-style TLDs), call **`evm_resolve_ens`** "
    "(**`ethereum`** only). Use **`resolved_address`** as the `0x` recipient for "
    "**`tx_prepare_*`**, **`swap_prepare`**, and similar tools.\n"
    "- Use **request_user_input** only when required fields are missing."
)


def wallet_context_for_deep_agent_prompt(settings: AureySettings) -> str:
    """Return a suffix for the deep agent system prompt, or an empty string if unset or invalid."""

    raw = (settings.deep_agent_wallet_address or "").strip()
    if not raw:
        return ""
    try:
        addr = normalize_evm_address(raw)
    except ValueError:
        _log.warning(
            "Ignoring invalid AUREY_DEEP_AGENT_WALLET_ADDRESS (must be a 0x-prefixed EVM address).",
        )
        return ""
    return (
        "\n\nPersistent operator context: primary EVM wallet is "
        f"{addr}. Use it when the user says \"my wallet\" or omits wallet / "
        "`from` arguments unless they specify otherwise."
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
    user_sys += wallet_context_for_deep_agent_prompt(runtime.settings)
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


__all__ = [
    "AUREY_DEEP_USER_PROMPT",
    "create_aurey_deep_agent",
    "wallet_context_for_deep_agent_prompt",
]
