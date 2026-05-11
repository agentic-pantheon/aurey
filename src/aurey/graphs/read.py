"""LangGraph: EVM reads + known-address resolution (Mercury-parity subset)."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

from aurey.custody.errors import SecretNotFoundError, SecretStoreUnavailableError
from aurey.graphs.chains import alchemy_rpc_url_for_chain, chain_id_for, chain_info
from aurey.graphs.evm_codec import normalize_evm_address
from aurey.graphs.results import (
    Erc20ReadPlaceholder,
    GraphErrorBody,
    KnownAddressResult,
    NativeBalanceResult,
)
from aurey.runtime import AureyRuntime

KNOWN_TICKER_ADDRESSES: dict[tuple[str, str], str] = {
    ("ethereum", "usdc"): "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    ("base", "usdc"): "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    ("ethereum", "weth"): "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
}


class ReadGraphInput(BaseModel):
    operation: Literal["native_balance", "known_address", "erc20_balance"]
    chain: str = Field(min_length=1)
    wallet_address: str | None = None
    token_address: str | None = None
    known_ticker: str | None = None


class ReadGraphState(TypedDict, total=False):
    input: dict[str, Any]
    error: dict[str, Any]
    result: dict[str, Any]


def _validation_error(exc: ValidationError) -> dict[str, Any]:
    return GraphErrorBody(
        code="invalid_input",
        message="Read graph input failed validation.",
        details={"errors": exc.errors()},
    ).model_dump()


def _validate_node(state: ReadGraphState) -> ReadGraphState:
    try:
        parsed = ReadGraphInput.model_validate(state.get("input") or {})
    except ValidationError as exc:
        return {"error": _validation_error(exc)}

    chain = parsed.chain
    if chain_info(chain) is None:
        return {
            "error": GraphErrorBody(
                code="unsupported_chain",
                message=f"Unsupported chain '{chain}'.",
            ).model_dump()
        }

    if parsed.operation == "native_balance":
        if not parsed.wallet_address:
            return {
                "error": GraphErrorBody(
                    code="invalid_input",
                    message="wallet_address is required for native_balance.",
                ).model_dump()
            }
        try:
            normalize_evm_address(parsed.wallet_address)
        except ValueError as exc:
            return {
                "error": GraphErrorBody(
                    code="invalid_input",
                    message="Invalid wallet address.",
                    details={"reason": str(exc)},
                ).model_dump()
            }
    if parsed.operation == "known_address":
        if not parsed.known_ticker:
            return {
                "error": GraphErrorBody(
                    code="invalid_input",
                    message="known_ticker is required for known_address.",
                ).model_dump()
            }
    if parsed.operation == "erc20_balance":
        if not parsed.wallet_address or not parsed.token_address:
            return {
                "error": GraphErrorBody(
                    code="invalid_input",
                    message="wallet_address and token_address are required for erc20_balance.",
                ).model_dump()
            }
        try:
            normalize_evm_address(parsed.wallet_address)
            normalize_evm_address(parsed.token_address)
        except ValueError as exc:
            return {
                "error": GraphErrorBody(
                    code="invalid_input",
                    message="Invalid wallet address.",
                    details={"reason": str(exc)},
                ).model_dump()
            }

    return {}


def _execute_node(runtime: AureyRuntime, state: ReadGraphState) -> ReadGraphState:
    if state.get("error"):
        return {}

    parsed = ReadGraphInput.model_validate(state["input"])
    chain = parsed.chain.strip().lower()

    if parsed.operation == "known_address":
        ticker = parsed.known_ticker or ""
        key = (chain, ticker.strip().lower())
        addr = KNOWN_TICKER_ADDRESSES.get(key)
        if addr is None:
            return {
                "error": GraphErrorBody(
                    code="invalid_input",
                    message="Unknown ticker for this chain.",
                    details={"ticker": ticker, "chain": chain},
                ).model_dump()
            }
        cid = chain_id_for(chain)
        assert cid is not None
        result = KnownAddressResult(chain=chain, ticker=ticker, resolved_address=addr)
        return {"result": result.model_dump()}

    if parsed.operation == "erc20_balance":
        placeholder = Erc20ReadPlaceholder(
            chain=chain,
            operation="erc20_balance",
            token_address=normalize_evm_address(parsed.token_address or ""),
        )
        return {"result": placeholder.model_dump()}

    alchemy_path = runtime.settings.alchemy_api_secret_path
    if not alchemy_path:
        return {
            "error": GraphErrorBody(
                code="secret_not_configured",
                message="Alchemy API secret path is not configured.",
                details={"chain": chain},
            ).model_dump()
        }

    try:
        alchemy_key = runtime.secret_store.get_secret(alchemy_path).reveal()
    except SecretNotFoundError:
        return {
            "error": GraphErrorBody(
                code="secret_not_found",
                message="Alchemy API secret could not be resolved.",
                details={"secret_kind": "alchemy_api"},
            ).model_dump()
        }
    except SecretStoreUnavailableError:
        return {
            "error": GraphErrorBody(
                code="secret_unavailable",
                message="Secret store unavailable while resolving Alchemy API key.",
                details={"secret_kind": "alchemy_api"},
            ).model_dump()
        }

    rpc_url = alchemy_rpc_url_for_chain(chain, alchemy_key)
    if rpc_url is None:
        return {
            "error": GraphErrorBody(
                code="unsupported_chain",
                message="No Alchemy RPC mapping for this chain.",
                details={"chain": chain},
            ).model_dump()
        }

    # The derived RPC URL contains the Alchemy key and is kept out of graph state and outputs.
    try:
        rpc = runtime.evm_rpc_factory(rpc_url)
        wallet = normalize_evm_address(parsed.wallet_address or "")
        balance_hex = rpc.call("eth_getBalance", [wallet, "latest"])
        if not isinstance(balance_hex, str):
            raise TypeError("unexpected balance type")
    except Exception:
        return {
            "error": GraphErrorBody(
                code="rpc_error",
                message="RPC read failed.",
            ).model_dump()
        }

    cid = chain_id_for(chain)
    assert cid is not None
    result = NativeBalanceResult(
        chain=chain,
        chain_id=cid,
        wallet_address=wallet,
        balance_wei_hex=balance_hex,
    )
    return {"result": result.model_dump()}


def _route_after_validate(state: ReadGraphState) -> Literal["execute", "done"]:
    return "done" if state.get("error") else "execute"


def build_read_graph(runtime: AureyRuntime):
    g: StateGraph = StateGraph(ReadGraphState)

    def validate(state: ReadGraphState) -> ReadGraphState:
        return _validate_node(state)

    def execute(state: ReadGraphState) -> ReadGraphState:
        return _execute_node(runtime, state)

    g.add_node("validate", validate)
    g.add_node("execute", execute)
    g.set_entry_point("validate")
    g.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"execute": "execute", "done": END},
    )
    g.add_edge("execute", END)
    return g.compile()
