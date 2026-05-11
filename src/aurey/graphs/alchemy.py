"""LangGraph: Alchemy prices, portfolio, and transfer history (HTTP + key via SecretStore)."""

from __future__ import annotations

from typing import Any, Literal, TypedDict
from urllib.parse import quote

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

from aurey.custody.errors import SecretNotFoundError, SecretStoreUnavailableError
from aurey.graphs.chains import chain_info
from aurey.graphs.evm_codec import normalize_evm_address
from aurey.graphs.results import (
    AlchemyPortfolioResult,
    AlchemyTokenPricesResult,
    AlchemyTransferHistoryResult,
    GraphErrorBody,
)
from aurey.runtime import AureyRuntime


def _alchemy_rpc_host(chain: str) -> str | None:
    name = chain.strip().lower()
    if name == "ethereum":
        return "eth-mainnet"
    if name == "base":
        return "base-mainnet"
    return None


class AlchemyGraphInput(BaseModel):
    operation: Literal["token_prices", "portfolio_tokens", "transfer_history"]
    chain: str = Field(min_length=1)
    wallet_address: str = Field(min_length=1)
    token_addresses: list[str] | None = None


class AlchemyGraphState(TypedDict, total=False):
    input: dict[str, Any]
    error: dict[str, Any]
    result: dict[str, Any]


def _validation_error(exc: ValidationError) -> dict[str, Any]:
    return GraphErrorBody(
        code="invalid_input",
        message="Alchemy graph input failed validation.",
        details={"errors": exc.errors()},
    ).model_dump()


def _validate_node(state: AlchemyGraphState) -> AlchemyGraphState:
    try:
        parsed = AlchemyGraphInput.model_validate(state.get("input") or {})
    except ValidationError as exc:
        return {"error": _validation_error(exc)}

    if chain_info(parsed.chain) is None:
        return {
            "error": GraphErrorBody(
                code="unsupported_chain",
                message=f"Unsupported chain '{parsed.chain}'.",
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

    if parsed.operation == "token_prices":
        addrs = parsed.token_addresses or []
        if not addrs:
            return {
                "error": GraphErrorBody(
                    code="invalid_input",
                    message="token_addresses is required for token_prices.",
                ).model_dump()
            }
    return {}


def _resolve_alchemy_key(runtime: AureyRuntime) -> tuple[str | None, dict[str, Any] | None]:
    path = runtime.settings.alchemy_api_secret_path
    if not path:
        err = GraphErrorBody(
            code="secret_not_configured",
            message="Alchemy API secret path is not configured.",
        ).model_dump()
        return None, err
    try:
        return runtime.secret_store.get_secret(path).reveal(), None
    except SecretNotFoundError:
        err = GraphErrorBody(
            code="secret_not_found",
            message="Alchemy API secret could not be resolved.",
            details={"secret_kind": "alchemy_api"},
        ).model_dump()
        return None, err
    except SecretStoreUnavailableError:
        err = GraphErrorBody(
            code="secret_unavailable",
            message="Secret store unavailable while resolving Alchemy API key.",
            details={"secret_kind": "alchemy_api"},
        ).model_dump()
        return None, err


def _execute_node(runtime: AureyRuntime, state: AlchemyGraphState) -> AlchemyGraphState:
    if state.get("error"):
        return {}

    parsed = AlchemyGraphInput.model_validate(state["input"])
    api_key, err = _resolve_alchemy_key(runtime)
    if err is not None:
        return {"error": err}
    assert api_key is not None

    chain = parsed.chain.strip().lower()
    wallet = normalize_evm_address(parsed.wallet_address)

    try:
        if parsed.operation == "token_prices":
            addrs = [normalize_evm_address(a) for a in (parsed.token_addresses or [])]
            joined = "%2C".join(quote(a, safe="") for a in addrs)
            url = f"https://api.g.alchemy.com/prices/v1/{api_key}/tokens/by-address?addresses={joined}"
            payload = runtime.http.request_json(method="GET", url=url, headers=None, json_body=None)
            raw_prices = payload.get("data") or payload.get("prices") or {}
            if not isinstance(raw_prices, dict):
                raise ValueError("unexpected prices shape")
            prices: dict[str, str] = {}
            for k, v in raw_prices.items():
                prices[str(k)] = str(v)
            result = AlchemyTokenPricesResult(chain=chain, prices_by_address=prices)
            return {"result": result.model_dump()}

        if parsed.operation == "portfolio_tokens":
            enc_wallet = quote(wallet, safe="")
            url = f"https://api.g.alchemy.com/portfolio/v1/{api_key}/wallets/{enc_wallet}/tokens"
            payload = runtime.http.request_json(method="GET", url=url, headers=None, json_body=None)
            tokens = payload.get("tokens")
            if not isinstance(tokens, list):
                tokens = payload.get("data") if isinstance(payload.get("data"), list) else []
            if not isinstance(tokens, list):
                raise ValueError("unexpected portfolio shape")
            result = AlchemyPortfolioResult(chain=chain, wallet_address=wallet, tokens=tokens)
            return {"result": result.model_dump()}

        host = _alchemy_rpc_host(chain)
        if host is None:
            return {
                "error": GraphErrorBody(
                    code="unsupported_chain",
                    message="No Alchemy network mapping for this chain.",
                    details={"chain": chain},
                ).model_dump()
            }

        rpc_url = f"https://{host}.g.alchemy.com/v2/{api_key}"
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "alchemy_getAssetTransfers",
            "params": [
                {
                    "fromBlock": "0x0",
                    "toBlock": "latest",
                    "category": ["external", "erc20"],
                    "withMetadata": False,
                    "excludeZeroValue": True,
                    "maxCount": "0x64",
                    "fromAddress": wallet,
                }
            ],
        }
        body = runtime.http.request_json(
            method="POST", url=rpc_url, headers={"Content-Type": "application/json"}, json_body=req
        )
        rpc_result = body.get("result") or {}
        transfers = rpc_result.get("transfers")
        if transfers is None:
            transfers = []
        if not isinstance(transfers, list):
            raise ValueError("unexpected transfers shape")
        xfer_models: list[dict[str, Any]] = []
        for row in transfers:
            if isinstance(row, dict):
                xfer_models.append(dict(row))
        result = AlchemyTransferHistoryResult(
            chain=chain, wallet_address=wallet, transfers=xfer_models
        )
        return {"result": result.model_dump()}
    except Exception:
        return {
            "error": GraphErrorBody(
                code="http_error",
                message="Alchemy request failed.",
            ).model_dump()
        }


def _route_after_validate(state: AlchemyGraphState) -> Literal["execute", "done"]:
    return "done" if state.get("error") else "execute"


def build_alchemy_graph(runtime: AureyRuntime):
    g: StateGraph = StateGraph(AlchemyGraphState)

    def validate(state: AlchemyGraphState) -> AlchemyGraphState:
        return _validate_node(state)

    def execute(state: AlchemyGraphState) -> AlchemyGraphState:
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
