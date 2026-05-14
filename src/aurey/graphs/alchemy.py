"""LangGraph: Alchemy prices, portfolio, and transfer history (HTTP + key via SecretStore).

REST:
  - Prices — POST https://api.g.alchemy.com/prices/v1/{apiKey}/tokens/by-address
  - Portfolio ("Tokens By Wallet") — POST https://api.g.alchemy.com/data/v1/{apiKey}/assets/tokens/by-address

Transfers JSON-RPC:
  - POST https://{network}.g.alchemy.com/v2/{apiKey}  method ``alchemy_getAssetTransfers``
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

from aurey.custody.errors import (
    SecretNotFoundError,
    SecretStoreUnavailableError,
    secret_unavailable_graph_details,
)
from aurey.graphs.chains import chain_info
from aurey.graphs.checkpoint_serde import uint256_checkpoint_str
from aurey.graphs.evm_codec import format_token_units, normalize_evm_address, parse_evm_uint
from aurey.graphs.results import (
    AlchemyPortfolioResult,
    AlchemyTokenPricesResult,
    AlchemyTransferHistoryResult,
    GraphErrorBody,
)
from aurey.runtime import AureyRuntime


def _alchemy_network(chain: str) -> str | None:
    info = chain_info(chain.strip().lower())
    return None if info is None else info.alchemy_network


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

    if chain_info(parsed.chain.strip().lower()) is None:
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
    except SecretStoreUnavailableError as exc:
        err = GraphErrorBody(
            code="secret_unavailable",
            message="Secret store unavailable while resolving Alchemy API key.",
            details=secret_unavailable_graph_details(secret_kind="alchemy_api", exc=exc),
        ).model_dump()
        return None, err


def _pick_usd_price(prices_raw: Any) -> str | None:
    """First USD price ``value``, else first available ``value``."""

    if not isinstance(prices_raw, list):
        return None
    fallback: str | None = None
    for row in prices_raw:
        if not isinstance(row, dict):
            continue
        val = row.get("value")
        if val is None:
            continue
        sval = str(val)
        cur = str(row.get("currency", "")).strip().upper()
        if cur == "USD":
            return sval
        if fallback is None:
            fallback = sval
    return fallback


def _parse_prices_payload(payload: dict[str, Any], token_addrs: list[str]) -> dict[str, str]:
    """Map normalized contract address → price string per Prices API ``data`` array."""

    rows = payload.get("data")
    if isinstance(rows, dict):
        rows = rows.get("prices") or rows.get("results")
    if not isinstance(rows, list):
        return {}

    want = {normalize_evm_address(a) for a in token_addrs}
    out: dict[str, str] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            addr = normalize_evm_address(str(row.get("address") or ""))
        except ValueError:
            continue
        if want and addr not in want:
            continue
        err = row.get("error")
        if err:
            out[addr] = f"<error:{err}>"
            continue
        price = _pick_usd_price(row.get("prices"))
        if price is not None:
            out[addr] = price

    return out


def _coerce_decimals(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = parse_evm_uint(value) if isinstance(value, int | str) else int(value)
    except (TypeError, ValueError):
        return None
    return parsed if 0 <= parsed <= 255 else None


def _token_decimals(token: dict[str, Any]) -> int | None:
    decimals = _coerce_decimals(token.get("decimals"))
    if decimals is not None:
        return decimals

    metadata = token.get("tokenMetadata")
    if isinstance(metadata, dict):
        return _coerce_decimals(metadata.get("decimals"))
    return None


def _normalize_portfolio_token(token: dict[str, Any]) -> dict[str, Any]:
    out = dict(token)
    raw_balance = token.get("tokenBalance")
    if not isinstance(raw_balance, int | str):
        return out

    try:
        balance_raw = parse_evm_uint(raw_balance)
    except ValueError:
        return out

    decimals = _token_decimals(token)
    # String: uint256 may exceed msgpack/orjson int64 range in LangGraph checkpoints.
    out["balance_raw"] = uint256_checkpoint_str(balance_raw)
    out["decimals"] = decimals
    out["balance_decimal"] = (
        format_token_units(balance_raw, decimals) if decimals is not None else None
    )
    return out


def _parse_portfolio_tokens(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Portfolio API nests tokens under ``data.tokens``."""

    data = payload.get("data")
    if isinstance(data, dict):
        tokens = data.get("tokens")
        if isinstance(tokens, list):
            return [_normalize_portfolio_token(x) for x in tokens if isinstance(x, dict)]
        return []
    return []


def _transfer_param_block(
    wallet: str,
    *,
    direction: Literal["from", "to"],
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "fromBlock": "0x0",
        "toBlock": "latest",
        "category": ["external", "erc20"],
        "withMetadata": True,
        "excludeZeroValue": True,
        "maxCount": "0x64",
        "order": "desc",
    }
    if direction == "from":
        block["fromAddress"] = wallet
    else:
        block["toAddress"] = wallet
    return block


def _post_asset_transfers(
    runtime: AureyRuntime,
    *,
    rpc_url: str,
    param_block: dict[str, Any],
) -> list[dict[str, Any]]:
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "alchemy_getAssetTransfers",
        "params": [param_block],
    }
    body = runtime.http.request_json(
        method="POST",
        url=rpc_url,
        headers={"Content-Type": "application/json"},
        json_body=req,
    )
    rpc_result = body.get("result") or {}
    if isinstance(rpc_result, str):
        return []
    transfers = rpc_result.get("transfers")
    if transfers is None:
        transfers = []
    if not isinstance(transfers, list):
        raise ValueError("unexpected transfers shape")
    return [dict(row) for row in transfers if isinstance(row, dict)]


def _merge_transfer_rows(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_uid: dict[str, dict[str, Any]] = {}
    for row in rows_a + rows_b:
        uid = str(row.get("uniqueId") or "")
        key = uid if uid else str(id(row))
        by_uid.setdefault(key, row)

    def sort_key(row: dict[str, Any]) -> int:
        raw = row.get("blockNum", "0x0")
        try:
            return int(str(raw), 16)
        except ValueError:
            return 0

    merged = sorted(by_uid.values(), key=sort_key, reverse=True)
    return merged


def _execute_node(runtime: AureyRuntime, state: AlchemyGraphState) -> AlchemyGraphState:
    if state.get("error"):
        return {}

    parsed = AlchemyGraphInput.model_validate(state["input"])
    api_key, err = _resolve_alchemy_key(runtime)
    if err is not None:
        return {"error": err}
    assert api_key is not None

    chain = parsed.chain.strip().lower()
    network = _alchemy_network(chain)
    if network is None:
        return {
            "error": GraphErrorBody(
                code="unsupported_chain",
                message="No Alchemy network mapping for this chain.",
                details={"chain": chain},
            ).model_dump()
        }

    wallet = normalize_evm_address(parsed.wallet_address)
    hdr_json = {"Content-Type": "application/json"}

    try:
        if parsed.operation == "token_prices":
            addrs_norm = [normalize_evm_address(a) for a in (parsed.token_addresses or [])]
            url = f"https://api.g.alchemy.com/prices/v1/{api_key}/tokens/by-address"
            payload = runtime.http.request_json(
                method="POST",
                url=url,
                headers=hdr_json,
                json_body={
                    "addresses": [{"network": network, "address": a} for a in addrs_norm],
                },
            )
            if not isinstance(payload, dict):
                raise ValueError("unexpected prices envelope")
            prices = _parse_prices_payload(payload, addrs_norm)
            result = AlchemyTokenPricesResult(chain=chain, prices_by_address=prices)
            return {"result": result.model_dump()}

        if parsed.operation == "portfolio_tokens":
            url = f"https://api.g.alchemy.com/data/v1/{api_key}/assets/tokens/by-address"
            payload = runtime.http.request_json(
                method="POST",
                url=url,
                headers=hdr_json,
                json_body={
                    "addresses": [{"address": wallet, "networks": [network]}],
                    "withMetadata": True,
                    "withPrices": True,
                    "includeNativeTokens": True,
                    "includeErc20Tokens": True,
                },
            )
            tokens = _parse_portfolio_tokens(payload if isinstance(payload, dict) else {})
            result = AlchemyPortfolioResult(chain=chain, wallet_address=wallet, tokens=tokens)
            return {"result": result.model_dump()}

        rpc_url = f"https://{network}.g.alchemy.com/v2/{api_key}"
        sent = _post_asset_transfers(
            runtime,
            rpc_url=rpc_url,
            param_block=_transfer_param_block(wallet, direction="from"),
        )
        received = _post_asset_transfers(
            runtime,
            rpc_url=rpc_url,
            param_block=_transfer_param_block(wallet, direction="to"),
        )
        xfer_models = _merge_transfer_rows(sent, received)
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
