"""LangGraph: LiFi-style swap preparation (injectable HTTP; API key via SecretStore)."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

from aurey.custody.errors import SecretNotFoundError, SecretStoreUnavailableError
from aurey.graphs.chains import chain_id_for, chain_info
from aurey.graphs.evm_codec import normalize_evm_address
from aurey.graphs.results import GraphErrorBody, LiFiPreparedTx, SwapPrepareResult
from aurey.runtime import AureyRuntime


class SwapPrepareInput(BaseModel):
    from_chain: str = Field(min_length=1)
    to_chain: str = Field(min_length=1)
    from_asset: str = Field(min_length=1)
    to_asset: str = Field(min_length=1)
    from_amount_wei: str = Field(min_length=1, pattern=r"^[0-9]+$")
    from_address: str = Field(min_length=1)
    to_address: str = Field(min_length=1)


class SwapGraphState(TypedDict, total=False):
    input: dict[str, Any]
    error: dict[str, Any]
    result: dict[str, Any]


def _validation_error(exc: ValidationError) -> dict[str, Any]:
    return GraphErrorBody(
        code="invalid_input",
        message="Swap prepare input failed validation.",
        details={"errors": exc.errors()},
    ).model_dump()


def _validate_node(state: SwapGraphState) -> SwapGraphState:
    try:
        parsed = SwapPrepareInput.model_validate(state.get("input") or {})
    except ValidationError as exc:
        return {"error": _validation_error(exc)}

    for label, chain in (("from_chain", parsed.from_chain), ("to_chain", parsed.to_chain)):
        if chain_info(chain) is None:
            return {
                "error": GraphErrorBody(
                    code="unsupported_chain",
                    message=f"Unsupported {label} '{chain}'.",
                ).model_dump()
            }

    try:
        normalize_evm_address(parsed.from_address)
        normalize_evm_address(parsed.to_address)
    except ValueError as exc:
        return {
            "error": GraphErrorBody(
                code="invalid_input",
                message="Invalid swap address.",
                details={"reason": str(exc)},
            ).model_dump()
        }

    return {}


def _resolve_lifi_key(runtime: AureyRuntime) -> tuple[str | None, dict[str, Any] | None]:
    path = runtime.settings.lifi_api_secret_path
    if not path:
        err = GraphErrorBody(
            code="secret_not_configured",
            message="LiFi API secret path is not configured.",
        ).model_dump()
        return None, err
    try:
        return runtime.secret_store.get_secret(path).reveal(), None
    except SecretNotFoundError:
        err = GraphErrorBody(
            code="secret_not_found",
            message="LiFi API secret could not be resolved.",
            details={"secret_kind": "lifi_api"},
        ).model_dump()
        return None, err
    except SecretStoreUnavailableError:
        err = GraphErrorBody(
            code="secret_unavailable",
            message="Secret store unavailable while resolving LiFi API key.",
            details={"secret_kind": "lifi_api"},
        ).model_dump()
        return None, err


def _execute_node(runtime: AureyRuntime, state: SwapGraphState) -> SwapGraphState:
    if state.get("error"):
        return {}

    parsed = SwapPrepareInput.model_validate(state["input"])
    api_key, err = _resolve_lifi_key(runtime)
    if err is not None:
        return {"error": err}
    assert api_key is not None

    from_cid = chain_id_for(parsed.from_chain)
    to_cid = chain_id_for(parsed.to_chain)
    if from_cid is None or to_cid is None:
        return {
            "error": GraphErrorBody(
                code="unsupported_chain",
                message="Could not resolve LiFi chain ids.",
            ).model_dump()
        }

    url = f"{runtime.lifi_base_url.rstrip('/')}/v1/quote"
    body: dict[str, Any] = {
        "fromChain": from_cid,
        "toChain": to_cid,
        "fromToken": parsed.from_asset,
        "toToken": parsed.to_asset,
        "fromAmount": parsed.from_amount_wei,
        "fromAddress": normalize_evm_address(parsed.from_address),
        "toAddress": normalize_evm_address(parsed.to_address),
    }
    try:
        payload = runtime.http.request_json(
            method="POST",
            url=url,
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": api_key,
            },
            json_body=body,
        )
        tx_request = payload.get("transactionRequest") or payload.get("tx") or {}
        route_id = str(payload.get("routeId") or payload.get("id") or "lifi-route")
        if not isinstance(tx_request, dict):
            raise TypeError("unexpected transaction request shape")
        prepared = LiFiPreparedTx(route_id=route_id, transaction_request=dict(tx_request))
        result = SwapPrepareResult(prepared=prepared)
        return {"result": result.model_dump()}
    except Exception:
        return {
            "error": GraphErrorBody(
                code="swap_prepare_failed",
                message="LiFi swap preparation failed.",
            ).model_dump()
        }


def _route_after_validate(state: SwapGraphState) -> Literal["execute", "done"]:
    return "done" if state.get("error") else "execute"


def build_swap_prepare_graph(runtime: AureyRuntime):
    g: StateGraph = StateGraph(SwapGraphState)

    def validate(state: SwapGraphState) -> SwapGraphState:
        return _validate_node(state)

    def execute(state: SwapGraphState) -> SwapGraphState:
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
