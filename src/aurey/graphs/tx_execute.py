"""LangGraph: validate and execute prepared envelopes via :class:`~aurey.runtime.AureyRuntime`."""

from __future__ import annotations

import hashlib
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ValidationError

from aurey.custody.errors import SecretNotFoundError, SecretStoreUnavailableError
from aurey.graphs.ports import TxPipelinePort
from aurey.graphs.results import (
    GraphErrorBody,
    PreparedTxEnvelope,
    TxExecuteResult,
    TxReceiptSummary,
)
from aurey.runtime import AureyRuntime


class TxExecuteInput(BaseModel):
    envelope: dict[str, Any]
    idempotency_key: str | None = None


class TxExecuteGraphState(TypedDict, total=False):
    input: dict[str, Any]
    error: dict[str, Any]
    result: dict[str, Any]


def _validation_error(exc: ValidationError) -> dict[str, Any]:
    return GraphErrorBody(
        code="invalid_input",
        message="Transaction execute input failed validation.",
        details={"errors": exc.errors()},
    ).model_dump()


def _validate_node(state: TxExecuteGraphState) -> TxExecuteGraphState:
    try:
        TxExecuteInput.model_validate(state.get("input") or {})
        PreparedTxEnvelope.model_validate((state.get("input") or {}).get("envelope") or {})
    except ValidationError as exc:
        return {"error": _validation_error(exc)}
    return {}


def _execute_node(runtime: AureyRuntime, state: TxExecuteGraphState) -> TxExecuteGraphState:
    if state.get("error"):
        return {}

    root = TxExecuteInput.model_validate(state["input"])
    envelope = PreparedTxEnvelope.model_validate(root.envelope)
    key_path = envelope.signing_key_secret_path

    try:
        signing_material = runtime.secret_store.get_secret(key_path).reveal()
    except SecretNotFoundError:
        return {
            "error": GraphErrorBody(
                code="secret_not_found",
                message="Signing secret could not be resolved.",
                details={"secret_kind": "signing_key"},
            ).model_dump()
        }
    except SecretStoreUnavailableError:
        return {
            "error": GraphErrorBody(
                code="secret_unavailable",
                message="Secret store unavailable while resolving signing material.",
                details={"secret_kind": "signing_key"},
            ).model_dump()
        }

    try:
        outcome = runtime.tx_pipeline.run_prepared(
            envelope,
            signing_key_material_hex=signing_material,
        )
    except RuntimeError as exc:
        message = str(exc)
        code: Literal["simulation_failed", "policy_rejected", "broadcast_failed"]
        if message.startswith("simulation_failed"):
            code = "simulation_failed"
        elif message.startswith("policy_rejected"):
            code = "policy_rejected"
        elif message.startswith("broadcast_failed"):
            code = "broadcast_failed"
        else:
            code = "simulation_failed"
        return {
            "error": GraphErrorBody(
                code=code,
                message=message,
                details=None,
            ).model_dump()
        }

    return {"result": outcome.model_dump()}


def _route_after_validate(state: TxExecuteGraphState) -> Literal["execute", "done"]:
    return "done" if state.get("error") else "execute"


def build_tx_execute_graph(runtime: AureyRuntime):
    g: StateGraph = StateGraph(TxExecuteGraphState)

    def validate(state: TxExecuteGraphState) -> TxExecuteGraphState:
        return _validate_node(state)

    def execute(state: TxExecuteGraphState) -> TxExecuteGraphState:
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


class DeterministicTxPipeline(TxPipelinePort):
    """Test-friendly pipeline with explicit stage hooks; never echoes signing material."""

    def __init__(
        self,
        *,
        fail_stage: Literal["simulate", "policy", "broadcast"] | None = None,
    ) -> None:
        self._fail_stage = fail_stage

    def run_prepared(
        self,
        envelope: PreparedTxEnvelope,
        *,
        signing_key_material_hex: str,
    ) -> TxExecuteResult:
        _ = signing_key_material_hex  # would feed a real signer in production
        if self._fail_stage == "simulate":
            raise RuntimeError("simulation_failed: deterministic test failure")

        if self._fail_stage == "policy":
            raise RuntimeError("policy_rejected: deterministic test failure")

        if self._fail_stage == "broadcast":
            raise RuntimeError("broadcast_failed: deterministic test failure")

        payload = "|".join(
            [
                str(envelope.chain_id),
                envelope.kind,
                envelope.from_address,
                envelope.to,
                envelope.data,
                envelope.value_hex,
            ]
        )
        tx_hash = "0x" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
        receipt = TxReceiptSummary(status=1, block_number=12_345, gas_used=21_000)
        stages = {
            "simulate": "ok",
            "policy": "ok",
            "sign": "ok",
            "broadcast": "ok",
        }
        return TxExecuteResult(tx_hash=tx_hash, receipt=receipt, stages=stages)
