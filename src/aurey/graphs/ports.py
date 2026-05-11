"""Injectable boundaries for RPC, HTTP, and transaction execution."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from aurey.graphs.results import PreparedTxEnvelope, TxExecuteResult


@runtime_checkable
class EvmJsonRpcPort(Protocol):
    """Minimal JSON-RPC surface used by read/prepare flows."""

    def call(self, method: str, params: list[Any]) -> Any:
        """Perform a JSON-RPC call and return the decoded ``result`` payload."""


@runtime_checkable
class HttpJsonPort(Protocol):
    """HTTP client used by Alchemy/LiFi adapters (callers build URLs)."""

    def request_json(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | list[Any] | None = None,
    ) -> dict[str, Any]:
        """Perform an HTTP request and return a JSON object body."""


@runtime_checkable
class TxPipelinePort(Protocol):
    """Simulate, sign, and broadcast a prepared EVM transaction."""

    def run_prepared(
        self,
        envelope: PreparedTxEnvelope,
        *,
        signing_key_material_hex: str,
    ) -> TxExecuteResult:
        """Consume signing material in-process only; never embed it in the returned model."""
