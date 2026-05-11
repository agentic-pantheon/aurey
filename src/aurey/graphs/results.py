"""Structured graph outputs and stable error codes (no secret values)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

GraphErrorCode = Literal[
    "needs_approval",
    "unsupported_chain",
    "simulation_failed",
    "secret_not_configured",
    "secret_unavailable",
    "secret_not_found",
    "invalid_input",
    "rpc_error",
    "http_error",
    "swap_prepare_failed",
    "policy_rejected",
    "broadcast_failed",
    "not_implemented",
]


class GraphErrorBody(BaseModel):
    """Machine-readable failure for deep-agent continuations."""

    model_config = ConfigDict(frozen=True)

    code: GraphErrorCode
    message: str
    details: dict[str, Any] | None = None


class NativeBalanceResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    chain: str
    chain_id: int
    wallet_address: str
    balance_wei_hex: str


class KnownAddressResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    chain: str
    ticker: str
    resolved_address: str


class Erc20ReadPlaceholder(BaseModel):
    model_config = ConfigDict(frozen=True)

    chain: str
    operation: Literal["erc20_balance", "erc20_allowance", "erc20_metadata", "contract_read"]
    status: Literal["placeholder"] = "placeholder"
    message: str = "On-chain ERC-20 read path not wired in this build."


class AlchemyTokenPricesResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    chain: str
    prices_by_address: dict[str, str]


class AlchemyPortfolioResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    chain: str
    wallet_address: str
    tokens: list[dict[str, Any]]


class AlchemyTransferHistoryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    chain: str
    wallet_address: str
    transfers: list[dict[str, Any]]


class LiFiPreparedTx(BaseModel):
    model_config = ConfigDict(frozen=True)

    route_id: str
    transaction_request: dict[str, Any]


class SwapPrepareResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Literal["lifi"] = "lifi"
    prepared: LiFiPreparedTx


TxKind = Literal["native_transfer", "erc20_transfer", "erc20_approval"]


class PreparedTxEnvelope(BaseModel):
    """Serializable transaction intent; signing material is referenced by vault path only."""

    model_config = ConfigDict(frozen=True)

    kind: TxKind
    chain_id: int
    from_address: str
    to: str
    data: str
    value_hex: str
    gas_limit_hex: str | None = None
    nonce: int | None = None
    signing_key_secret_path: str = Field(min_length=1)


class TxReceiptSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: int
    block_number: int
    gas_used: int


class TxExecuteResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tx_hash: str
    receipt: TxReceiptSummary
    stages: dict[str, Literal["ok"]]


class GraphRunResult(BaseModel):
    """Top-level graph response wrapper."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    result: dict[str, Any] | None = None
    error: GraphErrorBody | None = None
