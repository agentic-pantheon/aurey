"""Web3-backed transaction pipeline (Mercury-inspired flow; no Mercury dependency)."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from eth_account import Account
from web3 import Web3
from web3.exceptions import TimeExhausted

from aurey.custody.errors import SecretNotFoundError, SecretStoreUnavailableError
from aurey.custody.secret_store import SecretStore
from aurey.graphs.chains import alchemy_rpc_url_for_chain, chain_name_for_id
from aurey.graphs.ports import TxPipelinePort
from aurey.graphs.results import PreparedTxEnvelope, TxExecuteResult, TxReceiptSummary
from aurey.settings import AureySettings

_PRIVATE_KEY_HEX = re.compile(r"^(?:0x)?[a-fA-F0-9]{64}$")


def _simulation_failed(
    envelope: PreparedTxEnvelope,
    exc: BaseException,
    *,
    step: str,
) -> RuntimeError:
    msg = f"simulation_failed: {step} ({exc})"
    if envelope.kind in ("erc20_transfer", "erc20_approval"):
        low = str(exc).lower()
        if "exceeds balance" in low:
            msg += (
                " Hint: amount must be in the token's smallest units. USDC uses 6 decimals "
                "(0.01 USDC = 10_000 raw, not 10**16). Using ether-style 1e18 scaling on USDC "
                "mints a huge transfer and reverts with 'exceeds balance'."
            )
    return RuntimeError(msg)


class Web3TxPipeline(TxPipelinePort):
    """Alchemy JSON-RPC via HTTP: estimate gas, fees, eth.call, local sign, send_rawTransaction."""

    def __init__(
        self,
        *,
        settings: AureySettings,
        secret_store: SecretStore,
        web3_factory: Callable[[str], Web3] | None = None,
        receipt_timeout_s: float = 120.0,
    ) -> None:
        self._settings = settings
        self._secret_store = secret_store
        self._receipt_timeout_s = receipt_timeout_s
        self._web3_factory = web3_factory or (
            lambda url: Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 60}))
        )

    def run_prepared(
        self,
        envelope: PreparedTxEnvelope,
        *,
        signing_key_material_hex: str,
    ) -> TxExecuteResult:
        key_hex = _normalize_signing_key_hex(signing_key_material_hex)
        try:
            signer = Account.from_key(key_hex)
        except Exception as exc:
            raise RuntimeError("policy_rejected: invalid signing key material.") from exc

        if signer.address.lower() != envelope.from_address.lower():
            raise RuntimeError("policy_rejected: signing key does not match from_address.")

        alchemy_path = (self._settings.alchemy_api_secret_path or "").strip()
        if not alchemy_path:
            raise RuntimeError(
                "policy_rejected: alchemy_api_secret_path is required for transaction broadcast."
            )

        try:
            api_key = self._secret_store.get_secret(alchemy_path).reveal().strip()
        except SecretNotFoundError as exc:
            raise RuntimeError("policy_rejected: Alchemy API key secret not found.") from exc
        except SecretStoreUnavailableError as exc:
            raise RuntimeError(
                "policy_rejected: secret store unavailable while loading Alchemy API key."
            ) from exc

        if not api_key:
            raise RuntimeError("policy_rejected: Alchemy API key is empty.")

        chain_name = chain_name_for_id(envelope.chain_id)
        if chain_name is None:
            raise RuntimeError("policy_rejected: unsupported chain_id for transaction execution.")

        rpc_url = alchemy_rpc_url_for_chain(chain_name, api_key)
        if not rpc_url:
            raise RuntimeError("policy_rejected: could not derive Alchemy RPC URL.")

        w3 = self._web3_factory(rpc_url)

        if int(w3.eth.chain_id) != envelope.chain_id:
            raise RuntimeError("simulation_failed: RPC chain id does not match envelope.")

        from_cs = Web3.to_checksum_address(envelope.from_address)
        to_cs = Web3.to_checksum_address(envelope.to)
        value_wei = int(envelope.value_hex, 0)

        nonce = envelope.nonce
        if nonce is None:
            nonce = int(w3.eth.get_transaction_count(from_cs, "pending"))

        data = envelope.data if envelope.data else "0x"

        base: dict[str, Any] = {
            "chainId": envelope.chain_id,
            "from": from_cs,
            "to": to_cs,
            "value": value_wei,
            "nonce": nonce,
            "data": data,
        }

        if envelope.gas_limit_hex is not None:
            gas_limit = int(envelope.gas_limit_hex, 0)
        else:
            try:
                gas_limit = int(w3.eth.estimate_gas(base))
            except Exception as exc:
                raise _simulation_failed(envelope, exc, step="gas estimation failed") from exc

        fee_fields = _tx_fee_fields(w3)
        tx_body: dict[str, Any] = {**base, "gas": gas_limit, **fee_fields}

        max_fee_unit = int(tx_body.get("maxFeePerGas") or tx_body.get("gasPrice") or 0)
        balance = int(w3.eth.get_balance(from_cs))
        need = value_wei + gas_limit * max_fee_unit
        if balance < need:
            raise RuntimeError(
                "simulation_failed: insufficient native balance for value and maximum gas spend."
            )

        try:
            w3.eth.call(tx_body)
        except Exception as exc:
            raise _simulation_failed(envelope, exc, step="eth_call simulation failed") from exc

        try:
            signed = Account.sign_transaction(tx_body, key_hex)
        except Exception as exc:
            raise RuntimeError(f"policy_rejected: transaction signing failed ({exc}).") from exc

        raw = signed.raw_transaction
        raw_bytes = raw if isinstance(raw, (bytes, bytearray)) else bytes(raw)

        try:
            tx_hash = w3.eth.send_raw_transaction(raw_bytes)
        except Exception as exc:
            raise RuntimeError(f"broadcast_failed: {exc}") from exc

        tx_hash_hex = Web3.to_hex(tx_hash)
        receipt = _wait_receipt(w3, tx_hash, self._receipt_timeout_s)

        return TxExecuteResult(
            tx_hash=tx_hash_hex,
            receipt=receipt,
            stages={
                "simulate": "ok",
                "policy": "ok",
                "sign": "ok",
                "broadcast": "ok",
            },
        )


def _tx_fee_fields(w3: Web3) -> dict[str, Any]:
    try:
        priority = int(w3.eth.max_priority_fee)
        latest = w3.eth.get_block("latest")
        base_fee = int(latest["baseFeePerGas"])
        max_fee = (base_fee * 2) + priority
        return {
            "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": priority,
        }
    except Exception:
        return {"gasPrice": int(w3.eth.gas_price)}


def _wait_receipt(w3: Web3, tx_hash: Any, timeout_s: float) -> TxReceiptSummary:
    try:
        rec = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout_s)
    except TimeExhausted:
        return TxReceiptSummary(status=0, block_number=0, gas_used=0)

    st = int(rec.get("status", 0))
    bn = int(rec["blockNumber"])
    gu = int(rec["gasUsed"])
    return TxReceiptSummary(status=st, block_number=bn, gas_used=gu)


def _normalize_signing_key_hex(signing_key_material_hex: str) -> str:
    raw = signing_key_material_hex.strip()
    if not _PRIVATE_KEY_HEX.match(raw):
        raise RuntimeError("policy_rejected: signing key must be a 32-byte hex string.")
    return raw if raw.startswith("0x") else f"0x{raw}"


__all__ = ["Web3TxPipeline"]
