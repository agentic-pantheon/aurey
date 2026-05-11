"""Liability tests for LiFi ``transactionRequest`` → :class:`PreparedTxEnvelope`."""

from __future__ import annotations

import pytest

from aurey.graphs.lifi_envelope import lifi_transaction_request_to_envelope


def test_lifi_mapper_accepts_numeric_chain_id_and_value():
    env = lifi_transaction_request_to_envelope(
        chain_id=8453,
        from_address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        transaction_request={
            "to": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "data": "0x",
            "value": 0,
            "chainId": 8453,
        },
        signing_key_secret_path="vault/k",
    )
    assert env.kind == "lifi_swap"
    assert env.chain_id == 8453
    assert env.value_hex == "0x0"


def test_lifi_mapper_rejects_chain_id_mismatch():
    with pytest.raises(ValueError, match="chainId"):
        lifi_transaction_request_to_envelope(
            chain_id=8453,
            from_address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            transaction_request={
                "to": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "data": "0x",
                "value": 0,
                "chainId": 1,
            },
            signing_key_secret_path="vault/k",
        )


def test_lifi_mapper_rejects_from_mismatch():
    with pytest.raises(ValueError, match="from_address"):
        lifi_transaction_request_to_envelope(
            chain_id=8453,
            from_address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            transaction_request={
                "to": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "data": "0x",
                "value": 0,
                "from": "0xcccccccccccccccccccccccccccccccccccccccc",
            },
            signing_key_secret_path="vault/k",
        )
