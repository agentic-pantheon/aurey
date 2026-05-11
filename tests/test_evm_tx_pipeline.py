"""Unit tests for :class:`~aurey.graphs.evm_tx_pipeline.Web3TxPipeline` (Web3 mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from eth_account import Account
from hexbytes import HexBytes

from aurey.custody import FakeSecretStore
from aurey.graphs.evm_tx_pipeline import Web3TxPipeline
from aurey.graphs.results import PreparedTxEnvelope
from aurey.settings import AureySettings


def _envelope(*, signer: Account, chain_id: int = 8453) -> PreparedTxEnvelope:
    return PreparedTxEnvelope(
        kind="native_transfer",
        chain_id=chain_id,
        from_address=signer.address,
        to="0x1111111111111111111111111111111111111111",
        data="0x",
        value_hex="0x0",
        gas_limit_hex=None,
        nonce=None,
        signing_key_secret_path="vault/signing",
    )


def test_chain_name_for_id():
    from aurey.graphs.chains import chain_name_for_id

    assert chain_name_for_id(8453) == "base"
    assert chain_name_for_id(1) == "ethereum"
    assert chain_name_for_id(999_999) is None


def test_run_prepared_address_mismatch():
    signer = Account.create()
    other = Account.create()
    pipeline = Web3TxPipeline(
        settings=AureySettings(alchemy_api_secret_path="alchemy/k"),
        secret_store=FakeSecretStore({"alchemy/k": "test-alchemy-key"}),
        web3_factory=lambda _u: MagicMock(),
    )
    env = _envelope(signer=other)
    with pytest.raises(RuntimeError, match="policy_rejected"):
        pipeline.run_prepared(env, signing_key_material_hex=signer.key.hex())


def test_run_prepared_success_with_mock_w3():
    signer = Account.create()
    mock_w3 = MagicMock()
    mock_w3.eth.chain_id = 8453
    mock_w3.eth.get_transaction_count.return_value = 0
    mock_w3.eth.estimate_gas.return_value = 21_000
    mock_w3.eth.max_priority_fee = 1_000_000_000
    mock_w3.eth.get_block.return_value = {"baseFeePerGas": 2_000_000_000}
    mock_w3.eth.get_balance.return_value = 10**20
    mock_w3.eth.call.return_value = b""
    mock_w3.eth.send_raw_transaction.return_value = HexBytes(b"\xab" * 32)
    mock_w3.eth.wait_for_transaction_receipt.return_value = {
        "status": 1,
        "blockNumber": 1_000,
        "gasUsed": 21_000,
    }

    pipeline = Web3TxPipeline(
        settings=AureySettings(alchemy_api_secret_path="alchemy/k"),
        secret_store=FakeSecretStore({"alchemy/k": "test-alchemy-key"}),
        web3_factory=lambda _url: mock_w3,
        receipt_timeout_s=5.0,
    )
    env = _envelope(signer=signer)
    key_hex = "0x" + signer.key.hex()

    out = pipeline.run_prepared(env, signing_key_material_hex=key_hex)

    assert out.tx_hash.startswith("0x")
    assert len(out.tx_hash) == 66
    assert out.receipt.status == 1
    assert out.receipt.block_number == 1_000
    assert out.receipt.gas_used == 21_000
    mock_w3.eth.send_raw_transaction.assert_called_once()
    mock_w3.eth.wait_for_transaction_receipt.assert_called_once()


def test_simulation_failed_hint_for_erc20_balance_revert():
    from aurey.graphs.evm_tx_pipeline import _simulation_failed

    env = PreparedTxEnvelope(
        kind="erc20_transfer",
        chain_id=8453,
        from_address="0x" + "11" * 20,
        to="0x" + "22" * 20,
        data="0xa9059cbb",
        value_hex="0x0",
        gas_limit_hex=None,
        nonce=None,
        signing_key_secret_path="vault/signing",
    )
    err = _simulation_failed(
        env,
        Exception("execution reverted: ERC20: transfer amount exceeds balance"),
        step="gas estimation failed",
    )
    assert "6 decimals" in str(err).lower() or "10_000" in str(err)

    signer = Account.create()
    pipeline = Web3TxPipeline(
        settings=AureySettings(alchemy_api_secret_path=None),
        secret_store=FakeSecretStore({}),
        web3_factory=lambda _u: MagicMock(),
    )
    with pytest.raises(RuntimeError, match="policy_rejected"):
        pipeline.run_prepared(_envelope(signer=signer), signing_key_material_hex=signer.key.hex())
