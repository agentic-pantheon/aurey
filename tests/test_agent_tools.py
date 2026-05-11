"""LangChain tool schemas, subgraph invokes, and deep-agent factory wiring."""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aurey.custody import FakeSecretStore
from aurey.graphs import DeterministicTxPipeline, TxExecuteInput
from aurey.reasoning import create_aurey_deep_agent, make_memory_checkpointer, thread_config
from aurey.reasoning import deep_agent as deep_agent_mod
from aurey.runtime import AureyRuntime
from aurey.settings import AureySettings
from aurey.tools import build_aurey_subgraph_tools, reset_user_input_context
from aurey.tools.user_input import UserQuestion, get_pending_user_questions
from tests.fakes.evm_rpc import rpc_factory_from_mapping
from tests.fakes.http_client import ScriptedHttpClient


def _ban_values() -> tuple[str, ...]:
    return (
        "INJECTED_RPC_URL_SECRET_FRAGMENT",
        "INJECTED_ALCHEMY_KEY_AAA",
        "INJECTED_LIFI_KEY_BBB",
        "0x" + "ff" * 32,
    )


def _assert_no_banned_values(payload: object) -> None:
    blob = json.dumps(payload, default=str, sort_keys=True)
    for fragment in _ban_values():
        assert fragment not in blob


class _DummyChat(BaseChatModel):
    model_name: str = "stub"

    @property
    def _llm_type(self) -> str:
        return "dummy"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    def bind_tools(self, tools, **kwargs):
        return self


def _tool_by_name(tools, name: str):
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(f"missing tool {name}")


@pytest.fixture(autouse=True)
def _clear_user_input_ctx():
    reset_user_input_context()
    yield
    reset_user_input_context()


def test_tool_schemas_include_expected_names_and_descriptions():
    alchemy_path = "vault/alchemy"
    signing_path = "vault/signing/local"
    secrets = {
        alchemy_path: "INJECTED_ALCHEMY_KEY_AAA",
        signing_path: "0x" + "ff" * 32,
    }
    settings = AureySettings(
        alchemy_api_secret_path=alchemy_path,
        wallet_signing_key_secret_path=signing_path,
    )
    runtime = AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping({}),
        http=ScriptedHttpClient(),
        tx_pipeline=DeterministicTxPipeline(),
        lifi_base_url="https://li.quest",
    )
    tools = build_aurey_subgraph_tools(runtime)
    names = {t.name for t in tools}
    expected = {
        "evm_get_native_balance",
        "evm_get_erc20_decimals",
        "resolve_known_address",
        "evm_get_erc20_balance",
        "alchemy_get_token_prices",
        "alchemy_get_portfolio_tokens",
        "alchemy_get_transfer_history",
        "swap_prepare",
        "tx_prepare_native_transfer",
        "tx_prepare_erc20_transfer",
        "tx_prepare_erc20_approval",
        "tx_execute",
        "request_user_input",
    }
    assert expected <= names
    nb = _tool_by_name(tools, "evm_get_native_balance")
    assert "balance" in (nb.description or "").lower()
    rk = _tool_by_name(tools, "resolve_known_address")
    assert "ticker" in (rk.description or "").lower() or "address" in (rk.description or "").lower()


def test_evm_get_native_balance_tool_fake_runtime():
    alchemy_path = "vault/alchemy"
    signing_path = "vault/signing/local"
    secrets = {
        alchemy_path: "INJECTED_ALCHEMY_KEY_AAA",
        signing_path: "0x" + "ff" * 32,
    }
    settings = AureySettings(
        alchemy_api_secret_path=alchemy_path,
        wallet_signing_key_secret_path=signing_path,
    )
    runtime = AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping({"eth_getBalance": "0x10"}),
        http=ScriptedHttpClient(),
        tx_pipeline=DeterministicTxPipeline(),
        lifi_base_url="https://li.quest",
    )
    tool = _tool_by_name(build_aurey_subgraph_tools(runtime), "evm_get_native_balance")
    out = tool.invoke(
        {
            "chain": "ethereum",
            "wallet_address": "0x0000000000000000000000000000000000000001",
        }
    )
    assert out["ok"] is True
    assert out["result"]["balance_wei_hex"] == "0x10"
    _assert_no_banned_values(out)


def test_resolve_known_address_tool_fake_runtime():
    signing_path = "vault/signing/local"
    secrets = {signing_path: "0x" + "ff" * 32}
    settings = AureySettings(
        wallet_signing_key_secret_path=signing_path,
    )
    runtime = AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping({}),
        http=ScriptedHttpClient(),
        tx_pipeline=DeterministicTxPipeline(),
        lifi_base_url="https://li.quest",
    )
    tool = _tool_by_name(build_aurey_subgraph_tools(runtime), "resolve_known_address")
    out = tool.invoke({"chain": "ethereum", "known_ticker": "usdc"})
    assert out["ok"] is True
    assert out["result"]["resolved_address"] == "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
    assert out["result"]["symbol"] == "USDC"
    assert out["result"]["name"] == "USD Coin"
    _assert_no_banned_values(out)


def test_evm_get_erc20_balance_tool_stub():
    signing_path = "vault/signing/local"
    secrets = {signing_path: "0x" + "ff" * 32}
    settings = AureySettings(
        wallet_signing_key_secret_path=signing_path,
    )
    runtime = AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping({}),
        http=ScriptedHttpClient(),
        tx_pipeline=DeterministicTxPipeline(),
        lifi_base_url="https://li.quest",
    )
    tool = _tool_by_name(build_aurey_subgraph_tools(runtime), "evm_get_erc20_balance")
    out = tool.invoke(
        {
            "chain": "ethereum",
            "wallet_address": "0x00000000000000000000000000000000000000aa",
            "token_address": "0x2222222222222222222222222222222222222222",
        }
    )
    assert out["ok"] is True
    assert out["result"]["operation"] == "erc20_balance"
    assert out["result"]["token_address"] == "0x2222222222222222222222222222222222222222"
    _assert_no_banned_values(out)


def test_evm_get_erc20_decimals_tool():
    alchemy_path = "vault/alchemy"
    secrets = {alchemy_path: "INJECTED_ALCHEMY_KEY_AAA"}
    usdc_base = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

    def eth_call(params: list) -> str:
        assert params[0]["data"] == "0x313ce567"
        return "0x0000000000000000000000000000000000000000000000000000000000000006"

    runtime = AureyRuntime(
        settings=AureySettings(
            alchemy_api_secret_path=alchemy_path,
            wallet_signing_key_secret_path="vault/signing",
        ),
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping({"eth_call": eth_call}),
        http=ScriptedHttpClient(),
        tx_pipeline=DeterministicTxPipeline(),
        lifi_base_url="https://li.quest",
    )
    tool = _tool_by_name(build_aurey_subgraph_tools(runtime), "evm_get_erc20_decimals")
    out = tool.invoke({"chain": "base", "token_address": usdc_base})
    assert out["ok"] is True
    assert out["result"]["decimals"] == 6
    assert out["result"]["token_address"] == usdc_base.lower()
    _assert_no_banned_values(out)


def test_alchemy_get_token_prices_tool_fake_runtime():
    secrets = {"vault/alchemy": "INJECTED_ALCHEMY_KEY_AAA"}
    settings = AureySettings(alchemy_api_secret_path="vault/alchemy")

    def match_prices(**kw: object) -> bool:
        if kw.get("method") != "POST":
            return False
        url = str(kw.get("url") or "")
        if "/prices/v1/" not in url or "tokens/by-address" not in url:
            return False
        body = kw.get("json_body") or {}
        return isinstance(body, dict) and isinstance(body.get("addresses"), list)

    http = ScriptedHttpClient(
        [
            (
                match_prices,
                {
                    "data": [
                        {
                            "network": "eth-mainnet",
                            "address": "0x2222222222222222222222222222222222222222",
                            "prices": [
                                {
                                    "currency": "USD",
                                    "value": "3.14",
                                    "lastUpdatedAt": "2025-01-01T00:00:00Z",
                                }
                            ],
                            "error": None,
                        }
                    ]
                },
            )
        ]
    )
    runtime = AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping({}),
        http=http,
        tx_pipeline=DeterministicTxPipeline(),
        lifi_base_url="https://li.quest",
    )
    tool = _tool_by_name(build_aurey_subgraph_tools(runtime), "alchemy_get_token_prices")
    out = tool.invoke(
        {
            "chain": "ethereum",
            "wallet_address": "0x1111111111111111111111111111111111111111",
            "token_addresses": ["0x2222222222222222222222222222222222222222"],
        }
    )
    addr = "0x2222222222222222222222222222222222222222"
    assert out["ok"] is True
    assert out["result"]["prices_by_address"][addr] == "3.14"
    _assert_no_banned_values(out)


def test_request_user_input_shape_and_context():
    signing_path = "vault/signing/local"
    secrets = {signing_path: "0x" + "ff" * 32}
    settings = AureySettings(
        wallet_signing_key_secret_path=signing_path,
    )
    runtime = AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping({}),
        http=ScriptedHttpClient(),
        tx_pipeline=DeterministicTxPipeline(),
        lifi_base_url="https://li.quest",
    )
    tool = _tool_by_name(build_aurey_subgraph_tools(runtime), "request_user_input")
    q = UserQuestion(prompt="Which chain?", id="c1")
    out = tool.invoke({"questions": [q]})
    assert out == {"ok": True, "result": {"status": "needs_user_input", "question_count": 1}}
    pending = get_pending_user_questions()
    assert pending == [{"prompt": "Which chain?", "id": "c1"}]


def test_tx_prepare_named_tool_ignores_legacy_kind_field():
    signing_path = "vault/signing/local"
    secrets = {signing_path: "0x" + "ff" * 32}
    settings = AureySettings(wallet_signing_key_secret_path=signing_path)
    runtime = AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping({}),
        http=ScriptedHttpClient(),
        tx_pipeline=DeterministicTxPipeline(),
        lifi_base_url="https://li.quest",
    )
    tool = _tool_by_name(build_aurey_subgraph_tools(runtime), "tx_prepare_erc20_transfer")
    out = tool.invoke(
        {
            "kind": "erc20_transfer",
            "chain": "base",
            "from_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "token_address": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "to_address": "0xcccccccccccccccccccccccccccccccccccccccc",
            "amount_wei": 10_000,
        }
    )
    assert out["ok"] is True
    assert out["result"]["envelope"]["kind"] == "erc20_transfer"
    _assert_no_banned_values(out)


def test_tx_execute_tool_accepts_tx_execute_shape():
    signing_path = "vault/signing/local"
    secrets = {signing_path: "0x" + "ff" * 32}
    settings = AureySettings(wallet_signing_key_secret_path=signing_path)
    runtime = AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping({}),
        http=ScriptedHttpClient(),
        tx_pipeline=DeterministicTxPipeline(),
        lifi_base_url="https://li.quest",
    )
    env = {
        "kind": "native_transfer",
        "chain_id": 1,
        "from_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "to": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "data": "0x",
        "value_hex": "0x1",
        "gas_limit_hex": None,
        "nonce": None,
        "signing_key_secret_path": signing_path,
    }
    TxExecuteInput.model_validate({"envelope": env})
    tool = _tool_by_name(build_aurey_subgraph_tools(runtime), "tx_execute")
    out = tool.invoke({"envelope": env})
    assert out["ok"] is True
    assert out["result"]["tx_hash"].startswith("0x")
    _assert_no_banned_values(out)


def test_create_aurey_deep_agent_compiles():
    signing_path = "vault/signing/local"
    secrets = {signing_path: "0x" + "ff" * 32}
    settings = AureySettings(wallet_signing_key_secret_path=signing_path)
    runtime = AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping({}),
        http=ScriptedHttpClient(),
        tx_pipeline=DeterministicTxPipeline(),
        lifi_base_url="https://li.quest",
    )
    graph = create_aurey_deep_agent(
        runtime,
        model=_DummyChat(),
        checkpointer=make_memory_checkpointer(),
    )
    assert graph is not None
    cfg = thread_config("session-unit-test")
    assert "configurable" in cfg and cfg["configurable"]["thread_id"] == "session-unit-test"


def test_create_aurey_deep_agent_import_error_message(monkeypatch):
    monkeypatch.setattr(deep_agent_mod, "_create_deep_agent_impl", None)
    signing_path = "vault/signing/local"
    secrets = {signing_path: "0x" + "ff" * 32}
    settings = AureySettings(wallet_signing_key_secret_path=signing_path)
    runtime = AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping({}),
        http=ScriptedHttpClient(),
        tx_pipeline=DeterministicTxPipeline(),
        lifi_base_url="https://li.quest",
    )
    with pytest.raises(RuntimeError, match="deepagents"):
        create_aurey_deep_agent(runtime, model=_DummyChat())
