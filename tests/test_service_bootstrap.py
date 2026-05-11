"""Bootstrap wiring for optional HTTP service."""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aurey.custody import FakeSecretStore
from aurey.graphs import DeterministicTxPipeline
from aurey.reasoning import create_aurey_deep_agent, make_memory_checkpointer, thread_config
from aurey.runtime import AureyRuntime
from aurey.service.adapters import UrllibHttpJsonClient, make_evm_rpc_factory
from aurey.service.bootstrap import AureyServiceBootstrapError, bootstrap_aurey_service_state
from aurey.service.state import AureyServiceState
from aurey.settings import AureySettings
from tests.fakes.evm_rpc import rpc_factory_from_mapping
from tests.fakes.http_client import ScriptedHttpClient


class _DummyChat(BaseChatModel):
    model_name: str = "stub"

    @property
    def _llm_type(self) -> str:
        return "dummy"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    def bind_tools(self, tools, **kwargs):
        return self


def test_bootstrap_raises_without_vault_id(monkeypatch):
    monkeypatch.setenv("AUREY_ONECLAW_BOOTSTRAP_API_KEY", "k")
    s = AureySettings(oneclaw_vault_id="")
    with pytest.raises(AureyServiceBootstrapError, match="vault id"):
        bootstrap_aurey_service_state(s)


def test_bootstrap_raises_on_missing_bootstrap_env(monkeypatch):
    monkeypatch.delenv("AUREY_ONECLAW_BOOTSTRAP_API_KEY", raising=False)
    s = AureySettings(oneclaw_vault_id="v1")
    with pytest.raises(AureyServiceBootstrapError, match="Bootstrap 1Claw API key"):
        bootstrap_aurey_service_state(s)


def test_construct_service_state_get_graph_invoke_smoke(monkeypatch):
    """Fake runtime + patched deep agent avoids live model providers."""

    monkeypatch.setattr(
        "aurey.service.state.create_aurey_deep_agent",
        lambda runtime, *, model, checkpointer=None, **kw: create_aurey_deep_agent(
            runtime,
            model=_DummyChat(),
            checkpointer=checkpointer,
            **kw,
        ),
    )

    alchemy_path = "vault/alchemy"
    signing_path = "vault/signing/local"
    settings = AureySettings(
        alchemy_api_secret_path=alchemy_path,
        wallet_signing_key_secret_path=signing_path,
        deep_agent_default_model="stub-spec",
        oneclaw_vault_id="ignored-for-fake-runtime",
    )
    runtime = AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(
            {
                alchemy_path: "SECRET_FRAGMENT",
                signing_path: "0x" + "ff" * 32,
            }
        ),
        evm_rpc_factory=rpc_factory_from_mapping({}),
        http=ScriptedHttpClient(),
        tx_pipeline=DeterministicTxPipeline(),
    )
    state = AureyServiceState(
        settings=settings,
        runtime=runtime,
        checkpointer=make_memory_checkpointer(),
        default_model="stub-spec",
    )
    graph = state.get_or_create_graph("stub-spec")
    out = graph.invoke(
        {"messages": [HumanMessage(content="hi")]},
        config=thread_config("bootstrap-smoke-thread"),
    )
    assert out["messages"][-1].content == "ok"


def test_adapters_construct_runtime_dependencies():
    alchemy_path = "alchemy/path"
    s = AureySettings(alchemy_api_secret_path=alchemy_path)
    store = FakeSecretStore({alchemy_path: "alchemy-key"})
    rt = AureyRuntime(
        settings=s,
        secret_store=store,
        evm_rpc_factory=make_evm_rpc_factory(timeout_s=1.0),
        http=UrllibHttpJsonClient(timeout_s=1.0),
        tx_pipeline=DeterministicTxPipeline(),
    )
    assert rt.http is not None
    url = "https://eth-mainnet.g.alchemy.com/v2/" + rt.secret_store.get_secret(
        alchemy_path
    ).reveal()
    port = rt.evm_rpc_factory(url)
    assert port is not None
