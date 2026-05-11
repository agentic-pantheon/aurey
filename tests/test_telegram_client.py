"""Telegram client handling without requiring the optional runtime dependency."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from aurey.custody import FakeSecretStore
from aurey.graphs import DeterministicTxPipeline
from aurey.reasoning import make_memory_checkpointer
from aurey.runtime import AureyRuntime
from aurey.service.state import AureyServiceState
from aurey.settings import AureySettings
from aurey.telegram import (
    TelegramConfigurationError,
    handle_telegram_text,
    resolve_telegram_bot_token,
)
from tests.fakes.evm_rpc import rpc_factory_from_mapping
from tests.fakes.http_client import ScriptedHttpClient
from tests.leakage_helpers import (
    FAKE_ERROR_BODY_SECRET,
    FAKE_TELEGRAM_BOT_TOKEN,
    assert_no_sensitive_leakage,
)


class _RecordingGraph:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.payload: dict[str, Any] | None = None
        self.config: dict[str, Any] | None = None

    def invoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None):
        self.payload = payload
        self.config = config
        if self.fail:
            raise RuntimeError(FAKE_ERROR_BODY_SECRET)
        return {"messages": [AIMessage(content="telegram ok")]}


class _FakeServiceState:
    def __init__(self, graph: _RecordingGraph) -> None:
        self.graph = graph
        self.model: str | None = None

    def get_or_create_graph(self, model: str | None):
        self.model = model
        return self.graph


def _service_state_with_token(*, token_path: str | None, token: str | None) -> AureyServiceState:
    secrets = {}
    if token_path is not None and token is not None:
        secrets[token_path] = token
    settings = AureySettings(telegram_bot_token_secret_path=token_path)
    runtime = AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping({}),
        http=ScriptedHttpClient(),
        tx_pipeline=DeterministicTxPipeline(),
    )
    return AureyServiceState(
        settings=settings,
        runtime=runtime,
        checkpointer=make_memory_checkpointer(),
        default_model="stub-model",
    )


def test_handle_telegram_text_reuses_shared_agent_invocation() -> None:
    graph = _RecordingGraph()
    state = _FakeServiceState(graph)

    reply = handle_telegram_text(
        state,  # type: ignore[arg-type]
        chat_id=123,
        user_id=456,
        text="hello aurey",
        model="stub-model",
    )

    assert reply == "telegram ok"
    assert state.model == "stub-model"
    assert graph.payload is not None
    assert len(graph.payload["messages"]) == 1
    assert graph.payload["messages"][0].content == "hello aurey"
    assert graph.config == {
        "configurable": {
            "thread_id": "telegram:123",
            "aurey_context": {"telegram_chat_id": "123", "telegram_user_id": "456"},
        }
    }


def test_handle_telegram_text_sanitizes_agent_errors() -> None:
    state = _FakeServiceState(_RecordingGraph(fail=True))
    reply = handle_telegram_text(
        state,  # type: ignore[arg-type]
        chat_id="chat-secret",
        text="boom",
    )

    assert "agent_invoke_failed" in reply
    assert FAKE_ERROR_BODY_SECRET not in reply
    assert_no_sensitive_leakage({"reply": reply})


def test_resolve_telegram_bot_token_uses_secret_store() -> None:
    state = _service_state_with_token(
        token_path="aurey/telegram/bot_token",
        token=FAKE_TELEGRAM_BOT_TOKEN,
    )

    assert resolve_telegram_bot_token(state) == FAKE_TELEGRAM_BOT_TOKEN


def test_resolve_telegram_bot_token_missing_path_is_sanitized() -> None:
    state = _service_state_with_token(token_path=None, token=FAKE_TELEGRAM_BOT_TOKEN)

    with pytest.raises(TelegramConfigurationError) as exc:
        resolve_telegram_bot_token(state)

    blob = json.dumps({"error": str(exc.value)})
    assert "bot_token" not in blob
    assert_no_sensitive_leakage({"error": str(exc.value)})
