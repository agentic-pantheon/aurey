"""Environment parsing and callback safety for ``aurey.service.agent_trace``."""

from __future__ import annotations

import logging
from uuid import uuid4

import pytest

from aurey.service.agent_trace import (
    AureyAgentTraceHandler,
    agent_trace_detail,
    build_agent_trace_handler,
)


@pytest.mark.parametrize(
    ("env_val", "expected"),
    [
        ("", None),
        ("0", None),
        ("off", None),
        ("false", None),
        ("1", "info"),
        ("true", "info"),
        ("INFO", "info"),
        ("debug", "debug"),
        ("verbose", "debug"),
    ],
)
def test_agent_trace_detail_env(monkeypatch, env_val: str, expected: str | None) -> None:
    monkeypatch.delenv("AUREY_AGENT_TRACE", raising=False)
    if env_val != "":
        monkeypatch.setenv("AUREY_AGENT_TRACE", env_val)
    assert agent_trace_detail() == expected


def test_build_handler_none_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("AUREY_AGENT_TRACE", raising=False)
    assert build_agent_trace_handler(session_id="s1") is None


def test_build_handler_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AUREY_AGENT_TRACE", "1")
    h = build_agent_trace_handler(session_id="sess-x")
    assert isinstance(h, AureyAgentTraceHandler)


def test_chain_start_tolerates_none_serialized_info_mode(caplog) -> None:
    h = AureyAgentTraceHandler(session_id="s", detail="info")
    caplog.set_level(logging.INFO, logger="aurey.agent.trace")
    rid = uuid4()
    h.on_chain_start(
        None,
        {},
        run_id=rid,
        metadata={"langgraph_node": "model", "langgraph_step": 2},
    )
    assert "graph_node=model" in caplog.text
    assert "event=chain_start" in caplog.text


def test_chain_start_skips_middleware_in_info_mode(caplog) -> None:
    h = AureyAgentTraceHandler(session_id="s", detail="info")
    caplog.set_level(logging.INFO, logger="aurey.agent.trace")
    rid = uuid4()
    h.on_chain_start(
        None,
        {},
        run_id=rid,
        metadata={"langgraph_node": "PatchToolCallsMiddleware.before_agent", "langgraph_step": 1},
    )
    assert caplog.text == ""
