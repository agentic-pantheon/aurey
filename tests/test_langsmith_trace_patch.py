"""LangSmith + LangChain tool tracing patch."""

from __future__ import annotations

import pytest

pytest.importorskip("langsmith")
pytest.importorskip("langchain_core.tools")


def test_apply_langsmith_tool_output_patch_is_idempotent() -> None:
    from langsmith import run_helpers

    from aurey.reasoning.langsmith_trace import apply_langsmith_tool_output_patch

    apply_langsmith_tool_output_patch()
    first = run_helpers._container_end
    apply_langsmith_tool_output_patch()
    assert run_helpers._container_end is first
