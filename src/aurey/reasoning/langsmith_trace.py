"""LangSmith tracing compatibility for LangChain tools."""

from __future__ import annotations

from typing import Any

_applied: bool = False


def apply_langsmith_tool_output_patch() -> None:
    """Normalize LangChain tool instances before LangSmith ends a traced run.

    LangSmith's ``_container_end`` uses ``model_dump(mode="json")`` whenever ``outputs`` has a
    ``model_dump`` method. :class:`~langchain_core.tools.structured.StructuredTool` satisfies that
    but carries ``args_schema`` as a Pydantic model *class*, which is not JSON-serializable and
    raises :class:`pydantic.errors.PydanticSerializationError` (logged at DEBUG).
    """

    global _applied
    if _applied:
        return

    try:
        import langsmith.run_helpers as run_helpers
        from langchain_core.tools import BaseTool
    except ImportError:
        _applied = True
        return

    _orig = run_helpers._container_end

    def _container_end(
        container: Any,
        outputs: Any = None,
        error: Any = None,
    ) -> None:
        if isinstance(outputs, BaseTool):
            outputs = {
                "trace_tool": True,
                "tool_class": type(outputs).__name__,
                "name": outputs.name,
            }
        return _orig(container, outputs=outputs, error=error)

    run_helpers._container_end = _container_end  # type: ignore[assignment]
    _applied = True


__all__ = ["apply_langsmith_tool_output_patch"]
