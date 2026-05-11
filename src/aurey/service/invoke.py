"""Shared Deep Agent invocation path for HTTP and Telegram surfaces."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from aurey.reasoning import thread_config
from aurey.service.state import AureyServiceState


class AgentInvokeError(BaseModel):
    code: str
    message: str


class AgentInvokeResult(BaseModel):
    ok: bool
    session_id: str | None = None
    messages: list[dict[str, Any]] | None = None
    error: AgentInvokeError | None = None


def summarize_agent_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Return JSON-safe message summaries without provider-specific metadata."""

    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, BaseMessage):
            content = getattr(m, "content", None)
            body: Any = "<non-text>" if isinstance(content, list) else content
            out.append(
                {
                    "role": getattr(m, "type", m.__class__.__name__),
                    "type": m.__class__.__name__,
                    "content": body,
                }
            )
    return out


def invoke_deep_agent_turn(
    svc: AureyServiceState | None,
    *,
    message: str,
    session_id: str,
    context: dict[str, Any] | None = None,
    model: str | None = None,
) -> AgentInvokeResult:
    """Invoke the shared deep-agent graph with sanitized error responses."""

    if svc is None:
        return AgentInvokeResult(
            ok=False,
            session_id=session_id,
            error=AgentInvokeError(
                code="service_misconfigured",
                message="The service is missing required configuration or bootstrap credentials.",
            ),
        )

    extra: dict[str, Any] = {}
    if context is not None:
        extra["aurey_context"] = context
    config = thread_config(session_id, **extra)

    try:
        graph = svc.get_or_create_graph(model)
    except RuntimeError as exc:
        code = "deep_agent_unavailable"
        if "deepagents" in str(exc).lower():
            code = "deep_agent_dependency"
        return AgentInvokeResult(
            ok=False,
            session_id=session_id,
            error=AgentInvokeError(
                code=code,
                message="The deep agent runtime is not available or misconfigured.",
            ),
        )

    try:
        result = graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)
    except Exception:
        return AgentInvokeResult(
            ok=False,
            session_id=session_id,
            error=AgentInvokeError(
                code="agent_invoke_failed",
                message="The agent failed to complete this turn.",
            ),
        )

    raw_messages = result.get("messages") if isinstance(result, dict) else None
    if not isinstance(raw_messages, list):
        raw_messages = []

    return AgentInvokeResult(
        ok=True,
        session_id=session_id,
        messages=summarize_agent_messages(raw_messages),
    )


__all__ = [
    "AgentInvokeError",
    "AgentInvokeResult",
    "invoke_deep_agent_turn",
    "summarize_agent_messages",
]
