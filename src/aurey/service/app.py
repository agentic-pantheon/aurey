"""FastAPI application: health check and a single Deep Agent invoke endpoint."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field
from starlette.requests import Request

from aurey.reasoning import thread_config
from aurey.service.bootstrap import AureyServiceBootstrapError, bootstrap_aurey_service_state
from aurey.service.dependencies import get_aurey_service_state
from aurey.service.state import AureyServiceState
from aurey.settings import AureySettings


class InvokeBody(BaseModel):
    """Inbound chat turn."""

    model_config = ConfigDict(populate_by_name=True)

    message: str = Field(..., min_length=1, description="User message text.")
    session_id: str = Field(..., min_length=1, description="Stable session / thread identifier.")
    context: dict[str, Any] | None = Field(
        default=None,
        description="Optional values merged into configurable state under ``aurey_context``.",
    )
    agent_model_spec: str | None = Field(
        default=None,
        alias="model",
        description="Optional Deep Agents provider:model override (JSON key ``model``).",
    )


class InvokeError(BaseModel):
    code: str
    message: str


class InvokeResponse(BaseModel):
    ok: bool
    session_id: str | None = None
    messages: list[dict[str, Any]] | None = None
    error: InvokeError | None = None


def _summarize_agent_messages(messages: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, BaseMessage):
            content = getattr(m, "content", None)
            if isinstance(content, list):
                body: Any = "<non-text>"
            else:
                body = content
            out.append(
                {
                    "role": getattr(m, "type", m.__class__.__name__),
                    "type": m.__class__.__name__,
                    "content": body,
                }
            )
    return out


def _misconfigured_response(session_id: str) -> InvokeResponse:
    return InvokeResponse(
        ok=False,
        session_id=session_id,
        error=InvokeError(
            code="service_misconfigured",
            message="The service is missing required configuration or bootstrap credentials.",
        ),
    )


def create_fastapi_application(
    *,
    state: AureyServiceState | None = None,
    settings: AureySettings | None = None,
):
    """Build a FastAPI app; wiring runs in lifespan unless ``state`` is injected (tests).

    Installing ``aurey[api]`` is required to import :mod:`fastapi`.
    """

    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    injected = state

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if injected is not None:
            app.state.aurey = injected
        else:
            try:
                app.state.aurey = bootstrap_aurey_service_state(settings)
            except AureyServiceBootstrapError:
                app.state.aurey = None
        yield

    app = FastAPI(title="Aurey", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Literal[True]]:
        return {"ok": True}

    @app.post("/v1/invoke", response_model=InvokeResponse)
    def invoke(request: Request, turn: InvokeBody) -> InvokeResponse:
        svc = get_aurey_service_state(request)
        if svc is None:
            return _misconfigured_response(turn.session_id)

        extra: dict[str, Any] = {}
        if turn.context is not None:
            extra["aurey_context"] = turn.context
        config = thread_config(turn.session_id, **extra)

        try:
            graph = svc.get_or_create_graph(turn.agent_model_spec)
        except RuntimeError as exc:
            code = "deep_agent_unavailable"
            msg_lower = str(exc).lower()
            if "deepagents" in msg_lower:
                code = "deep_agent_dependency"
            return InvokeResponse(
                ok=False,
                session_id=turn.session_id,
                error=InvokeError(
                    code=code,
                    message="The deep agent runtime is not available or misconfigured.",
                ),
            )

        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=turn.message)]},
                config=config,
            )
        except Exception:
            return InvokeResponse(
                ok=False,
                session_id=turn.session_id,
                error=InvokeError(
                    code="agent_invoke_failed",
                    message="The agent failed to complete this turn.",
                ),
            )

        raw_messages = result.get("messages") if isinstance(result, dict) else None
        if not isinstance(raw_messages, list):
            raw_messages = []

        return InvokeResponse(
            ok=True,
            session_id=turn.session_id,
            messages=_summarize_agent_messages(raw_messages),
        )

    return app


def create_default_application():
    """Entry point for ``uvicorn aurey.service.app:create_default_application --factory``."""

    return create_fastapi_application()


# Uvicorn ASGI factory: ``uvicorn aurey.service.app:app --factory``
app = create_default_application


__all__ = [
    "InvokeBody",
    "InvokeError",
    "InvokeResponse",
    "app",
    "create_default_application",
    "create_fastapi_application",
]
