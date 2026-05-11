"""Optional Telegram client that reuses the shared Aurey invoke service path."""

from __future__ import annotations

from typing import Any

from aurey.custody.errors import SecretNotFoundError, SecretStoreUnavailableError
from aurey.service.bootstrap import bootstrap_aurey_service_state
from aurey.service.invoke import AgentInvokeResult, invoke_deep_agent_turn
from aurey.service.state import AureyServiceState
from aurey.settings import AureySettings


class TelegramConfigurationError(RuntimeError):
    """Telegram setup failed without exposing token paths or values."""


def resolve_telegram_bot_token(state: AureyServiceState) -> str:
    """Resolve the Telegram bot token via SecretStore using a configured vault path."""

    path = state.settings.telegram_bot_token_secret_path
    if not path:
        raise TelegramConfigurationError("Telegram bot token secret path is not configured.")
    try:
        return state.runtime.secret_store.get_secret(path).reveal()
    except SecretNotFoundError as exc:
        raise TelegramConfigurationError("Telegram bot token could not be resolved.") from exc
    except SecretStoreUnavailableError as exc:
        raise TelegramConfigurationError("Secret store unavailable for Telegram token.") from exc


def _last_text_message(result: AgentInvokeResult) -> str:
    if result.messages:
        for row in reversed(result.messages):
            content = row.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return "Done."


def handle_telegram_text(
    state: AureyServiceState,
    *,
    chat_id: int | str,
    text: str,
    user_id: int | str | None = None,
    model: str | None = None,
) -> str:
    """Handle one inbound Telegram text message and return safe text for ``reply_text``."""

    session_id = f"telegram:{chat_id}"
    context: dict[str, Any] = {"telegram_chat_id": str(chat_id)}
    if user_id is not None:
        context["telegram_user_id"] = str(user_id)
    result = invoke_deep_agent_turn(
        state,
        message=text,
        session_id=session_id,
        context=context,
        model=model,
    )
    if result.ok:
        return _last_text_message(result)
    assert result.error is not None
    return f"Aurey error ({result.error.code}): {result.error.message}"


def _import_telegram_ext():
    try:
        from telegram import Update
        from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Telegram support requires the optional 'telegram' extra: "
            "pip install -e '.[telegram]'"
        ) from exc
    return Application, CommandHandler, ContextTypes, MessageHandler, Update, filters


def build_telegram_application(
    *,
    state: AureyServiceState,
    token: str | None = None,
    model: str | None = None,
):
    """Build a python-telegram-bot Application with Aurey's shared invoke path."""

    (
        Application,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        Update,
        filters,
    ) = _import_telegram_ext()
    bot_token = token or resolve_telegram_bot_token(state)

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        _ = context
        if update.effective_message is not None:
            await update.effective_message.reply_text(
                "Aurey is ready. Send a message to invoke the agent."
            )

    async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        _ = context
        msg = update.effective_message
        if msg is None or not msg.text:
            return
        chat = update.effective_chat
        user = update.effective_user
        reply = handle_telegram_text(
            state,
            chat_id=getattr(chat, "id", "unknown"),
            user_id=getattr(user, "id", None),
            text=msg.text,
            model=model,
        )
        await msg.reply_text(reply)

    app = Application.builder().token(bot_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    return app


def create_telegram_application(
    *,
    state: AureyServiceState | None = None,
    settings: AureySettings | None = None,
    model: str | None = None,
):
    """Bootstrap service state and return a Telegram polling Application."""

    svc = state or bootstrap_aurey_service_state(settings)
    return build_telegram_application(state=svc, model=model)


__all__ = [
    "TelegramConfigurationError",
    "build_telegram_application",
    "create_telegram_application",
    "handle_telegram_text",
    "resolve_telegram_bot_token",
]
