# Aurey

Python 3.12+ scaffold for a standalone **Deep Agent** with **LangGraph**-backed tools, **Pydantic Settings**, and an optional **FastAPI** invoke API. Production wiring resolves provider material through **1Claw** (`SecretStore`); EVM RPC URLs are derived from the Alchemy API key path, and only the bootstrap API key is read from the environment variable named by `oneclaw_api_key_secret_source`.

## Layout

- `src/aurey/settings/` - Pydantic settings: 1Claw connection fields, **vault path** references only (never inline secrets). Bootstrap API key is read via the env var **named** by `oneclaw_api_key_secret_source` (default `AUREY_ONECLAW_BOOTSTRAP_API_KEY`).
- `src/aurey/custody/` - `SecretStore` protocol, `SecretValue`, `OneClawHttpClient` / `OneClawSecretStore`, and in-memory `Fake`* helpers for tests.
- `src/aurey/reasoning/` - deep agent harness and factory.
- `src/aurey/tools/` - LangChain tool definitions.
- `src/aurey/graphs/` - compiled subgraphs per tool.
- `src/aurey/service/` - optional HTTP boundary (`bootstrap`, adapters, `FastAPI` app, DI helpers).
- `src/aurey/telegram/` - optional Telegram bot client reusing the service invoke path.

## Setup

```bash
uv sync --group dev
```

Optional HTTP stack:

```bash
uv sync --group dev --extra api
```

Optional Telegram stack:

```bash
uv sync --group dev --extra telegram
```

Copy `.env.example` to `.env` and set at least `AUREY_ONECLAW_VAULT_ID` and `AUREY_ONECLAW_BOOTSTRAP_API_KEY` before running the service.

For default models like `openai:gpt-4o-mini`, set `OPENAI_API_KEY` in `.env`. The project depends on `langchain-openai`; run `uv sync --group dev` after pulling so it is installed.

### Optional HTTP server

After installing `aurey[api]`, run Uvicorn with the ASGI factory (builds the app after lifespan wiring):

```bash
uv run uvicorn aurey.service.app:app --factory --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health` - readiness (`{"ok": true}` when bootstrap succeeded, `{"ok": false}` if required wiring such as 1Claw is missing).
- `POST /v1/invoke` - JSON body: `message`, `session_id`, optional `context` (stored under configurable `aurey_context`), optional `model`. Responses are structured (`InvokeResponse`); misconfiguration and agent failures use stable error codes without embedding secrets.

With `DATABASE_URL` or `AUREY_DATABASE_URL` set and `aurey[api]` installed, the service uses a **PostgreSQL** LangGraph checkpointer (tables are created on startup via `setup()`). Without a DB URL it uses an in-memory saver.

### Deploying on Railway

1. Create a **Postgres** service in the same Railway project as the app.
2. On the app service, set `DATABASE_URL` to `${{Postgres.DATABASE_URL}}` (use the exact Postgres service name Railway shows; references are case-sensitive).
3. Set the usual secrets: `AUREY_ONECLAW_VAULT_ID`, `AUREY_ONECLAW_BOOTSTRAP_API_KEY`, provider keys (e.g. `OPENAI_API_KEY`), and optional LangSmith vars (`LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`). See `.env.example`.
4. The repo includes [`railway.toml`](railway.toml) with `uv sync --extra api` and a Uvicorn start command bound to `0.0.0.0` and `$PORT`.

The **Telegram** bot (`run_telegram.py`) is a separate process; run it as another Railway service if you need it in production.

For tests, inject `state=` into `create_fastapi_application` or monkeypatch `create_aurey_deep_agent` so no live model or network is required.

### Optional Telegram bot

Telegram reuses the same `AureyServiceState` and deep-agent invocation path as `POST /v1/invoke`. Store the bot token in 1Claw and configure only the vault path:

```bash
AUREY_TELEGRAM_BOT_TOKEN_SECRET_PATH=aurey/telegram/bot_token
```

Then bootstrap and run polling from a small entrypoint:

```python
from aurey.telegram import create_telegram_application

create_telegram_application().run_polling()
```

Do not place the Telegram token in `.env`; only the 1Claw path belongs in configuration.

To **restrict** which conversations can use the bot, set `AUREY_TELEGRAM_ALLOWED_CHAT_IDS` to a comma- or whitespace-separated list of numeric Telegram **chat** ids (omit or leave empty for no restriction). In a private chat with you, the chat id is the same as your user id; groups and supergroups use negative ids (often starting with `-100`). Discover ids by forwarding a message to a bot such as `@RawDataBot` or by temporarily logging `effective_chat.id` from updates.

## Development

```bash
uv run ruff check src tests
uv run pytest
```

