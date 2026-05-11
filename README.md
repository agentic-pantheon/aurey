# Aurey

Python 3.12+ scaffold for a standalone **Deep Agent** with **LangGraph**-backed tools, **Pydantic Settings**, and an optional **FastAPI** invoke API. Production wiring resolves RPC and provider material through **1Claw** (`SecretStore`); only the bootstrap API key is read from the environment variable named by `oneclaw_api_key_secret_source`.

## Layout

- `src/aurey/settings/` - Pydantic settings: 1Claw connection fields, **vault path** references only (never inline secrets). Bootstrap API key is read via the env var **named** by `oneclaw_api_key_secret_source` (default `AUREY_ONECLAW_BOOTSTRAP_API_KEY`).
- `src/aurey/custody/` - `SecretStore` protocol, `SecretValue`, `OneClawHttpClient` / `OneClawSecretStore`, and in-memory `Fake*` helpers for tests.
- `src/aurey/reasoning/` - deep agent harness and factory.
- `src/aurey/tools/` - LangChain tool definitions.
- `src/aurey/graphs/` - compiled subgraphs per tool.
- `src/aurey/service/` - optional HTTP boundary (`bootstrap`, adapters, `FastAPI` app, DI helpers).
- `src/aurey/telegram/` - optional Telegram bot client reusing the service invoke path.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional HTTP stack:

```bash
pip install -e ".[dev,api]"
```

Optional Telegram stack:

```bash
pip install -e ".[dev,telegram]"
```

Copy `.env.example` to `.env` and set at least `AUREY_ONECLAW_VAULT_ID` and `AUREY_ONECLAW_BOOTSTRAP_API_KEY` before running the service.

### Optional HTTP server

After installing `aurey[api]`, run Uvicorn with the ASGI factory (builds the app after lifespan wiring):

```bash
uvicorn aurey.service.app:app --factory --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health` - liveness (`{"ok": true}`).
- `POST /v1/invoke` - JSON body: `message`, `session_id`, optional `context` (stored under configurable `aurey_context`), optional `model`. Responses are structured (`InvokeResponse`); misconfiguration and agent failures use stable error codes without embedding secrets.

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

## Development

```bash
ruff check src tests
pytest
```
