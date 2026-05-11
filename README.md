# Aurey

Python 3.12+ service scaffold for a standalone **Deep Agent** with **LangGraph**-backed tools and **Pydantic Settings**. Runtime wiring (1Claw secret store, FastAPI) is introduced in follow-up work.

## Layout

- `src/aurey/settings/` - Pydantic settings: 1Claw connection fields, **vault path** references only (never inline secrets). Bootstrap API key is read via the env var **named** by `oneclaw_api_key_secret_source` (default `AUREY_ONECLAW_BOOTSTRAP_API_KEY`).
- `src/aurey/custody/` - `SecretStore` protocol, `SecretValue`, `OneClawHttpClient` / `OneClawSecretStore`, and in-memory `Fake*` helpers for tests.
- `src/aurey/reasoning/` - deep agent harness and factory.
- `src/aurey/tools/` - LangChain tool definitions.
- `src/aurey/graphs/` - compiled subgraphs per tool.

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

Copy `.env.example` to `.env` and adjust placeholders as integration lands.

## Development

```bash
ruff check src tests
pytest
```
