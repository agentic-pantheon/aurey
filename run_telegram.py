"""Run Aurey's Telegram bot (long polling). This is separate from the HTTP server (Uvicorn).

Sync deps: ``uv sync --group dev --extra telegram``
To keep Uvicorn too: ``uv sync --group dev --extra telegram --extra api``
"""

from __future__ import annotations

import argparse
import logging

from aurey.logging_setup import configure_aurey_console_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Aurey Telegram + Rich console logs")
    parser.add_argument(
        "--log-level",
        default="debug",
        help="Root / aurey log level (debug, info, warning, …)",
    )
    args = parser.parse_args()

    level = getattr(logging, args.log_level.upper(), logging.DEBUG)
    configure_aurey_console_logging(level=level)

    from aurey.telegram import create_telegram_application

    create_telegram_application().run_polling()


if __name__ == "__main__":
    main()
