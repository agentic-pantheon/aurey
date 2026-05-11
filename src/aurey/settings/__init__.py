"""Application settings (paths-only secret references, 1Claw connection config).

Note: Configuration lives in this package intentionally; do not add a sibling
``aurey/settings.py`` module, which would conflict with this package name.
"""

from __future__ import annotations

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AureySettings(BaseSettings):
    """Loaded from environment variables with prefix ``AUREY_``.

    Secret *paths* refer to vault paths resolved at runtime via a ``SecretStore``;
    the bootstrap API key is read once from whichever env variable name you set in
    ``oneclaw_api_key_secret_source`` (another env variable's **name**, not its value).
    """

    model_config = SettingsConfigDict(
        env_prefix="AUREY_",
        extra="ignore",
    )

    oneclaw_base_url: str = Field(
        default="https://api.1claw.xyz",
        description="1Claw API base URL (no trailing slash required).",
    )
    oneclaw_vault_id: str = Field(
        default="",
        description="Vault identifier for secret reads.",
    )
    oneclaw_api_key_secret_source: str = Field(
        default="AUREY_ONECLAW_BOOTSTRAP_API_KEY",
        description="Name of the environment variable that holds the bootstrap 1Claw API key.",
    )
    oneclaw_agent_id: str | None = Field(
        default=None,
        description="Optional agent id for hosted token exchange flow.",
    )

    ethereum_rpc_secret_path: str | None = Field(
        default=None,
        description="1Claw vault path for Ethereum RPC URL (never the URL itself here).",
    )
    base_rpc_secret_path: str | None = Field(
        default=None,
        description="1Claw vault path for Base RPC URL.",
    )
    alchemy_api_secret_path: str | None = Field(
        default=None,
        description="1Claw vault path for Alchemy API key.",
    )
    lifi_api_secret_path: str | None = Field(
        default=None,
        description="1Claw vault path for LiFi API key.",
    )
    wallet_signing_key_secret_path: str | None = Field(
        default=None,
        description="1Claw vault path for signing material.",
    )

    def resolve_oneclaw_bootstrap_api_key(self) -> str:
        """Return bootstrap API key from the env named by ``oneclaw_api_key_secret_source``."""

        name = self.oneclaw_api_key_secret_source.strip()
        if not name:
            raise ValueError("oneclaw_api_key_secret_source must not be empty.")
        raw = os.environ.get(name)
        if raw is None:
            raise KeyError(name)
        value = raw.strip()
        if not value:
            raise ValueError(f"Environment variable {name!r} is set but empty.")
        return value
