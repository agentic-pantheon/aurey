"""AureySettings defaults and environment behavior."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aurey.settings import AureySettings


def test_settings_defaults():
    s = AureySettings()
    assert s.oneclaw_base_url == "https://api.1claw.xyz"
    assert s.oneclaw_vault_id == ""
    assert s.oneclaw_api_key_secret_source == "AUREY_ONECLAW_BOOTSTRAP_API_KEY"
    assert s.oneclaw_agent_id is None
    assert s.alchemy_api_secret_path is None
    assert s.lifi_api_secret_path is None
    assert s.lifi_integrator == "aurey"
    assert s.evm_signing_mode == "vault_key"
    assert s.evm_signing_requires_wallet_signing_key_secret_path is True
    assert s.wallet_signing_key_secret_path is None
    assert s.telegram_bot_token_secret_path is None
    assert s.deep_agent_default_model == "openai:gpt-4o-mini"
    assert s.database_url is None


def test_settings_env_override(monkeypatch):
    monkeypatch.delenv("AUREY_ONECLAW_BASE_URL", raising=False)
    monkeypatch.delenv("AUREY_ONECLAW_VAULT_ID", raising=False)
    monkeypatch.delenv("AUREY_ALCHEMY_API_SECRET_PATH", raising=False)
    monkeypatch.delenv("AUREY_EVM_SIGNING_MODE", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("AUREY_DATABASE_URL", raising=False)
    monkeypatch.setenv("AUREY_ONECLAW_BASE_URL", "https://example.invalid/v1/")
    monkeypatch.setenv("AUREY_ONECLAW_VAULT_ID", "vault-env-123")
    monkeypatch.setenv("AUREY_ALCHEMY_API_SECRET_PATH", "aurey/apis/alchemy")

    s = AureySettings()
    assert s.oneclaw_base_url == "https://example.invalid/v1/"
    assert s.oneclaw_vault_id == "vault-env-123"
    assert s.alchemy_api_secret_path == "aurey/apis/alchemy"


def test_settings_evm_signing_mode_env_override(monkeypatch):
    monkeypatch.delenv("AUREY_EVM_SIGNING_MODE", raising=False)
    monkeypatch.setenv("AUREY_EVM_SIGNING_MODE", "oneclaw_intents")

    s = AureySettings()
    assert s.evm_signing_mode == "oneclaw_intents"
    assert s.evm_signing_requires_wallet_signing_key_secret_path is False


def test_settings_evm_signing_mode_invalid_rejected(monkeypatch):
    monkeypatch.setenv("AUREY_EVM_SIGNING_MODE", "bogus")

    with pytest.raises(ValidationError):
        AureySettings()


def test_settings_database_url_from_database_url_env(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("AUREY_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@db:5432/app")
    s = AureySettings()
    assert s.database_url == "postgres://user:pass@db:5432/app"


def test_settings_database_url_aurey_prefixed(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("AUREY_DATABASE_URL", raising=False)
    monkeypatch.setenv("AUREY_DATABASE_URL", "postgres://local/aurey")
    s = AureySettings()
    assert s.database_url == "postgres://local/aurey"


def test_resolve_oneclaw_bootstrap_api_key(monkeypatch):
    monkeypatch.setenv("CUSTOM_BOOTSTRAP", "bootstrap-value")
    s = AureySettings(oneclaw_api_key_secret_source="CUSTOM_BOOTSTRAP")
    assert s.resolve_oneclaw_bootstrap_api_key() == "bootstrap-value"


def test_resolve_oneclaw_bootstrap_api_key_trim(monkeypatch):
    monkeypatch.setenv("CUSTOM_BOOTSTRAP", "  spaced  ")
    s = AureySettings(oneclaw_api_key_secret_source="CUSTOM_BOOTSTRAP")
    assert s.resolve_oneclaw_bootstrap_api_key() == "spaced"


def test_resolve_oneclaw_bootstrap_api_key_missing(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    s = AureySettings(oneclaw_api_key_secret_source="MISSING_KEY")
    with pytest.raises(KeyError):
        s.resolve_oneclaw_bootstrap_api_key()


def test_resolve_oneclaw_bootstrap_api_key_empty_env(monkeypatch):
    monkeypatch.setenv("EMPTY_KEY", "")
    s = AureySettings(oneclaw_api_key_secret_source="EMPTY_KEY")
    with pytest.raises(ValueError):
        s.resolve_oneclaw_bootstrap_api_key()
