"""Path-based secret store: 1Claw HTTP client, in-memory fakes, and SecretValue wrapper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from aurey.custody.errors import (
    EmptySecretValueError,
    SecretNotFoundError,
    SecretStoreUnavailableError,
)


@dataclass(frozen=True, repr=False)
class SecretValue:
    """Typed secret value whose raw string is only available by explicit reveal."""

    path: str
    _value: str

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("Secret path must not be empty.")
        if not self._value.strip():
            raise EmptySecretValueError(self.path)

    def reveal(self) -> str:
        """Return the raw secret value for the narrow caller that needs it."""

        return self._value

    def __repr__(self) -> str:
        return f"SecretValue(path={self.path!r}, value=<redacted>)"

    def __str__(self) -> str:
        return self.__repr__()


@runtime_checkable
class SecretStore(Protocol):
    """Path-based secret store interface."""

    def get_secret(self, path: str) -> SecretValue:
        """Resolve a secret path into a typed secret value."""


@runtime_checkable
class OneClawClient(Protocol):
    """Minimal 1Claw client shape used by the secret-store wrapper."""

    def get_secret(self, *, vault_id: str, path: str, agent_id: str | None = None) -> str:
        """Return the raw secret string for a path."""


class OneClawSecretStore:
    """SecretStore implementation backed by a 1Claw client."""

    def __init__(
        self,
        *,
        client: OneClawClient,
        vault_id: str,
        agent_id: str | None = None,
    ) -> None:
        if not vault_id.strip():
            raise ValueError("1Claw vault ID must not be empty.")

        self._client = client
        self._vault_id = vault_id
        self._agent_id = agent_id

    def get_secret(self, path: str) -> SecretValue:
        """Resolve a non-wallet secret path through 1Claw."""

        if not path.strip():
            raise ValueError("Secret path must not be empty.")

        try:
            value = self._client.get_secret(
                vault_id=self._vault_id,
                path=path,
                agent_id=self._agent_id,
            )
        except SecretNotFoundError:
            raise
        except EmptySecretValueError:
            raise
        except SecretStoreUnavailableError:
            raise
        except Exception as exc:
            raise SecretStoreUnavailableError(path, store_name="1Claw") from exc

        if value is None:
            raise SecretNotFoundError(path)

        return SecretValue(path=path, _value=value)


class OneClawHttpClient:
    """Small stdlib HTTP adapter for 1Claw-compatible secret reads.

    Hosted ``api.1claw.xyz`` flow (when ``agent_id`` is passed on each read):

    1. ``POST /v1/auth/agent-token`` with ``agent_id`` and ``api_key`` (agent API key).
    2. ``GET /v1/vaults/{vault_id}/secrets/{path}`` with ``Authorization: Bearer <jwt>``.

    When ``agent_id`` is omitted, uses legacy ``POST .../secrets:resolve`` with the API key
    as the bearer token (for older or self-hosted deployments).
    """

    def __init__(self, *, base_url: str, api_key: str) -> None:
        if not base_url.strip():
            raise ValueError("1Claw base URL must not be empty.")
        if not api_key.strip():
            raise ValueError("1Claw API key must not be empty.")

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._access_token: str | None = None
        self._access_token_agent: str | None = None

    def get_secret(self, *, vault_id: str, path: str, agent_id: str | None = None) -> str:
        """Read a secret value from 1Claw without exposing it in errors."""

        agent = agent_id.strip() if agent_id and agent_id.strip() else None
        if agent is not None:
            return self._get_secret_hosted(vault_id=vault_id, path=path, agent_id=agent)
        return self._get_secret_legacy_resolve(vault_id=vault_id, path=path)

    def _invalidate_access_token(self, agent_id: str) -> None:
        if self._access_token_agent == agent_id:
            self._access_token = None
            self._access_token_agent = None

    def _fetch_access_token(self, agent_id: str) -> str:
        url = f"{self._base_url}/v1/auth/agent-token"
        body = json.dumps({"agent_id": agent_id, "api_key": self._api_key}).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise SecretStoreUnavailableError(
                "/v1/auth/agent-token",
                store_name="1Claw",
                detail=(
                    f"Agent token exchange failed with HTTP {exc.code}. "
                    "This happens before any vault secret (e.g. Telegram path) is read. "
                    "Check `oneclaw_agent_id`, bootstrap API key, and 1Claw availability."
                ),
            ) from exc
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise SecretStoreUnavailableError(
                "/v1/auth/agent-token",
                store_name="1Claw",
                detail=(
                    "Agent token exchange failed (network error or invalid JSON). "
                    "This step runs before reading a vault secret path."
                ),
            ) from exc

        token = payload.get("access_token")
        if not isinstance(token, str) or not token.strip():
            raise SecretStoreUnavailableError(
                "/v1/auth/agent-token",
                store_name="1Claw",
                detail="Agent token response contained no usable `access_token`.",
            )

        self._access_token = token.strip()
        self._access_token_agent = agent_id
        return self._access_token

    def _bearer_for_agent(self, agent_id: str) -> str:
        if self._access_token and self._access_token_agent == agent_id:
            return self._access_token
        return self._fetch_access_token(agent_id)

    def _secret_url_path_suffix(self, path: str) -> str:
        normalized = path.strip().lstrip("/")
        if not normalized:
            raise ValueError("Secret path must not be empty.")
        return quote(normalized, safe="/")

    def _get_secret_hosted(self, *, vault_id: str, path: str, agent_id: str) -> str:
        bearer = self._bearer_for_agent(agent_id)
        try:
            return self._http_get_secret_value(vault_id, path, bearer)
        except HTTPError as exc:
            if exc.code == 401:
                self._invalidate_access_token(agent_id)
                bearer = self._bearer_for_agent(agent_id)
                return self._http_get_secret_value(vault_id, path, bearer)
            if exc.code == 404:
                raise SecretNotFoundError(path) from exc
            if exc.code == 403:
                raise SecretStoreUnavailableError(
                    path,
                    store_name="1Claw",
                    detail=(
                        "HTTP 403 Forbidden - the agent token is valid but this principal "
                        "is not allowed to read that vault or secret path."
                    ),
                ) from exc
            raise SecretStoreUnavailableError(path, store_name="1Claw") from exc

    def _http_get_secret_value(self, vault_id: str, path: str, bearer: str) -> str:
        suffix = self._secret_url_path_suffix(path)
        url = f"{self._base_url}/v1/vaults/{vault_id}/secrets/{suffix}"
        request = Request(
            url,
            headers={"Authorization": f"Bearer {bearer}"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=10) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError:
            raise
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise SecretStoreUnavailableError(path, store_name="1Claw") from exc

        value = _extract_secret_value(response_payload)
        if value is None:
            raise SecretNotFoundError(path)
        return value

    def _get_secret_legacy_resolve(self, *, vault_id: str, path: str) -> str:
        url = f"{self._base_url}/v1/vaults/{vault_id}/secrets:resolve"
        payload = json.dumps({"path": path}).encode("utf-8")
        request = Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=10) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 404:
                raise SecretNotFoundError(path) from exc
            raise SecretStoreUnavailableError(path, store_name="1Claw") from exc
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise SecretStoreUnavailableError(path, store_name="1Claw") from exc

        value = _extract_secret_value(response_payload)
        if value is None:
            raise SecretNotFoundError(path)
        return value


class FakeSecretStore:
    """In-memory SecretStore for tests."""

    def __init__(self, secrets: dict[str, str] | None = None) -> None:
        self._secrets = dict(secrets or {})

    def get_secret(self, path: str) -> SecretValue:
        try:
            value = self._secrets[path]
        except KeyError as exc:
            raise SecretNotFoundError(path) from exc

        return SecretValue(path=path, _value=value)


class FakeOneClawClient:
    """In-memory OneClawClient for wrapper tests."""

    def __init__(self, secrets: dict[str, str] | None = None) -> None:
        self._secrets = dict(secrets or {})
        self.requests: list[dict[str, str | None]] = []

    def get_secret(self, *, vault_id: str, path: str, agent_id: str | None = None) -> str:
        self.requests.append({"vault_id": vault_id, "path": path, "agent_id": agent_id})
        try:
            return self._secrets[path]
        except KeyError as exc:
            raise SecretNotFoundError(path) from exc


def _extract_secret_value(payload: dict[str, Any]) -> str | None:
    value = payload.get("value")
    if isinstance(value, str):
        return value

    secret = payload.get("secret")
    if isinstance(secret, dict):
        nested_value = secret.get("value")
        if isinstance(nested_value, str):
            return nested_value

    return None
