"""Sanitized custody-layer exceptions."""


class CustodyError(RuntimeError):
    """Base class for custody and secret-store failures."""


class SecretStoreError(CustodyError):
    """Base class for secret-store failures."""


class SecretNotFoundError(SecretStoreError):
    """Raised when a secret path is not present in the secret store."""

    def __init__(self, path: str) -> None:
        super().__init__(f"Secret not found at path '{path}'.")
        self.path = path


class EmptySecretValueError(SecretStoreError):
    """Raised when a secret exists but has no usable value."""

    def __init__(self, path: str) -> None:
        super().__init__(f"Secret at path '{path}' is empty.")
        self.path = path


class OneClawSigningError(RuntimeError):
    """Raised when 1Claw returns a signing response that cannot be used (no secrets in messages)."""


class SecretStoreUnavailableError(SecretStoreError):
    """Raised when the backing secret store cannot be reached or queried."""

    def __init__(
        self,
        path: str,
        store_name: str = "secret store",
        *,
        detail: str | None = None,
    ) -> None:
        self.path = path
        self.store_name = store_name
        if detail is not None:
            super().__init__(f"{store_name}: {detail}")
        else:
            super().__init__(f"{store_name} could not resolve secret path '{path}'.")
