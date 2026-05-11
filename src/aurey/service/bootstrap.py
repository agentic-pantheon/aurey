"""Build :class:`~aurey.service.state.AureyServiceState` from settings."""

from __future__ import annotations

from aurey.custody.secret_store import OneClawHttpClient, OneClawSecretStore
from aurey.graphs.evm_tx_pipeline import Web3TxPipeline
from aurey.reasoning import make_memory_checkpointer
from aurey.runtime import AureyRuntime
from aurey.service.adapters import UrllibHttpJsonClient, make_evm_rpc_factory
from aurey.service.state import AureyServiceState
from aurey.settings import AureySettings


class AureyServiceBootstrapError(RuntimeError):
    """Mandatory service wiring failed; messages must not contain secret values."""


def bootstrap_aurey_service_state(settings: AureySettings | None = None) -> AureyServiceState:
    """Wire 1Claw secret store, runtime, and one in-memory checkpointer per process."""

    s = settings or AureySettings()
    vault_id = (s.oneclaw_vault_id or "").strip()
    if not vault_id:
        raise AureyServiceBootstrapError("1Claw vault id is not configured.")

    try:
        api_key = s.resolve_oneclaw_bootstrap_api_key()
    except KeyError:
        raise AureyServiceBootstrapError(
            "Bootstrap 1Claw API key is unavailable (bootstrap env var unset)."
        ) from None
    except ValueError:
        raise AureyServiceBootstrapError(
            "Bootstrap 1Claw API key configuration is invalid."
        ) from None

    client = OneClawHttpClient(base_url=s.oneclaw_base_url.strip(), api_key=api_key)
    store = OneClawSecretStore(client=client, vault_id=vault_id, agent_id=s.oneclaw_agent_id)

    runtime = AureyRuntime(
        settings=s,
        secret_store=store,
        evm_rpc_factory=make_evm_rpc_factory(),
        http=UrllibHttpJsonClient(),
        tx_pipeline=Web3TxPipeline(settings=s, secret_store=store),
    )

    default_model = (s.deep_agent_default_model or "").strip() or "openai:gpt-4o-mini"

    return AureyServiceState(
        settings=s,
        runtime=runtime,
        checkpointer=make_memory_checkpointer(),
        default_model=default_model,
    )


__all__ = ["AureyServiceBootstrapError", "bootstrap_aurey_service_state"]
