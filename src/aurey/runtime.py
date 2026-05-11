"""Wires settings, SecretStore, and injectable client factories for graphs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from aurey.custody.secret_store import SecretStore
from aurey.graphs.ports import EvmJsonRpcPort, HttpJsonPort, TxPipelinePort
from aurey.settings import AureySettings


@dataclass(frozen=True)
class AureyRuntime:
    """Process-level dependencies; secret values are revealed only inside graph nodes."""

    settings: AureySettings
    secret_store: SecretStore
    evm_rpc_factory: Callable[[str], EvmJsonRpcPort]
    http: HttpJsonPort
    tx_pipeline: TxPipelinePort
    lifi_base_url: str = "https://li.quest"
