"""Small chain metadata helpers (Mercury-parity naming, minimal surface)."""

from __future__ import annotations

from dataclasses import dataclass

from aurey.settings import AureySettings


@dataclass(frozen=True)
class ChainInfo:
    name: str
    chain_id: int
    rpc_secret_settings_attr: str


CHAIN_INDEX: dict[str, ChainInfo] = {
    "ethereum": ChainInfo("ethereum", 1, "ethereum_rpc_secret_path"),
    "base": ChainInfo("base", 8453, "base_rpc_secret_path"),
}


def chain_info(name: str) -> ChainInfo | None:
    key = name.strip().lower()
    return CHAIN_INDEX.get(key)


def rpc_secret_path_for_chain(settings: AureySettings, name: str) -> str | None:
    info = chain_info(name)
    if info is None:
        return None
    return getattr(settings, info.rpc_secret_settings_attr)


def chain_id_for(name: str) -> int | None:
    info = chain_info(name)
    return None if info is None else info.chain_id
