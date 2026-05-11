"""Small chain metadata helpers (Mercury-parity naming, minimal surface)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChainInfo:
    name: str
    chain_id: int
    alchemy_network: str


CHAIN_INDEX: dict[str, ChainInfo] = {
    "ethereum": ChainInfo("ethereum", 1, "eth-mainnet"),
    "base": ChainInfo("base", 8453, "base-mainnet"),
}


def chain_info(name: str) -> ChainInfo | None:
    key = name.strip().lower()
    return CHAIN_INDEX.get(key)


def alchemy_rpc_url_for_chain(name: str, api_key: str) -> str | None:
    info = chain_info(name)
    if info is None:
        return None
    return f"https://{info.alchemy_network}.g.alchemy.com/v2/{api_key}"


def chain_id_for(name: str) -> int | None:
    info = chain_info(name)
    return None if info is None else info.chain_id


def chain_name_for_id(chain_id: int) -> str | None:
    """Return canonical chain slug (e.g. ``base``) for a numeric chain id, if known."""

    for name, info in CHAIN_INDEX.items():
        if info.chain_id == chain_id:
            return name
    return None
