"""Minimal ABI encoding helpers (no web3 dependency)."""

from __future__ import annotations


def normalize_evm_address(addr: str) -> str:
    raw = addr.strip()
    if len(raw) != 42 or not raw.startswith("0x"):
        raise ValueError("EVM address must be 0x-prefixed 20 bytes.")
    body = raw[2:].lower()
    int(body, 16)
    if len(body) != 40:
        raise ValueError("EVM address must be 20 bytes hex.")
    return "0x" + body


def _strip_0x(h: str) -> str:
    return h[2:] if h.startswith(("0x", "0X")) else h


def _pad_addr(addr: str) -> str:
    core = _strip_0x(normalize_evm_address(addr))
    return (64 - len(core)) * "0" + core


def _pad_uint256(n: int) -> str:
    if n < 0:
        raise ValueError("uint256 must be non-negative.")
    return f"{n:064x}"


def erc20_transfer_data(to: str, amount_wei: int) -> str:
    return "0xa9059cbb" + _pad_addr(to) + _pad_uint256(amount_wei)


def erc20_approve_data(spender: str, amount_wei: int) -> str:
    return "0x095ea7b3" + _pad_addr(spender) + _pad_uint256(amount_wei)
