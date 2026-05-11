"""Map LiFi ``transactionRequest`` dicts into ``PreparedTxEnvelope``."""

from __future__ import annotations

from typing import Any

from aurey.graphs.evm_codec import normalize_evm_address, parse_evm_uint
from aurey.graphs.results import PreparedTxEnvelope


def lifi_transaction_request_to_envelope(
    *,
    chain_id: int,
    from_address: str,
    transaction_request: dict[str, Any],
    signing_key_secret_path: str,
) -> PreparedTxEnvelope:
    """Normalize an ethers-style tx request from LiFi into our execute envelope.

    ``from`` / ``chainId`` fields on the LiFi payload, when present, must match expectations.
    """

    tr = transaction_request
    if not isinstance(tr, dict):
        raise ValueError("transaction_request must be an object.")

    to_raw = tr.get("to")
    if not to_raw:
        raise ValueError("transaction_request.to is required.")

    to_norm = normalize_evm_address(str(to_raw))
    from_norm = normalize_evm_address(from_address)

    if "from" in tr and tr["from"] is not None:
        li_from = normalize_evm_address(str(tr["from"]))
        if li_from != from_norm:
            raise ValueError("transaction_request.from does not match from_address.")

    if "chainId" in tr and tr["chainId"] is not None:
        raw_cid = tr["chainId"]
        if isinstance(raw_cid, str):
            cid = int(raw_cid, 0)
        else:
            cid = int(raw_cid)
        if cid != chain_id:
            raise ValueError("transaction_request.chainId does not match chain.")

    data = tr.get("data")
    if data is None or data == "":
        data = "0x"
    elif isinstance(data, str):
        d = data.strip()
        data = d if d.startswith("0x") else ("0x" + d)
    else:
        raise ValueError("transaction_request.data must be a hex string.")

    value_raw = tr.get("value")
    if value_raw is None:
        value_int = 0
    else:
        value_int = parse_evm_uint(value_raw)

    value_hex = hex(value_int)

    gas_limit_hex: str | None = None
    gas_raw = tr.get("gasLimit")
    if gas_raw is None:
        gas_raw = tr.get("gas")
    if gas_raw is not None:
        gas_limit_hex = hex(parse_evm_uint(gas_raw))

    nonce: int | None = None
    if tr.get("nonce") is not None:
        nonce = int(parse_evm_uint(tr["nonce"]))

    return PreparedTxEnvelope(
        kind="lifi_swap",
        chain_id=chain_id,
        from_address=from_norm,
        to=to_norm,
        data=data,
        value_hex=value_hex,
        gas_limit_hex=gas_limit_hex,
        nonce=nonce,
        signing_key_secret_path=signing_key_secret_path,
    )
