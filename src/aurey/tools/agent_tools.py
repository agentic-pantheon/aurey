"""LangChain tools wrapping compiled Aurey LangGraph subgraph invocations."""

from __future__ import annotations

import time
from typing import Any, Literal, Self

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aurey.graphs import (
    SwapPrepareInput,
    TxExecuteInput,
    TxPrepareErc20Approval,
    TxPrepareErc20Transfer,
    TxPrepareLiFiInput,
    TxPrepareNative,
    build_alchemy_graph,
    build_read_graph,
    build_swap_prepare_graph,
    build_tx_execute_graph,
    build_tx_prepare_graph,
    build_tx_prepare_lifi_graph,
)
from aurey.graphs.chains import chain_name_for_id
from aurey.graphs.read import ReadGraphInput
from aurey.graphs.swap_diag import SWAP_LOG, log_swap_tool
from aurey.runtime import AureyRuntime
from aurey.tools.user_input import RequestUserInputArgs, UserQuestion, note_user_input_request


def _graph_payload(state: dict[str, Any]) -> dict[str, Any]:
    err = state.get("error")
    res = state.get("result")
    if err is not None:
        return {"ok": False, "error": err}
    return {"ok": True, "result": res}


def _parse_chain_id_field(raw: Any) -> int | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        if isinstance(raw, int):
            return raw if raw >= 0 else None
        if isinstance(raw, str):
            s = raw.strip()
            return int(s, 16) if s.startswith(("0x", "0X")) else int(s, 10)
    except (TypeError, ValueError):
        return None
    return None


def _try_coerce_lifi_prepared_to_execute_envelope(
    envelope: dict[str, Any],
    prepare_lifi_g: Any,
) -> dict[str, Any] | None:
    """Build execute envelope when ``prepared`` (route + tx) was passed instead."""

    if envelope.get("kind"):
        return None
    rid = envelope.get("route_id") or envelope.get("routeId")
    tx_req = envelope.get("transaction_request") or envelope.get("transactionRequest")
    if not rid or not isinstance(tx_req, dict):
        return None

    cid = _parse_chain_id_field(tx_req.get("chainId"))
    if cid is None:
        return None
    chain = chain_name_for_id(cid)
    if chain is None:
        return None
    from_raw = tx_req.get("from")
    if from_raw is None or str(from_raw).strip() == "":
        return None

    payload = TxPrepareLiFiInput(
        chain=chain,
        from_address=str(from_raw).strip(),
        prepared={"route_id": str(rid), "transaction_request": dict(tx_req)},
    )
    state = prepare_lifi_g.invoke(
        {"input": payload.model_dump(mode="json", exclude_none=True)}
    )
    err = state.get("error")
    res = state.get("result")
    if err is not None or not isinstance(res, dict):
        return None
    fixed = res.get("envelope")
    return fixed if isinstance(fixed, dict) else None


def _data_selector(data_text: str) -> str | None:
    if data_text.startswith("0x") and len(data_text) >= 10:
        return data_text[:10]
    return None


def _data_bytes(data_text: str) -> int | None:
    if not data_text.startswith("0x"):
        return None
    return max((len(data_text) - 2) // 2, 0)


def _tx_request_summary(
    tx_req: dict[str, Any],
    *,
    route_id: str,
    prepared_id: str,
) -> dict[str, Any]:
    data = tx_req.get("data")
    data_text = data.strip() if isinstance(data, str) else ""
    chain_id = _parse_chain_id_field(tx_req.get("chainId"))
    return {
        "route_id": route_id,
        "prepared_id": prepared_id,
        "chain_id": chain_id,
        "from": tx_req.get("from"),
        "to": tx_req.get("to"),
        "value": tx_req.get("value"),
        "gas_limit": tx_req.get("gasLimit") or tx_req.get("gas"),
        "data_selector": _data_selector(data_text),
        "data_bytes": _data_bytes(data_text),
    }


def _envelope_summary(
    envelope: dict[str, Any],
    *,
    prepared_id: str | None = None,
) -> dict[str, Any]:
    data = envelope.get("data")
    data_text = data.strip() if isinstance(data, str) else ""
    out: dict[str, Any] = {
        "kind": envelope.get("kind"),
        "chain_id": envelope.get("chain_id"),
        "from_address": envelope.get("from_address"),
        "to": envelope.get("to"),
        "value_hex": envelope.get("value_hex"),
        "gas_limit_hex": envelope.get("gas_limit_hex"),
        "data_selector": _data_selector(data_text),
        "data_bytes": _data_bytes(data_text),
    }
    if prepared_id is not None:
        out["prepared_id"] = prepared_id
    return out


def _invalid_prepared_id(prepared_id: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "code": "invalid_input",
            "message": "Prepared transaction id was not found or has expired.",
            "details": {"prepared_id": prepared_id},
        },
    }


class AlchemyTokenPricesArgs(BaseModel):
    """Token prices quoted by contract address."""

    chain: str = Field(min_length=1)
    wallet_address: str = Field(min_length=1)
    token_addresses: list[str] = Field(min_length=1)


class AlchemyPortfolioArgs(BaseModel):
    """Portfolio token holdings for a wallet."""

    chain: str = Field(min_length=1)
    wallet_address: str = Field(min_length=1)


class AlchemyTransferHistoryArgs(BaseModel):
    """Transfers via ``alchemy_getAssetTransfers``."""

    chain: str = Field(min_length=1)
    wallet_address: str = Field(min_length=1)


class TxExecuteToolArgs(BaseModel):
    """Simulate/policy/sign/broadcast for a typed prepare envelope."""

    model_config = ConfigDict(extra="ignore")

    envelope: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Required. The exact `envelope` object from a successful `tx_prepare_*` or "
            "`tx_prepare_lifi_swap` call: `prepare_output['result']['envelope']`. "
            "You cannot call this tool with only `idempotency_key`."
        ),
    )
    prepared_id: str | None = Field(
        default=None,
        description=(
            "Preferred for LiFi swaps: short server-side prepared transaction id returned by "
            "`swap_prepare` or `tx_prepare_lifi_swap`. Avoids sending large calldata through the "
            "model."
        ),
    )
    idempotency_key: str | None = Field(
        default=None,
        description="Optional idempotency key for broadcast (never pass without `envelope`).",
    )

    @model_validator(mode="after")
    def _tx_reference_required(self) -> Self:
        if not self.envelope and not self.prepared_id:
            raise ValueError(
                "tx_execute requires `prepared_id` (preferred for LiFi swaps) or `envelope`: use "
                "`prepared_id` from `swap_prepare`/`tx_prepare_lifi_swap`, or the exact dict from "
                "a successful prepare tool at `result['envelope']`. `idempotency_key` alone is "
                "invalid."
            )
        return self


class TxPrepareNativeArgs(BaseModel):
    """Public args for native transfer preparation; tool name supplies the kind."""

    chain: str = Field(min_length=1)
    from_address: str = Field(min_length=1)
    to_address: str = Field(min_length=1)
    value_wei: int = Field(ge=0)


_ERC20_AMOUNT_FIELD = Field(
    ge=0,
    description=(
        "Token amount in the contract's smallest units (not necessarily 1e18). "
        "Lookup token decimals: USDC on Base/Ethereum uses **6** — e.g. 0.01 USDC = **10_000**, "
        "1 USDC = **1_000_000**. WETH uses 18. Do not scale USDC by 10**18."
    ),
)


class TxPrepareErc20TransferArgs(BaseModel):
    """Public args for ERC-20 transfer preparation; tool name supplies the kind."""

    chain: str = Field(min_length=1)
    from_address: str = Field(min_length=1)
    token_address: str = Field(min_length=1)
    to_address: str = Field(min_length=1)
    amount_wei: int = _ERC20_AMOUNT_FIELD


class TxPrepareErc20ApprovalArgs(BaseModel):
    """Public args for ERC-20 approval preparation; tool name supplies the kind."""

    chain: str = Field(min_length=1)
    from_address: str = Field(min_length=1)
    token_address: str = Field(min_length=1)
    spender_address: str = Field(min_length=1)
    amount_wei: int = _ERC20_AMOUNT_FIELD


class EvmGetNativeBalanceArgs(BaseModel):
    """Native gas-token balance via JSON-RPC (chain + wallet)."""

    chain: str = Field(min_length=1)
    wallet_address: str = Field(min_length=1)


class ResolveKnownAddressArgs(BaseModel):
    """Resolve a bundled ticker (e.g. USDC, WETH) to a contract address without RPC."""

    chain: str = Field(min_length=1)
    known_ticker: str = Field(min_length=1, description="Ticker key, e.g. usdc, weth.")


class EvmGetErc20BalanceArgs(BaseModel):
    """ERC-20 balance read (Mercury-parity stub until full token address wiring)."""

    chain: str = Field(min_length=1)
    wallet_address: str = Field(min_length=1)
    token_address: str = Field(min_length=1)


class EvmGetErc20DecimalsArgs(BaseModel):
    """Read token ``decimals()`` via deterministic ``eth_call``."""

    chain: str = Field(min_length=1)
    token_address: str = Field(min_length=1)


class EvmResolveEnsArgs(BaseModel):
    """Resolve an ENS name to an Ethereum checksum-normalized hex address."""

    name: str = Field(
        min_length=1,
        description="ENS name such as nick.eth (trimmed and lower-cased by the tool).",
    )
    chain: str = Field(
        default="ethereum",
        min_length=1,
        description="Must be 'ethereum'. ENS forward resolution is only defined on L1 mainnet.",
    )


def build_aurey_subgraph_tools(runtime: AureyRuntime) -> list[BaseTool]:
    """Compile subgraphs once and expose strict LangChain tools (validated graph inputs only)."""

    read_g = build_read_graph(runtime)
    alchemy_g = build_alchemy_graph(runtime)
    swap_g = build_swap_prepare_graph(runtime)
    prepare_g = build_tx_prepare_graph(runtime)
    prepare_lifi_g = build_tx_prepare_lifi_graph(runtime)
    execute_g = build_tx_execute_graph(runtime)

    @tool(args_schema=EvmGetNativeBalanceArgs)
    def evm_get_native_balance(chain: str, wallet_address: str) -> dict[str, Any]:
        """Return native token balance (wei hex) for wallet on chain via configured RPC path."""
        payload = EvmGetNativeBalanceArgs(chain=chain, wallet_address=wallet_address)
        graph_in = ReadGraphInput(
            operation="native_balance",
            chain=payload.chain,
            wallet_address=payload.wallet_address,
        )
        return _graph_payload(read_g.invoke({"input": graph_in.model_dump()}))

    @tool(args_schema=ResolveKnownAddressArgs)
    def resolve_known_address(chain: str, known_ticker: str) -> dict[str, Any]:
        """Resolve ticker to address + name using bundled ``known_addresses.json``."""
        payload = ResolveKnownAddressArgs(chain=chain, known_ticker=known_ticker)
        graph_in = ReadGraphInput(
            operation="known_address",
            chain=payload.chain,
            known_ticker=payload.known_ticker,
        )
        return _graph_payload(read_g.invoke({"input": graph_in.model_dump()}))

    @tool(args_schema=EvmGetErc20BalanceArgs)
    def evm_get_erc20_balance(
        chain: str,
        wallet_address: str,
        token_address: str,
    ) -> dict[str, Any]:
        """ERC-20 balance placeholder result (validates wallet and token addresses)."""
        payload = EvmGetErc20BalanceArgs(
            chain=chain,
            wallet_address=wallet_address,
            token_address=token_address,
        )
        graph_in = ReadGraphInput(
            operation="erc20_balance",
            chain=payload.chain,
            wallet_address=payload.wallet_address,
            token_address=payload.token_address,
        )
        return _graph_payload(read_g.invoke({"input": graph_in.model_dump()}))

    @tool(args_schema=EvmGetErc20DecimalsArgs)
    def evm_get_erc20_decimals(chain: str, token_address: str) -> dict[str, Any]:
        """Return ERC-20 decimals from the token contract's decimals() view (eth_call)."""
        payload = EvmGetErc20DecimalsArgs(chain=chain, token_address=token_address)
        graph_in = ReadGraphInput(
            operation="erc20_decimals",
            chain=payload.chain,
            token_address=payload.token_address,
        )
        return _graph_payload(read_g.invoke({"input": graph_in.model_dump()}))

    @tool(args_schema=EvmResolveEnsArgs)
    def evm_resolve_ens(name: str, chain: str = "ethereum") -> dict[str, Any]:
        """Resolve ENS on **ethereum mainnet** to a ``0x`` address (registry + resolver).

        Call **before** ``tx_prepare_*`` / ``swap_prepare`` when the user gives an ENS name as
        ``to_address`` or recipient; use ``result['resolved_address']``. Other chains reject.
        """
        payload = EvmResolveEnsArgs(name=name, chain=chain)
        graph_in = ReadGraphInput(
            operation="ens_resolve",
            chain=payload.chain,
            ens_name=payload.name,
        )
        return _graph_payload(read_g.invoke({"input": graph_in.model_dump()}))

    tools: list[BaseTool] = [
        evm_get_native_balance,
        evm_get_erc20_decimals,
        evm_resolve_ens,
        resolve_known_address,
        evm_get_erc20_balance,
    ]

    @tool(args_schema=AlchemyTokenPricesArgs)
    def alchemy_get_token_prices(
        chain: str,
        wallet_address: str,
        token_addresses: list[str],
    ) -> dict[str, Any]:
        """Fetch quoted prices by token contract address."""
        payload = AlchemyTokenPricesArgs(
            chain=chain,
            wallet_address=wallet_address,
            token_addresses=list(token_addresses),
        )
        state = alchemy_g.invoke(
            {"input": {**payload.model_dump(), "operation": "token_prices"}},
        )
        return _graph_payload(state)

    tools.append(alchemy_get_token_prices)

    @tool(args_schema=AlchemyPortfolioArgs)
    def alchemy_get_portfolio_tokens(
        chain: str,
        wallet_address: str,
    ) -> dict[str, Any]:
        """Portfolio balances for wallet on chain."""
        state = alchemy_g.invoke(
            {
                "input": {
                    "operation": "portfolio_tokens",
                    "chain": chain,
                    "wallet_address": wallet_address,
                }
            }
        )
        return _graph_payload(state)

    tools.append(alchemy_get_portfolio_tokens)

    @tool(args_schema=AlchemyTransferHistoryArgs)
    def alchemy_get_transfer_history(
        chain: str,
        wallet_address: str,
    ) -> dict[str, Any]:
        """Recent external and ERC-20 transfers for wallet via Alchemy."""
        state = alchemy_g.invoke(
            {
                "input": {
                    "operation": "transfer_history",
                    "chain": chain,
                    "wallet_address": wallet_address,
                }
            }
        )
        return _graph_payload(state)

    tools.append(alchemy_get_transfer_history)

    @tool(args_schema=SwapPrepareInput)
    def swap_prepare(
        from_chain: str,
        to_chain: str,
        from_asset: str,
        to_asset: str,
        from_amount_wei: str,
        from_address: str,
        to_address: str,
        slippage: float | None = None,
        order: Literal["FASTEST", "CHEAPEST"] | None = None,
    ) -> dict[str, Any]:
        """LiFi swap quote + next unsigned tx request (`transaction_request`).

        Uses LiFi ``GET /v1/quote`` (see https://docs.li.fi/llms.txt). Optional ``slippage`` is
        a decimal fraction (e.g. ``0.005`` = 0.5%%). Optional ``order`` is ``FASTEST`` or
        ``CHEAPEST``. Prefer **checksum** ``0x`` token addresses when possible.

        On success, call ``tx_execute(prepared_id=result['prepared_id'])``. The full LiFi
        transaction request is stored server-side so the model does not need to copy calldata.

        When ``result`` includes ``allowance``, the wallet must approve the spender for the
        sell token before the swap simulates (unless allowance was already sufficient—then
        ``swap_prepare`` omits ``allowance`` when Alchemy is configured). Use
        ``tx_prepare_erc20_approval`` then ``tx_execute`` that tx first (see system rules).

        If ``to_address`` (or ``from_address``) is an ENS name like ``alice.eth``, run
        ``evm_resolve_ens`` on **ethereum** first and pass the returned hex address here.
        """
        payload = SwapPrepareInput(
            from_chain=from_chain,
            to_chain=to_chain,
            from_asset=from_asset,
            to_asset=to_asset,
            from_amount_wei=from_amount_wei,
            from_address=from_address,
            to_address=to_address,
            slippage=slippage,
            order=order,
        )
        t0 = time.perf_counter()
        out = _graph_payload(swap_g.invoke({"input": payload.model_dump()}))
        rid = None
        if out.get("ok") and isinstance(out.get("result"), dict):
            prep = out["result"].get("prepared")
            if isinstance(prep, dict):
                rid = prep.get("route_id")
                tx_req = prep.get("transaction_request") or prep.get("transactionRequest")
                if isinstance(rid, str) and isinstance(tx_req, dict):
                    lifi_prepared_id = runtime.prepared_txs.put(
                        kind="lifi_prepared",
                        payload=prep,
                        summary=_tx_request_summary(tx_req, route_id=rid, prepared_id=""),
                    )
                    prepared_state = prepare_lifi_g.invoke(
                        {
                            "input": {
                                "chain": from_chain,
                                "from_address": from_address,
                                "prepared": prep,
                            }
                        }
                    )
                    err = prepared_state.get("error")
                    res = prepared_state.get("result")
                    if err is not None or not isinstance(res, dict):
                        out = {"ok": False, "error": err or {"code": "invalid_input"}}
                    else:
                        envelope = res.get("envelope")
                        if isinstance(envelope, dict):
                            prepared_id = runtime.prepared_txs.put(
                                kind="execute_envelope",
                                payload=envelope,
                                summary=_envelope_summary(envelope),
                            )
                            compact = _tx_request_summary(
                                tx_req,
                                route_id=rid,
                                prepared_id=prepared_id,
                            )
                            compact["execute_prepared_id"] = prepared_id
                            compact["lifi_prepared_id"] = lifi_prepared_id
                            out["result"]["prepared"] = compact
                            out["result"]["prepared_id"] = prepared_id
        log_swap_tool(
            name="swap_prepare",
            wall_ms=(time.perf_counter() - t0) * 1000,
            ok=out.get("ok"),
            route_id=rid,
            from_chain=from_chain,
            to_chain=to_chain,
        )
        return out

    tools.append(swap_prepare)

    @tool(args_schema=TxPrepareLiFiInput)
    def tx_prepare_lifi_swap(
        chain: str,
        from_address: str,
        prepared: dict[str, Any] | None = None,
        prepared_id: str | None = None,
        route_id: str | None = None,
        transaction_request: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert ``swap_prepare`` output into a ``tx_execute`` envelope.

        Prefer ``prepared_id`` from ``swap_prepare``; it keeps large calldata out of the model
        context. Legacy callers may still pass ``prepared`` verbatim or ``route_id`` plus
        ``transaction_request``. On success, call ``tx_execute(prepared_id=result['prepared_id'])``.
        Resolve ENS names on ethereum with ``evm_resolve_ens`` before supplying ``from_address``.
        """
        if prepared_id:
            record = runtime.prepared_txs.get(prepared_id)
            if record is None:
                return _invalid_prepared_id(prepared_id)
            if record.kind == "execute_envelope":
                summary = _envelope_summary(record.payload, prepared_id=prepared_id)
                return {"ok": True, "result": {"prepared_id": prepared_id, "envelope": summary}}
            if record.kind == "lifi_prepared":
                prepared = dict(record.payload)

        payload = TxPrepareLiFiInput(
            chain=chain,
            from_address=from_address,
            prepared=prepared,
            prepared_id=prepared_id,
            route_id=route_id,
            transaction_request=transaction_request,
        )
        t0 = time.perf_counter()
        out = _graph_payload(
            prepare_lifi_g.invoke({"input": payload.model_dump(mode="json", exclude_none=True)})
        )
        rid = route_id
        if rid is None and isinstance(prepared, dict):
            rid = prepared.get("route_id")
        if out.get("ok") and isinstance(out.get("result"), dict):
            envelope = out["result"].get("envelope")
            if isinstance(envelope, dict):
                stored_id = runtime.prepared_txs.put(
                    kind="execute_envelope",
                    payload=envelope,
                    summary=_envelope_summary(envelope),
                )
                out["result"] = {
                    "prepared_id": stored_id,
                    "envelope": _envelope_summary(envelope, prepared_id=stored_id),
                }
        log_swap_tool(
            name="tx_prepare_lifi_swap",
            wall_ms=(time.perf_counter() - t0) * 1000,
            ok=out.get("ok"),
            route_id=rid,
            chain=chain,
        )
        return out

    tools.append(tx_prepare_lifi_swap)

    @tool(args_schema=TxPrepareNativeArgs)
    def tx_prepare_native_transfer(
        chain: str,
        from_address: str,
        to_address: str,
        value_wei: int,
    ) -> dict[str, Any]:
        """Prepare native gas-token transfer envelope (signing path only, no key material).

        On success (`ok` true), broadcast with `tx_execute(envelope=result['envelope'])` using that
        dict verbatim. If ``to_address`` is an ENS name, call ``evm_resolve_ens`` first on
        ethereum and use ``resolved_address`` as ``to_address``.
        """
        payload = TxPrepareNative(
            chain=chain,
            from_address=from_address,
            to_address=to_address,
            value_wei=value_wei,
        )
        return _graph_payload(prepare_g.invoke({"input": payload.model_dump()}))

    tools.append(tx_prepare_native_transfer)

    @tool(args_schema=TxPrepareErc20TransferArgs)
    def tx_prepare_erc20_transfer(
        chain: str,
        from_address: str,
        token_address: str,
        to_address: str,
        amount_wei: int,
    ) -> dict[str, Any]:
        """Prepare ERC-20 transfer envelope.

        `amount_wei` is misleadingly named: use the token's **native decimals** (raw integer),
        not ETH wei. USDC = 6 decimals. On success (`ok` true), call
        `tx_execute(envelope=result['envelope'])` with the returned envelope unchanged.
        Never call `tx_execute` without `envelope`. Resolve ENS recipients with
        ``evm_resolve_ens`` (ethereum) before passing ``to_address``.
        """
        payload = TxPrepareErc20Transfer(
            chain=chain,
            from_address=from_address,
            token_address=token_address,
            to_address=to_address,
            amount_wei=amount_wei,
        )
        return _graph_payload(prepare_g.invoke({"input": payload.model_dump()}))

    tools.append(tx_prepare_erc20_transfer)

    @tool(args_schema=TxPrepareErc20ApprovalArgs)
    def tx_prepare_erc20_approval(
        chain: str,
        from_address: str,
        token_address: str,
        spender_address: str,
        amount_wei: int,
    ) -> dict[str, Any]:
        """Prepare ERC-20 approval envelope.

        `amount_wei` must be raw token units per token decimals (USDC: 6). On success (`ok` true),
        broadcast with `tx_execute(envelope=result['envelope'])` using that dict verbatim.
        """
        payload = TxPrepareErc20Approval(
            chain=chain,
            from_address=from_address,
            token_address=token_address,
            spender_address=spender_address,
            amount_wei=amount_wei,
        )
        return _graph_payload(prepare_g.invoke({"input": payload.model_dump()}))

    tools.append(tx_prepare_erc20_approval)

    @tool(args_schema=TxExecuteToolArgs)
    def tx_execute(
        envelope: dict[str, Any] | None = None,
        prepared_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Run simulate/policy/sign/broadcast for a prepared transaction envelope.

        Prefer ``prepared_id`` from ``swap_prepare`` or ``tx_prepare_lifi_swap`` for LiFi swaps.
        Legacy callers may pass the exact ``result['envelope']`` dict from a successful
        ``tx_prepare_*`` tool. If you mistakenly pass ``swap_prepare``'s legacy ``prepared`` object
        (``route_id`` + ``transaction_request`` only), this tool attempts to repair it.
        """
        if not prepared_id and isinstance(envelope, dict) and envelope.get("prepared_id"):
            prepared_id = str(envelope["prepared_id"])

        if prepared_id:
            record = runtime.prepared_txs.get(prepared_id)
            if record is None:
                return _invalid_prepared_id(prepared_id)
            if record.kind == "execute_envelope":
                envelope = dict(record.payload)
            elif record.kind == "lifi_prepared":
                fixed = _try_coerce_lifi_prepared_to_execute_envelope(
                    dict(record.payload),
                    prepare_lifi_g,
                )
                if fixed is None:
                    return _invalid_prepared_id(prepared_id)
                envelope = fixed

        if isinstance(envelope, dict):
            fixed = _try_coerce_lifi_prepared_to_execute_envelope(envelope, prepare_lifi_g)
            if fixed is not None:
                SWAP_LOG.info(
                    "tx_execute auto-prepared LiFi envelope from mistaken prepared blob "
                    "route_id=%s",
                    envelope.get("route_id") or envelope.get("routeId"),
                )
                envelope = fixed

        root = TxExecuteToolArgs(
            envelope=envelope,
            prepared_id=prepared_id,
            idempotency_key=idempotency_key,
        )
        execute_in = TxExecuteInput.model_validate(root.model_dump()).model_dump()
        t0 = time.perf_counter()
        out = _graph_payload(execute_g.invoke({"input": execute_in}))
        kind = envelope.get("kind") if isinstance(envelope, dict) else None
        th = None
        if out.get("ok") and isinstance(out.get("result"), dict):
            th = out["result"].get("tx_hash")
        log_swap_tool(
            name="tx_execute",
            wall_ms=(time.perf_counter() - t0) * 1000,
            ok=out.get("ok"),
            kind=kind,
            tx_hash=th,
        )
        return out

    tools.append(tx_execute)

    @tool(args_schema=RequestUserInputArgs)
    def request_user_input(questions: list[UserQuestion]) -> dict[str, Any]:
        """Queue concise follow-ups for missing user/host context (blocking only)."""
        count = note_user_input_request(questions)
        return {"ok": True, "result": {"status": "needs_user_input", "question_count": count}}

    tools.append(request_user_input)

    return tools


__all__ = [
    "AlchemyPortfolioArgs",
    "AlchemyTokenPricesArgs",
    "AlchemyTransferHistoryArgs",
    "EvmGetErc20BalanceArgs",
    "EvmGetNativeBalanceArgs",
    "EvmResolveEnsArgs",
    "ResolveKnownAddressArgs",
    "SwapPrepareInput",
    "TxPrepareErc20ApprovalArgs",
    "TxPrepareErc20TransferArgs",
    "TxPrepareNativeArgs",
    "TxPrepareErc20Approval",
    "TxPrepareErc20Transfer",
    "TxPrepareNative",
    "TxExecuteToolArgs",
    "build_aurey_subgraph_tools",
]
