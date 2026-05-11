"""LangChain tools wrapping compiled Aurey LangGraph subgraph invocations."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from aurey.graphs import (
    SwapPrepareInput,
    TxExecuteInput,
    TxPrepareErc20Approval,
    TxPrepareErc20Transfer,
    TxPrepareNative,
    build_alchemy_graph,
    build_read_graph,
    build_swap_prepare_graph,
    build_tx_execute_graph,
    build_tx_prepare_graph,
)
from aurey.graphs.read import ReadGraphInput
from aurey.runtime import AureyRuntime
from aurey.tools.user_input import RequestUserInputArgs, UserQuestion, note_user_input_request


def _graph_payload(state: dict[str, Any]) -> dict[str, Any]:
    err = state.get("error")
    res = state.get("result")
    if err is not None:
        return {"ok": False, "error": err}
    return {"ok": True, "result": res}


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

    envelope: dict[str, Any] = Field(
        ...,
        description=(
            "Required. The exact `envelope` object from a successful `tx_prepare_*` call: "
            "`prepare_output['result']['envelope']`. "
            "Pass the dict unchanged (do not omit this field)."
        ),
    )
    idempotency_key: str | None = Field(
        default=None,
        description="Optional idempotency key for the execute/broadcast pipeline.",
    )


class TxPrepareNativeArgs(BaseModel):
    """Public args for native transfer preparation; tool name supplies the kind."""

    chain: str = Field(min_length=1)
    from_address: str = Field(min_length=1)
    to_address: str = Field(min_length=1)
    value_wei: int = Field(ge=0)


class TxPrepareErc20TransferArgs(BaseModel):
    """Public args for ERC-20 transfer preparation; tool name supplies the kind."""

    chain: str = Field(min_length=1)
    from_address: str = Field(min_length=1)
    token_address: str = Field(min_length=1)
    to_address: str = Field(min_length=1)
    amount_wei: int = Field(ge=0)


class TxPrepareErc20ApprovalArgs(BaseModel):
    """Public args for ERC-20 approval preparation; tool name supplies the kind."""

    chain: str = Field(min_length=1)
    from_address: str = Field(min_length=1)
    token_address: str = Field(min_length=1)
    spender_address: str = Field(min_length=1)
    amount_wei: int = Field(ge=0)


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


def build_aurey_subgraph_tools(runtime: AureyRuntime) -> list[BaseTool]:
    """Compile subgraphs once and expose strict LangChain tools (validated graph inputs only)."""

    read_g = build_read_graph(runtime)
    alchemy_g = build_alchemy_graph(runtime)
    swap_g = build_swap_prepare_graph(runtime)
    prepare_g = build_tx_prepare_graph(runtime)
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
        """Map a known ticker to an on-chain contract address (no RPC round-trip)."""
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

    tools: list[BaseTool] = [
        evm_get_native_balance,
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
    ) -> dict[str, Any]:
        """LiFi swap quote + next unsigned tx request."""
        payload = SwapPrepareInput(
            from_chain=from_chain,
            to_chain=to_chain,
            from_asset=from_asset,
            to_asset=to_asset,
            from_amount_wei=from_amount_wei,
            from_address=from_address,
            to_address=to_address,
        )
        return _graph_payload(swap_g.invoke({"input": payload.model_dump()}))

    tools.append(swap_prepare)

    @tool(args_schema=TxPrepareNativeArgs)
    def tx_prepare_native_transfer(
        chain: str,
        from_address: str,
        to_address: str,
        value_wei: int,
    ) -> dict[str, Any]:
        """Prepare native gas-token transfer envelope (signing path only, no key material).

        On success (`ok` true), broadcast with `tx_execute(envelope=result['envelope'])` using that
        dict verbatim.
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

        On success (`ok` true), call `tx_execute(envelope=result['envelope'])` with the returned
        envelope object unchanged. Never call `tx_execute` without `envelope`.
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

        On success (`ok` true), broadcast with `tx_execute(envelope=result['envelope'])` using that
        dict verbatim.
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
        envelope: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Run simulate/policy/sign/broadcast for a prepared transaction envelope.

        You MUST pass `envelope`: the exact `result.envelope` dict from the latest successful
        `tx_prepare_native_transfer`, `tx_prepare_erc20_transfer`, or `tx_prepare_erc20_approval`
        tool output. Omitting `envelope` is invalid.
        """
        root = TxExecuteToolArgs(envelope=envelope, idempotency_key=idempotency_key)
        execute_in = TxExecuteInput.model_validate(root.model_dump()).model_dump()
        return _graph_payload(execute_g.invoke({"input": execute_in}))

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
