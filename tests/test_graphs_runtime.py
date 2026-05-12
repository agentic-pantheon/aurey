"""Runtime + LangGraph coverage (fake SecretStore and scripted clients)."""

from __future__ import annotations

import json
from typing import Any

import ormsgpack

from aurey.custody import FakeSecretStore
from aurey.graphs import (
    DeterministicTxPipeline,
    build_alchemy_graph,
    build_read_graph,
    build_swap_prepare_graph,
    build_tx_execute_graph,
    build_tx_prepare_graph,
    build_tx_prepare_lifi_graph,
)
from aurey.graphs.ens_eth import (
    ENS_REGISTRY_MAINNET,
    ens_addr_calldata,
    ens_namehash,
    ens_resolver_calldata,
)
from aurey.graphs.ports import HttpJsonPort, HttpJsonRequestError
from aurey.runtime import AureyRuntime
from aurey.settings import AureySettings
from tests.fakes.evm_rpc import rpc_factory_from_mapping
from tests.fakes.http_client import ScriptedHttpClient


class _LifiUnauthorizedHttp(HttpJsonPort):
    def request_json(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | list[Any] | None = None,
    ) -> dict[str, Any]:
        _ = method, url, headers, json_body
        raise HttpJsonRequestError(
            status_code=401,
            body_text='{"message":"Invalid API key","code":1010}',
            payload={"message": "Invalid API key", "code": 1010},
        )


def _runtime(
    *,
    secrets: dict[str, str],
    settings: AureySettings,
    http: HttpJsonPort,
    rpc_map: dict[str, object],
) -> AureyRuntime:
    return AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping(rpc_map),
        http=http,
        tx_pipeline=DeterministicTxPipeline(),
        lifi_base_url="https://li.quest",
    )


def _ban_values() -> tuple[str, ...]:
    return (
        "INJECTED_RPC_URL_SECRET_FRAGMENT",
        "INJECTED_ALCHEMY_KEY_AAA",
        "INJECTED_LIFI_KEY_BBB",
        "0x" + "ff" * 32,
    )


def _assert_no_banned_values(payload: object) -> None:
    blob = json.dumps(payload, default=str, sort_keys=True)
    for fragment in _ban_values():
        assert fragment not in blob


def test_read_native_balance_graph():
    alchemy_path = "vault/alchemy/1"
    signing_path = "vault/signing/local"
    secrets = {
        alchemy_path: "INJECTED_ALCHEMY_KEY_AAA",
        "vault/lifi/1": "INJECTED_LIFI_KEY_BBB",
        signing_path: "0x" + "ff" * 32,
    }
    settings = AureySettings(
        alchemy_api_secret_path=alchemy_path,
        lifi_api_secret_path="vault/lifi/1",
        wallet_signing_key_secret_path=signing_path,
    )
    http = ScriptedHttpClient()
    rpc_urls: list[str] = []

    def rpc_factory(url: str):
        rpc_urls.append(url)
        return rpc_factory_from_mapping({"eth_getBalance": "0x10"})(url)

    runtime = _runtime(
        secrets=secrets,
        settings=settings,
        http=http,
        rpc_map={},
    )
    runtime = AureyRuntime(
        settings=runtime.settings,
        secret_store=runtime.secret_store,
        evm_rpc_factory=rpc_factory,
        http=runtime.http,
        tx_pipeline=runtime.tx_pipeline,
        lifi_base_url=runtime.lifi_base_url,
    )
    graph = build_read_graph(runtime)
    out = graph.invoke(
        {
            "input": {
                "operation": "native_balance",
                "chain": "ethereum",
                "wallet_address": "0x0000000000000000000000000000000000000001",
            }
        }
    )
    assert out.get("error") is None
    assert out["result"]["balance_wei_hex"] == "0x10"
    assert out["result"]["balance_wei"] == 16
    assert out["result"]["balance_eth"] == "0.000000000000000016"
    assert rpc_urls == ["https://eth-mainnet.g.alchemy.com/v2/INJECTED_ALCHEMY_KEY_AAA"]
    _assert_no_banned_values(out)


def test_read_erc20_decimals_graph():
    alchemy_path = "vault/alchemy/x"
    secrets = {alchemy_path: "INJECTED_ALCHEMY_KEY_AAA"}
    settings = AureySettings(alchemy_api_secret_path=alchemy_path)

    def eth_call(params: list) -> str:
        assert params[0]["data"] == "0x313ce567"
        assert params[1] == "latest"
        return "0x0000000000000000000000000000000000000000000000000000000000000006"

    runtime = AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping({"eth_call": eth_call}),
        http=ScriptedHttpClient(),
        tx_pipeline=DeterministicTxPipeline(),
        lifi_base_url="https://li.quest",
    )
    graph = build_read_graph(runtime)
    tok = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
    out = graph.invoke(
        {
            "input": {
                "operation": "erc20_decimals",
                "chain": "base",
                "token_address": tok,
            }
        }
    )
    assert out.get("error") is None
    assert out["result"]["decimals"] == 6
    assert out["result"]["chain_id"] == 8453
    assert out["result"]["token_address"] == tok.lower()
    _assert_no_banned_values(out)


def test_read_ens_resolve_graph():
    alchemy_path = "vault/alchemy/y"
    secrets = {alchemy_path: "INJECTED_ALCHEMY_KEY_AAA"}
    settings = AureySettings(alchemy_api_secret_path=alchemy_path)

    ens_name = "foo.eth"
    node = ens_namehash(ens_name)
    resolver_addr = "0x2222222222222222222222222222222222222222"
    resolved_wallet = "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"
    padded_resolver = "0x" + "0" * 24 + resolver_addr[2:]
    padded_wallet = "0x" + "0" * 24 + resolved_wallet[2:]
    expected_registry = ENS_REGISTRY_MAINNET.lower()
    resolver_l = resolver_addr.lower()

    def eth_call(params: list) -> str:
        body = params[0]
        to_l = body["to"].lower()
        data = body["data"].lower()
        if to_l == expected_registry:
            assert data == ens_resolver_calldata(node).lower()
            return padded_resolver
        if to_l == resolver_l:
            assert data == ens_addr_calldata(node).lower()
            return padded_wallet
        raise AssertionError((to_l, data[:10]))

    runtime = AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping({"eth_call": eth_call}),
        http=ScriptedHttpClient(),
        tx_pipeline=DeterministicTxPipeline(),
        lifi_base_url="https://li.quest",
    )
    graph = build_read_graph(runtime)
    out = graph.invoke(
        {
            "input": {
                "operation": "ens_resolve",
                "chain": "ethereum",
                "ens_name": "  FOO.ETH ",
            },
        }
    )
    assert out.get("error") is None
    assert out["result"]["name"] == ens_name
    assert out["result"]["resolved_address"] == resolved_wallet.lower()
    assert out["result"]["chain"] == "ethereum"
    assert out["result"]["chain_id"] == 1
    _assert_no_banned_values(out)


def test_read_ens_resolve_unsupported_chain():
    alchemy_path = "vault/alchemy/z"
    secrets = {alchemy_path: "INJECTED_ALCHEMY_KEY_AAA"}
    settings = AureySettings(alchemy_api_secret_path=alchemy_path)
    runtime = AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping({}),
        http=ScriptedHttpClient(),
        tx_pipeline=DeterministicTxPipeline(),
        lifi_base_url="https://li.quest",
    )
    graph = build_read_graph(runtime)
    out = graph.invoke(
        {
            "input": {
                "operation": "ens_resolve",
                "chain": "base",
                "ens_name": "x.eth",
            },
        }
    )
    assert out.get("result") is None
    assert out["error"]["code"] == "unsupported_chain"


def test_read_ens_resolve_no_resolver_returns_ens_not_found():
    alchemy_path = "vault/alchemy/nf"
    secrets = {alchemy_path: "INJECTED_ALCHEMY_KEY_AAA"}
    settings = AureySettings(alchemy_api_secret_path=alchemy_path)
    padded_zero = "0x" + "0" * 64

    def eth_call(params: list) -> str:
        body = params[0]
        assert body["to"].lower() == ENS_REGISTRY_MAINNET.lower()
        assert body["data"].lower().startswith("0x0178b8bf")
        assert params[1] == "latest"
        return padded_zero

    runtime = AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping({"eth_call": eth_call}),
        http=ScriptedHttpClient(),
        tx_pipeline=DeterministicTxPipeline(),
        lifi_base_url="https://li.quest",
    )
    graph = build_read_graph(runtime)
    out = graph.invoke(
        {
            "input": {
                "operation": "ens_resolve",
                "chain": "Ethereum",
                "ens_name": "does-not-exist-12345.eth",
            },
        }
    )
    assert out["error"]["code"] == "ens_not_found"


def test_read_known_address_graph():
    secrets = {}
    settings = AureySettings()
    runtime = _runtime(
        secrets=secrets,
        settings=settings,
        http=ScriptedHttpClient(),
        rpc_map={},
    )
    graph = build_read_graph(runtime)
    out = graph.invoke(
        {"input": {"operation": "known_address", "chain": "ethereum", "known_ticker": "usdc"}}
    )
    assert out["result"]["resolved_address"] == "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
    assert out["result"]["symbol"] == "USDC"
    assert out["result"]["name"] == "USD Coin"
    _assert_no_banned_values(out)


def test_alchemy_token_prices_graph():
    secrets = {"vault/alchemy": "INJECTED_ALCHEMY_KEY_AAA"}
    settings = AureySettings(alchemy_api_secret_path="vault/alchemy")

    def match_prices(**kw: object) -> bool:
        if kw.get("method") != "POST":
            return False
        url = str(kw.get("url") or "")
        if "/prices/v1/" not in url or "tokens/by-address" not in url:
            return False
        body = kw.get("json_body") or {}
        return isinstance(body, dict) and isinstance(body.get("addresses"), list)

    http = ScriptedHttpClient(
        [
            (
                match_prices,
                {
                    "data": [
                        {
                            "network": "eth-mainnet",
                            "address": "0x2222222222222222222222222222222222222222",
                            "prices": [
                                {
                                    "currency": "USD",
                                    "value": "3.14",
                                    "lastUpdatedAt": "2025-01-01T00:00:00Z",
                                }
                            ],
                            "error": None,
                        }
                    ]
                },
            )
        ]
    )
    runtime = _runtime(
        secrets=secrets,
        settings=settings,
        http=http,
        rpc_map={},
    )
    graph = build_alchemy_graph(runtime)
    out = graph.invoke(
        {
            "input": {
                "operation": "token_prices",
                "chain": "ethereum",
                "wallet_address": "0x1111111111111111111111111111111111111111",
                "token_addresses": ["0x2222222222222222222222222222222222222222"],
            }
        }
    )
    addr = "0x2222222222222222222222222222222222222222"
    assert out["result"]["prices_by_address"][addr] == "3.14"
    _assert_no_banned_values(out)


def test_alchemy_portfolio_and_transfers_graphs():
    secrets = {"vault/alchemy": "INJECTED_ALCHEMY_KEY_AAA"}
    settings = AureySettings(alchemy_api_secret_path="vault/alchemy")
    wallet = "0x1111111111111111111111111111111111111111"

    def match_portfolio(**kw: object) -> bool:
        if kw.get("method") != "POST":
            return False
        u = str(kw.get("url") or "")
        return "/data/v1/" in u and "assets/tokens/by-address" in u

    def match_transfers_from(**kw: object) -> bool:
        body = kw.get("json_body") or {}
        params = body.get("params") if isinstance(body, dict) else None
        block = params[0] if isinstance(params, list) and params else {}
        return (
            kw.get("method") == "POST"
            and ".g.alchemy.com/v2/" in str(kw.get("url") or "")
            and isinstance(body, dict)
            and body.get("method") == "alchemy_getAssetTransfers"
            and isinstance(block, dict)
            and "fromAddress" in block
        )

    def match_transfers_to(**kw: object) -> bool:
        body = kw.get("json_body") or {}
        params = body.get("params") if isinstance(body, dict) else None
        block = params[0] if isinstance(params, list) and params else {}
        return (
            kw.get("method") == "POST"
            and ".g.alchemy.com/v2/" in str(kw.get("url") or "")
            and isinstance(body, dict)
            and body.get("method") == "alchemy_getAssetTransfers"
            and isinstance(block, dict)
            and "toAddress" in block
        )

    http = ScriptedHttpClient(
        [
            (
                match_portfolio,
                {
                    "data": {
                        "tokens": [
                            {
                                "address": wallet,
                                "network": "base-mainnet",
                                "tokenAddress": None,
                                "tokenBalance": "1000000000000000000",
                                "tokenMetadata": {
                                    "decimals": 18,
                                    "symbol": "ETH",
                                    "name": "Ether",
                                },
                            },
                            {
                                "address": wallet,
                                "network": "base-mainnet",
                                "tokenAddress": "0x2222222222222222222222222222222222222222",
                                "tokenBalance": (
                                    "0x000000000000000000000000000000000000000000000000"
                                    "0000000000000f569"
                                ),
                                "tokenMetadata": {
                                    "decimals": "6",
                                    "symbol": "USDC",
                                    "name": "USD Coin",
                                },
                            },
                            {
                                "address": wallet,
                                "network": "base-mainnet",
                                "tokenAddress": "0x3333333333333333333333333333333333333333",
                                "tokenBalance": hex(2**63),
                                "tokenMetadata": {
                                    "decimals": 18,
                                    "symbol": "HUGE",
                                    "name": "Int64 overflow balance fixture",
                                },
                            },
                        ]
                    }
                },
            ),
            (
                match_transfers_from,
                {"result": {"transfers": [{"uniqueId": "t-high", "blockNum": "0x10"}]}},
            ),
            (
                match_transfers_to,
                {"result": {"transfers": [{"uniqueId": "t-low", "blockNum": "0x5"}]}},
            ),
        ]
    )
    runtime = _runtime(secrets=secrets, settings=settings, http=http, rpc_map={})

    portfolio = build_alchemy_graph(runtime).invoke(
        {
            "input": {
                "operation": "portfolio_tokens",
                "chain": "base",
                "wallet_address": wallet,
            }
        }
    )
    assert portfolio["result"]["tokens"][0]["tokenMetadata"]["symbol"] == "ETH"
    assert portfolio["result"]["tokens"][0]["balance_raw"] == "1000000000000000000"
    assert portfolio["result"]["tokens"][0]["decimals"] == 18
    assert portfolio["result"]["tokens"][0]["balance_decimal"] == "1"
    assert portfolio["result"]["tokens"][1]["balance_raw"] == "62825"
    assert portfolio["result"]["tokens"][1]["decimals"] == 6
    assert portfolio["result"]["tokens"][1]["balance_decimal"] == "0.062825"
    assert portfolio["result"]["tokens"][2]["balance_raw"] == str(2**63)
    assert portfolio["result"]["tokens"][2]["decimals"] == 18
    ormsgpack.packb(portfolio["result"])
    _assert_no_banned_values(portfolio)

    transfers = build_alchemy_graph(runtime).invoke(
        {
            "input": {
                "operation": "transfer_history",
                "chain": "base",
                "wallet_address": wallet,
            }
        }
    )
    assert transfers["result"]["transfers"][0]["uniqueId"] == "t-high"
    assert {t["uniqueId"] for t in transfers["result"]["transfers"]} == {"t-high", "t-low"}
    _assert_no_banned_values(transfers)


def test_swap_prepare_graph():
    secrets = {"vault/lifi": "INJECTED_LIFI_KEY_BBB"}
    settings = AureySettings(lifi_api_secret_path="vault/lifi")

    def match_quote(**kw: object) -> bool:
        headers = kw.get("headers") or {}
        u = str(kw.get("url") or "")
        return (
            kw.get("method") == "GET"
            and "li.quest" in u
            and "/v1/quote?" in u
            and "fromChain=1" in u
            and "toChain=8453" in u
            and isinstance(headers, dict)
            and headers.get("x-lifi-api-key") == "INJECTED_LIFI_KEY_BBB"
            and "integrator=aurey" in u
        )

    http = ScriptedHttpClient(
        [
            (
                match_quote,
                {
                    "id": "swap-route-1",
                    "estimate": {
                        "approvalAddress": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    },
                    "action": {
                        "fromAmount": "1000000",
                        "fromToken": {
                            "address": "0x1111111111111111111111111111111111111111",
                        },
                    },
                    "transactionRequest": {
                        "to": "0x3333333333333333333333333333333333333333",
                        "data": "0x",
                    },
                },
            )
        ]
    )
    runtime = _runtime(
        secrets=secrets,
        settings=settings,
        http=http,
        rpc_map={},
    )
    out = build_swap_prepare_graph(runtime).invoke(
        {
            "input": {
                "from_chain": "ethereum",
                "to_chain": "base",
                "from_asset": "0x1111111111111111111111111111111111111111",
                "to_asset": "0x2222222222222222222222222222222222222222",
                "from_amount_wei": "1000000",
                "from_address": "0x4444444444444444444444444444444444444444",
                "to_address": "0x5555555555555555555555555555555555555555",
            }
        }
    )
    assert out["result"]["prepared"]["route_id"] == "swap-route-1"
    al = out["result"]["allowance"]
    assert al is not None
    assert al["token_address"] == "0x1111111111111111111111111111111111111111"
    assert al["spender_address"] == "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert al["amount_raw"] == "1000000"
    _assert_no_banned_values(out)


def test_swap_prepare_graph_quote_url_includes_slippage_order_integrator():
    """LiFi GET /v1/quote query matches OpenAPI-style integrator, slippage, order."""

    secrets = {"vault/lifi": "INJECTED_LIFI_KEY_BBB"}
    settings = AureySettings(lifi_api_secret_path="vault/lifi")
    urls: list[str] = []

    def match_quote(**kw: object) -> bool:
        urls.append(str(kw.get("url") or ""))
        return kw.get("method") == "GET" and "/v1/quote?" in str(kw.get("url") or "")

    http = ScriptedHttpClient(
        [
            (
                match_quote,
                {
                    "id": "q-slippage",
                    "transactionRequest": {
                        "to": "0x3333333333333333333333333333333333333333",
                        "data": "0x",
                    },
                },
            )
        ]
    )
    runtime = _runtime(secrets=secrets, settings=settings, http=http, rpc_map={})
    out = build_swap_prepare_graph(runtime).invoke(
        {
            "input": {
                "from_chain": "ethereum",
                "to_chain": "base",
                "from_asset": "usdc",
                "to_asset": "eth",
                "from_amount_wei": "1000000",
                "from_address": "0x4444444444444444444444444444444444444444",
                "to_address": "0x5555555555555555555555555555555555555555",
                "slippage": 0.01,
                "order": "CHEAPEST",
            }
        }
    )
    assert out["result"]["prepared"]["route_id"] == "q-slippage"
    assert len(urls) == 1
    u = urls[0]
    assert "integrator=aurey" in u
    assert "slippage=0.01" in u
    assert "order=CHEAPEST" in u
    _assert_no_banned_values(out)


def test_swap_prepare_graph_skips_allowance_hint_when_on_chain_sufficient():
    """When Alchemy-backed allowance is already >= LiFi fromAmount, omit approve hint."""

    alchemy_path = "vault/alchemy/x"
    secrets = {"vault/lifi": "INJECTED_LIFI_KEY_BBB", alchemy_path: "INJECTED_ALCHEMY_KEY_AAA"}
    settings = AureySettings(
        lifi_api_secret_path="vault/lifi",
        alchemy_api_secret_path=alchemy_path,
    )
    http = ScriptedHttpClient(
        [
            (
                lambda **kw: kw.get("method") == "GET" and "/v1/quote?" in str(kw.get("url") or ""),
                {
                    "id": "swap-route-allow",
                    "estimate": {
                        "approvalAddress": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    },
                    "action": {
                        "fromAmount": "1000000",
                        "fromToken": {
                            "address": "0x1111111111111111111111111111111111111111",
                        },
                    },
                    "transactionRequest": {
                        "to": "0x3333333333333333333333333333333333333333",
                        "data": "0x",
                    },
                },
            )
        ]
    )

    def eth_call(params: list[object]) -> str:
        assert params[0]["to"] == "0x1111111111111111111111111111111111111111"
        return "0x00000000000000000000000000000000000000000000000000000000000f4240"

    runtime = _runtime(
        secrets=secrets,
        settings=settings,
        http=http,
        rpc_map={"eth_call": eth_call},
    )
    out = build_swap_prepare_graph(runtime).invoke(
        {
            "input": {
                "from_chain": "ethereum",
                "to_chain": "base",
                "from_asset": "0x1111111111111111111111111111111111111111",
                "to_asset": "0x2222222222222222222222222222222222222222",
                "from_amount_wei": "1000000",
                "from_address": "0x4444444444444444444444444444444444444444",
                "to_address": "0x5555555555555555555555555555555555555555",
            }
        }
    )
    assert out["result"]["prepared"]["route_id"] == "swap-route-allow"
    assert out["result"].get("allowance") is None
    _assert_no_banned_values(out)


def test_swap_prepare_graph_keeps_allowance_hint_when_on_chain_low():
    secrets = {"vault/lifi": "INJECTED_LIFI_KEY_BBB", "vault/alchemy/x": "INJECTED_ALCHEMY_KEY_AAA"}
    settings = AureySettings(
        lifi_api_secret_path="vault/lifi",
        alchemy_api_secret_path="vault/alchemy/x",
    )
    http = ScriptedHttpClient(
        [
            (
                lambda **kw: kw.get("method") == "GET" and "/v1/quote?" in str(kw.get("url") or ""),
                {
                    "id": "swap-low",
                    "estimate": {
                        "approvalAddress": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    },
                    "action": {
                        "fromAmount": "1000000",
                        "fromToken": {
                            "address": "0x1111111111111111111111111111111111111111",
                        },
                    },
                    "transactionRequest": {
                        "to": "0x3333333333333333333333333333333333333333",
                        "data": "0x",
                    },
                },
            )
        ]
    )

    def eth_call(_params: list[object]) -> str:
        return "0x0000000000000000000000000000000000000000000000000000000000000064"

    runtime = _runtime(
        secrets=secrets,
        settings=settings,
        http=http,
        rpc_map={"eth_call": eth_call},
    )
    out = build_swap_prepare_graph(runtime).invoke(
        {
            "input": {
                "from_chain": "ethereum",
                "to_chain": "base",
                "from_asset": "0x1111111111111111111111111111111111111111",
                "to_asset": "0x2222222222222222222222222222222222222222",
                "from_amount_wei": "1000000",
                "from_address": "0x4444444444444444444444444444444444444444",
                "to_address": "0x5555555555555555555555555555555555555555",
            }
        }
    )
    al = out["result"]["allowance"]
    assert al is not None
    assert al["amount_raw"] == "1000000"
    _assert_no_banned_values(out)


def test_swap_prepare_graph_without_lifi_api_key():
    settings = AureySettings(lifi_api_secret_path=None)

    def match_quote(**kw: object) -> bool:
        headers = kw.get("headers") or {}
        u = str(kw.get("url") or "")
        return (
            kw.get("method") == "GET"
            and "li.quest" in u
            and "/v1/quote?" in u
            and isinstance(headers, dict)
            and "x-lifi-api-key" not in headers
            and "integrator=aurey" in u
        )

    http = ScriptedHttpClient(
        [
            (
                match_quote,
                {
                    "id": "public-quote",
                    "transactionRequest": {
                        "to": "0x3333333333333333333333333333333333333333",
                        "data": "0x",
                    },
                },
            )
        ]
    )
    runtime = _runtime(secrets={}, settings=settings, http=http, rpc_map={})
    out = build_swap_prepare_graph(runtime).invoke(
        {
            "input": {
                "from_chain": "base",
                "to_chain": "base",
                "from_asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                "to_asset": "0x4200000000000000000000000000000000000006",
                "from_amount_wei": "1000000",
                "from_address": "0x4444444444444444444444444444444444444444",
                "to_address": "0x5555555555555555555555555555555555555555",
            }
        }
    )
    assert out["result"]["prepared"]["route_id"] == "public-quote"
    assert out["result"].get("allowance") is None


def test_swap_prepare_graph_maps_lifi_http_json_errors():
    settings = AureySettings(lifi_api_secret_path="vault/lifi")
    runtime = _runtime(
        secrets={"vault/lifi": "not-a-real-key"},
        settings=settings,
        http=_LifiUnauthorizedHttp(),
        rpc_map={},
    )
    out = build_swap_prepare_graph(runtime).invoke(
        {
            "input": {
                "from_chain": "base",
                "to_chain": "base",
                "from_asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                "to_asset": "0x4200000000000000000000000000000000000006",
                "from_amount_wei": "1000000",
                "from_address": "0x4444444444444444444444444444444444444444",
                "to_address": "0x5555555555555555555555555555555555555555",
            }
        }
    )
    err = out["error"]
    assert err["code"] == "http_error"
    assert err["details"]["http_status"] == 401
    assert err["details"]["lifi_message"] == "Invalid API key"
    assert err["details"]["lifi_code"] == 1010


def test_tx_prepare_lifi_swap_graph():
    signing_path = "vault/signing/local"
    secrets = {signing_path: "0x" + "ff" * 32}
    settings = AureySettings(wallet_signing_key_secret_path=signing_path)
    runtime = _runtime(secrets=secrets, settings=settings, http=ScriptedHttpClient(), rpc_map={})
    wallet = "0xc1923710468607b8b7db38a6afbb9b432744390c"
    prepared = {
        "route_id": "4026c5d3-23c3-494d-8c1e-b1c9ba89657c:0",
        "transaction_request": {
            "to": "0x1234567890123456789012345678901234567890",
            "data": "0xcafe",
            "value": "0x0",
            "chainId": 8453,
            "from": wallet,
            "gasLimit": "0x5208",
        },
    }
    out = build_tx_prepare_lifi_graph(runtime).invoke(
        {
            "input": {
                "chain": "base",
                "from_address": wallet,
                "prepared": prepared,
            }
        }
    )
    assert out.get("error") is None
    env = out["result"]["envelope"]
    assert env["kind"] == "lifi_swap"
    assert env["chain_id"] == 8453
    assert env["to"] == "0x1234567890123456789012345678901234567890"
    assert env["data"] == "0xcafe"
    assert env["value_hex"] == "0x0"
    assert env["gas_limit_hex"] == "0x5208"
    assert env["signing_key_secret_path"] == signing_path
    assert env["signing_mode"] == "vault_key"
    _assert_no_banned_values(out)


def test_tx_prepare_lifi_swap_graph_flat_route_and_transaction_request():
    signing_path = "vault/signing/local"
    secrets = {signing_path: "0x" + "ff" * 32}
    settings = AureySettings(wallet_signing_key_secret_path=signing_path)
    runtime = _runtime(secrets=secrets, settings=settings, http=ScriptedHttpClient(), rpc_map={})
    wallet = "0xc1923710468607b8b7db38a6afbb9b432744390c"
    out = build_tx_prepare_lifi_graph(runtime).invoke(
        {
            "input": {
                "chain": "base",
                "from_address": wallet,
                "route_id": "f3288cb0-08fb-4b91-8b39-98f41ffad017:0",
                "transaction_request": {
                    "to": "0x1234567890123456789012345678901234567890",
                    "data": "0x",
                    "value": "0x0",
                    "chainId": 8453,
                },
            }
        }
    )
    assert out.get("error") is None
    assert out["result"]["envelope"]["kind"] == "lifi_swap"


def test_tx_prepare_lifi_swap_then_execute_roundtrip():
    signing_path = "vault/signing/local"
    secrets = {signing_path: "0x" + "ff" * 32}
    settings = AureySettings(wallet_signing_key_secret_path=signing_path)
    runtime = _runtime(secrets=secrets, settings=settings, http=ScriptedHttpClient(), rpc_map={})
    wallet = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    prepared = {
        "route_id": "lane-1",
        "transaction_request": {
            "to": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "data": "0xdeadbeef",
            "value": 0,
            "chainId": 8453,
        },
    }
    prep = build_tx_prepare_lifi_graph(runtime).invoke(
        {"input": {"chain": "base", "from_address": wallet, "prepared": prepared}}
    )
    execute = build_tx_execute_graph(runtime).invoke(
        {"input": {"envelope": prep["result"]["envelope"]}}
    )
    assert execute.get("error") is None
    assert execute["result"]["tx_hash"].startswith("0x")
    assert execute["result"]["receipt"]["status"] == 1
    _assert_no_banned_values(execute)


def test_tx_prepare_and_execute_native_roundtrip():
    signing_path = "vault/signing/local"
    secrets = {
        signing_path: "0x" + "ff" * 32,
    }
    settings = AureySettings(
        wallet_signing_key_secret_path=signing_path,
    )
    runtime = _runtime(secrets=secrets, settings=settings, http=ScriptedHttpClient(), rpc_map={})

    prepare = build_tx_prepare_graph(runtime).invoke(
        {
            "input": {
                "kind": "native_transfer",
                "chain": "ethereum",
                "from_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "to_address": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "value_wei": 5,
            }
        }
    )
    envelope = prepare["result"]["envelope"]
    assert envelope["kind"] == "native_transfer"
    assert envelope["data"] == "0x"
    assert envelope["signing_key_secret_path"] == signing_path
    _assert_no_banned_values(prepare)

    execute = build_tx_execute_graph(runtime).invoke({"input": {"envelope": envelope}})
    assert execute.get("error") is None
    assert execute["result"]["tx_hash"].startswith("0x")
    assert execute["result"]["receipt"]["status"] == 1
    _assert_no_banned_values(execute)


def test_tx_prepare_erc20_paths():
    signing_path = "vault/signing/local"
    secrets = {signing_path: "0x" + "ff" * 32}
    settings = AureySettings(wallet_signing_key_secret_path=signing_path)
    runtime = _runtime(secrets=secrets, settings=settings, http=ScriptedHttpClient(), rpc_map={})

    transfer = build_tx_prepare_graph(runtime).invoke(
        {
            "input": {
                "kind": "erc20_transfer",
                "chain": "base",
                "from_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "token_address": "0xcccccccccccccccccccccccccccccccccccccccc",
                "to_address": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "amount_wei": 42,
            }
        }
    )
    assert transfer["result"]["envelope"]["kind"] == "erc20_transfer"
    assert transfer["result"]["envelope"]["data"].startswith("0xa9059cbb")

    approval = build_tx_prepare_graph(runtime).invoke(
        {
            "input": {
                "kind": "erc20_approval",
                "chain": "base",
                "from_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "token_address": "0xcccccccccccccccccccccccccccccccccccccccc",
                "spender_address": "0xdddddddddddddddddddddddddddddddddddddddd",
                "amount_wei": 99,
            }
        }
    )
    assert approval["result"]["envelope"]["kind"] == "erc20_approval"
    assert approval["result"]["envelope"]["data"].startswith("0x095ea7b3")
    _assert_no_banned_values(approval)


def test_tx_execute_simulation_failure():
    signing_path = "vault/signing/local"
    secrets = {signing_path: "0x" + "ff" * 32}
    settings = AureySettings(wallet_signing_key_secret_path=signing_path)
    http = ScriptedHttpClient()
    runtime = AureyRuntime(
        settings=settings,
        secret_store=FakeSecretStore(secrets),
        evm_rpc_factory=rpc_factory_from_mapping({}),
        http=http,
        tx_pipeline=DeterministicTxPipeline(fail_stage="simulate"),
        lifi_base_url="https://li.quest",
    )
    env = {
        "kind": "native_transfer",
        "chain_id": 1,
        "from_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "to": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "data": "0x",
        "value_hex": "0x1",
        "gas_limit_hex": None,
        "nonce": None,
        "signing_key_secret_path": signing_path,
    }
    out = build_tx_execute_graph(runtime).invoke({"input": {"envelope": env}})
    assert out.get("result") is None
    assert out["error"]["code"] == "simulation_failed"
    assert "simulation_failed" in out["error"]["message"]
