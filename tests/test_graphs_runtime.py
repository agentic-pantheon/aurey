"""Runtime + LangGraph coverage (fake SecretStore and scripted clients)."""

from __future__ import annotations

import json

from aurey.custody import FakeSecretStore
from aurey.graphs import (
    DeterministicTxPipeline,
    build_alchemy_graph,
    build_read_graph,
    build_swap_prepare_graph,
    build_tx_execute_graph,
    build_tx_prepare_graph,
)
from aurey.runtime import AureyRuntime
from aurey.settings import AureySettings
from tests.fakes.evm_rpc import rpc_factory_from_mapping
from tests.fakes.http_client import ScriptedHttpClient


def _runtime(
    *,
    secrets: dict[str, str],
    settings: AureySettings,
    http: ScriptedHttpClient,
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
    assert rpc_urls == ["https://eth-mainnet.g.alchemy.com/v2/INJECTED_ALCHEMY_KEY_AAA"]
    _assert_no_banned_values(out)


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
    assert out["result"]["resolved_address"] == "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
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
                            }
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
        return (
            kw.get("method") == "POST"
            and "li.quest" in str(kw.get("url") or "")
            and isinstance(headers, dict)
            and headers.get("X-API-KEY") == "INJECTED_LIFI_KEY_BBB"
        )

    http = ScriptedHttpClient(
        [
            (
                match_quote,
                {
                    "routeId": "swap-route-1",
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
    _assert_no_banned_values(out)


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
