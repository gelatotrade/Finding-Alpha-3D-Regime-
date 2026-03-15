"""
Crypto market data fetchers with resilience, validation, and oracle aggregation.
  - Chainlink: on-chain price feeds via ETH RPC
  - Pyth Network: Hermes REST API
  - CoinGecko: free public API for OHLCV
  - Oracle Aggregator: median-of-medians across sources
"""
import logging
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from config.settings import CHAINLINK_ETH_RPC, PYTH_API_URL, CRYPTO_WATCHLIST
from config.resilience import ResilientClient
from config.validation import validate_ohlcv

logger = logging.getLogger(__name__)

# ── Chainlink feed addresses (Ethereum mainnet) ─────────────────────────
CHAINLINK_FEEDS: Dict[str, dict] = {
    "ETH": {"address": "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419", "decimals": 8},
    "BTC": {"address": "0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c", "decimals": 8},
    "LINK": {"address": "0x2c1d072e956AFFC0D435Cb7AC38EF18d24d9127c", "decimals": 8},
    "SOL": {"address": "0x4ffC43a60e009B551865A93d232E33Fce9f01507", "decimals": 8},
}

# ── Pyth feed IDs ────────────────────────────────────────────────────────
PYTH_FEED_IDS: Dict[str, str] = {
    "BTC": "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
    "ETH": "0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
    "SOL": "0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
}

# Max oracle staleness (seconds) — reject prices older than this
MAX_ORACLE_STALENESS_S = 600


class ChainlinkFetcher:
    """Read Chainlink oracle prices via ETH JSON-RPC with staleness checks."""

    SELECTOR = "0xfeaf968c"  # latestRoundData()

    def __init__(self, rpc_url: str = CHAINLINK_ETH_RPC):
        self.client = ResilientClient(
            name="chainlink_rpc", base_url=rpc_url,
            timeout=10.0, rate_limit_per_min=30,
        )

    def _call_contract(self, address: str, decimals: int) -> Optional[dict]:
        payload = {
            "jsonrpc": "2.0", "id": 1, "method": "eth_call",
            "params": [{"to": address, "data": self.SELECTOR}, "latest"],
        }
        resp = self.client.post("", json=payload)
        if resp is None:
            return None

        try:
            result = resp.json().get("result", "0x")
        except (ValueError, AttributeError):
            return None

        # Validate response length: 5 * 32 bytes = 320 hex chars + "0x"
        if not isinstance(result, str) or len(result) < 322:
            logger.warning("Chainlink: unexpected response length for %s", address)
            return None

        try:
            # Decode ABI: (roundId, answer, startedAt, updatedAt, answeredInRound)
            answer_hex = result[66:130]
            updated_hex = result[194:258]
            price = int(answer_hex, 16) / (10 ** decimals)
            updated_at = int(updated_hex, 16)

            # Staleness check
            now = int(time.time())
            age_s = now - updated_at
            if age_s > MAX_ORACLE_STALENESS_S:
                logger.warning("Chainlink %s: stale by %ds (max %ds)",
                               address, age_s, MAX_ORACLE_STALENESS_S)
                return None

            return {"price": price, "updated_at": updated_at, "age_s": age_s}
        except (ValueError, OverflowError) as e:
            logger.error("Chainlink decode error for %s: %s", address, e)
            return None

    def get_prices(self) -> pd.DataFrame:
        rows = []
        for symbol, feed in CHAINLINK_FEEDS.items():
            data = self._call_contract(feed["address"], feed["decimals"])
            if data:
                rows.append({
                    "symbol": symbol,
                    "price": data["price"],
                    "updated_at": pd.Timestamp(data["updated_at"], unit="s"),
                    "age_s": data["age_s"],
                    "source": "chainlink",
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame()


class PythFetcher:
    """Read Pyth Network prices via Hermes REST API with confidence tracking."""

    def __init__(self):
        self.client = ResilientClient(
            name="pyth", base_url=PYTH_API_URL,
            timeout=10.0, rate_limit_per_min=60,
        )

    def get_prices(self) -> pd.DataFrame:
        feed_ids = list(PYTH_FEED_IDS.values())
        resp = self.client.get("v2/updates/price/latest", params={"ids[]": feed_ids})
        if resp is None:
            return pd.DataFrame()

        try:
            data = resp.json()
        except ValueError:
            return pd.DataFrame()

        parsed = data.get("parsed", [])
        if not parsed:
            return pd.DataFrame()

        id_to_symbol = {v: k for k, v in PYTH_FEED_IDS.items()}
        rows = []
        for entry in parsed:
            fid = "0x" + entry.get("id", "")
            symbol = id_to_symbol.get(fid, fid[:10])
            price_data = entry.get("price", {})

            try:
                expo = int(price_data.get("expo", 0))
                price = int(price_data.get("price", 0)) * (10 ** expo)
                conf = int(price_data.get("conf", 0)) * (10 ** expo)
            except (ValueError, OverflowError):
                continue

            pub_time = price_data.get("publish_time", 0)

            # Staleness check
            age_s = int(time.time()) - int(pub_time)
            if age_s > MAX_ORACLE_STALENESS_S:
                logger.warning("Pyth %s stale by %ds", symbol, age_s)
                continue

            # Confidence check: reject if confidence > 2% of price
            if price > 0 and conf / price > 0.02:
                logger.warning("Pyth %s: low confidence (conf/price=%.2f%%)",
                               symbol, (conf / price) * 100)

            rows.append({
                "symbol": symbol,
                "price": price,
                "confidence": conf,
                "confidence_pct": (conf / price * 100) if price > 0 else None,
                "publish_time": pd.Timestamp(int(pub_time), unit="s"),
                "age_s": age_s,
                "source": "pyth",
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()


class CoinGeckoFetcher:
    """CoinGecko OHLCV with rate limiting and validation."""

    COIN_MAP = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
        "LINK": "chainlink", "AVAX": "avalanche-2",
        "MATIC": "matic-network", "ARB": "arbitrum",
    }

    def __init__(self):
        self.client = ResilientClient(
            name="coingecko", base_url="https://api.coingecko.com/api/v3",
            timeout=15.0, rate_limit_per_min=25,  # conservative for free tier
        )

    def get_ohlcv(self, symbol: str, days: int = 90) -> pd.DataFrame:
        coin_id = self.COIN_MAP.get(symbol.upper())
        if not coin_id:
            logger.warning("Unknown crypto symbol: %s", symbol)
            return pd.DataFrame()

        resp = self.client.get(f"coins/{coin_id}/ohlc",
                               params={"vs_currency": "usd", "days": days})
        if resp is None:
            return pd.DataFrame()

        try:
            data = resp.json()
        except ValueError:
            return pd.DataFrame()

        if not isinstance(data, list) or not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)
        df["symbol"] = symbol.upper()
        df.set_index("timestamp", inplace=True)

        # Validate
        df, report = validate_ohlcv(df, f"coingecko_{symbol}")
        if report.warnings:
            logger.info("CoinGecko %s: %s", symbol, report)
        return df

    def get_all_ohlcv(self, symbols: Optional[List[str]] = None,
                      days: int = 90) -> Dict[str, pd.DataFrame]:
        symbols = symbols or CRYPTO_WATCHLIST
        result = {}
        for sym in symbols:
            result[sym] = self.get_ohlcv(sym, days)
        return result


class OracleAggregator:
    """
    Aggregate prices across multiple oracle sources.
    Uses median price when multiple sources available — resilient
    to single-source manipulation or failure.
    """

    def __init__(self):
        self.chainlink = ChainlinkFetcher()
        self.pyth = PythFetcher()

    def get_aggregated_prices(self) -> pd.DataFrame:
        cl = self.chainlink.get_prices()
        py = self.pyth.get_prices()

        if cl.empty and py.empty:
            return pd.DataFrame()

        # Merge on symbol
        all_prices = []
        symbols = set()
        if not cl.empty:
            symbols |= set(cl["symbol"])
        if not py.empty:
            symbols |= set(py["symbol"])

        for sym in symbols:
            prices = []
            sources = []

            if not cl.empty:
                cl_row = cl[cl["symbol"] == sym]
                if not cl_row.empty:
                    prices.append(cl_row.iloc[0]["price"])
                    sources.append("chainlink")

            if not py.empty:
                py_row = py[py["symbol"] == sym]
                if not py_row.empty:
                    prices.append(py_row.iloc[0]["price"])
                    sources.append("pyth")

            if prices:
                median_price = float(np.median(prices))
                spread = (max(prices) - min(prices)) / median_price * 100 if len(prices) > 1 else 0

                all_prices.append({
                    "symbol": sym,
                    "price": median_price,
                    "n_sources": len(prices),
                    "sources": ",".join(sources),
                    "cross_source_spread_pct": spread,
                    "timestamp": pd.Timestamp.now(),
                })

                # Warn on large spread between oracles
                if spread > 1.0:
                    logger.warning("Oracle divergence for %s: %.2f%% spread across %s",
                                   sym, spread, sources)

        return pd.DataFrame(all_prices) if all_prices else pd.DataFrame()
