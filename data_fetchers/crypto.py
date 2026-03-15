"""
Crypto market data fetchers.
  - Chainlink: on-chain price feeds via ETH RPC (public, no key)
  - Pyth Network: Hermes REST API (public, no key)
  - CoinGecko: free public API for OHLCV
"""
import logging
import time
from typing import List, Optional
import requests
import pandas as pd
from config.settings import (
    CHAINLINK_ETH_RPC,
    PYTH_API_URL,
    CRYPTO_WATCHLIST,
)

logger = logging.getLogger(__name__)

# Chainlink ETH/USD aggregator on Ethereum mainnet
CHAINLINK_FEEDS = {
    "ETH": "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419",
    "BTC": "0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c",
    "LINK": "0x2c1d072e956AFFC0D435Cb7AC38EF18d24d9127c",
    "SOL": "0x4ffC43a60e009B551865A93d232E33Fce9f01507",
}

# Pyth price feed IDs (mainnet)
PYTH_FEED_IDS = {
    "BTC": "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
    "ETH": "0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
    "SOL": "0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
}


class ChainlinkFetcher:
    """Read Chainlink oracle prices via ETH JSON-RPC (public, free)."""

    # latestRoundData() selector
    SELECTOR = "0xfeaf968c"

    def __init__(self, rpc_url: str = CHAINLINK_ETH_RPC):
        self.rpc = rpc_url

    def _call_contract(self, address: str) -> Optional[dict]:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_call",
            "params": [{"to": address, "data": self.SELECTOR}, "latest"],
        }
        try:
            resp = requests.post(self.rpc, json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json().get("result", "0x")
            if len(result) < 66:
                return None
            # Decode: roundId(uint80), answer(int256), startedAt, updatedAt, answeredInRound
            answer_hex = result[66:130]
            updated_hex = result[194:258]
            price = int(answer_hex, 16) / 1e8
            updated_at = int(updated_hex, 16)
            return {"price": price, "updated_at": updated_at}
        except Exception as e:
            logger.error("Chainlink RPC call failed for %s: %s", address, e)
            return None

    def get_prices(self) -> pd.DataFrame:
        """Get latest prices from Chainlink feeds."""
        rows = []
        for symbol, address in CHAINLINK_FEEDS.items():
            data = self._call_contract(address)
            if data:
                rows.append({
                    "symbol": symbol,
                    "price": data["price"],
                    "updated_at": pd.Timestamp(data["updated_at"], unit="s"),
                    "source": "chainlink",
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame()


class PythFetcher:
    """Read Pyth Network prices via Hermes REST API (public, free)."""

    BASE = PYTH_API_URL

    def get_prices(self) -> pd.DataFrame:
        """Get latest prices from Pyth Network."""
        feed_ids = list(PYTH_FEED_IDS.values())
        try:
            resp = requests.get(
                f"{self.BASE}/v2/updates/price/latest",
                params={"ids[]": feed_ids},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            parsed = data.get("parsed", [])
            if not parsed:
                return pd.DataFrame()

            id_to_symbol = {v: k for k, v in PYTH_FEED_IDS.items()}
            rows = []
            for entry in parsed:
                fid = "0x" + entry.get("id", "")
                symbol = id_to_symbol.get(fid, fid[:10])
                price_data = entry.get("price", {})
                price = int(price_data.get("price", 0)) * 10 ** int(
                    price_data.get("expo", 0)
                )
                rows.append({
                    "symbol": symbol,
                    "price": price,
                    "confidence": int(price_data.get("conf", 0))
                    * 10 ** int(price_data.get("expo", 0)),
                    "publish_time": pd.Timestamp(
                        entry.get("price", {}).get("publish_time", 0), unit="s"
                    ),
                    "source": "pyth",
                })
            return pd.DataFrame(rows)
        except Exception as e:
            logger.error("Pyth fetch failed: %s", e)
            return pd.DataFrame()


class CoinGeckoFetcher:
    """Free CoinGecko API for OHLCV history (no key, rate-limited)."""

    BASE = "https://api.coingecko.com/api/v3"
    COIN_MAP = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
        "LINK": "chainlink", "AVAX": "avalanche-2",
        "MATIC": "matic-network", "ARB": "arbitrum",
    }

    def get_ohlcv(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """Get OHLCV data for a crypto asset."""
        coin_id = self.COIN_MAP.get(symbol.upper())
        if not coin_id:
            logger.warning("Unknown crypto symbol: %s", symbol)
            return pd.DataFrame()
        try:
            resp = requests.get(
                f"{self.BASE}/coins/{coin_id}/ohlc",
                params={"vs_currency": "usd", "days": days},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["symbol"] = symbol.upper()
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logger.error("CoinGecko fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()

    def get_all_ohlcv(self, symbols: Optional[List[str]] = None, days: int = 90) -> dict:
        """Get OHLCV for multiple symbols with rate-limit respect."""
        symbols = symbols or CRYPTO_WATCHLIST
        result = {}
        for sym in symbols:
            result[sym] = self.get_ohlcv(sym, days)
            time.sleep(1.5)  # CoinGecko rate limit
        return result
