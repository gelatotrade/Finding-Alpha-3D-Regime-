"""
Polymarket prediction market data fetcher.
Uses the public CLOB API (no key required for reads).
"""
import logging
from typing import Optional
import requests
import pandas as pd
from config.settings import POLYMARKET_API_URL

logger = logging.getLogger(__name__)


class PolymarketFetcher:
    """Fetch live market data from Polymarket's public CLOB API."""

    BASE = POLYMARKET_API_URL

    def get_markets(self, limit: int = 50, active_only: bool = True) -> pd.DataFrame:
        """Return a DataFrame of current prediction markets."""
        try:
            resp = requests.get(
                f"{self.BASE}/markets",
                params={"limit": limit, "active": active_only},
                timeout=15,
            )
            resp.raise_for_status()
            markets = resp.json()
            if not markets:
                return pd.DataFrame()

            rows = []
            for m in markets:
                rows.append({
                    "market_id": m.get("condition_id", ""),
                    "question": m.get("question", ""),
                    "outcome_yes": float(m.get("tokens", [{}])[0].get("price", 0))
                    if m.get("tokens") else None,
                    "outcome_no": float(m.get("tokens", [{}])[1].get("price", 0))
                    if m.get("tokens") and len(m.get("tokens", [])) > 1 else None,
                    "volume": float(m.get("volume", 0)),
                    "liquidity": float(m.get("liquidity", 0)),
                    "end_date": m.get("end_date_iso", ""),
                    "source": "polymarket",
                })
            return pd.DataFrame(rows)
        except Exception as e:
            logger.error("Polymarket fetch failed: %s", e)
            return pd.DataFrame()

    def get_market_orderbook(self, token_id: str) -> dict:
        """Get orderbook for a specific market token."""
        try:
            resp = requests.get(
                f"{self.BASE}/book",
                params={"token_id": token_id},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("Polymarket orderbook fetch failed: %s", e)
            return {}

    def get_market_trades(self, token_id: str, limit: int = 100) -> pd.DataFrame:
        """Get recent trades for a market."""
        try:
            resp = requests.get(
                f"{self.BASE}/trades",
                params={"token_id": token_id, "limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            trades = resp.json()
            if not trades:
                return pd.DataFrame()
            df = pd.DataFrame(trades)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            return df
        except Exception as e:
            logger.error("Polymarket trades fetch failed: %s", e)
            return pd.DataFrame()
