"""
Polymarket prediction market data fetcher.
Uses the public CLOB API with resilient HTTP client.
"""
import logging
from typing import Optional
import pandas as pd
from config.settings import POLYMARKET_CONFIG
from config.resilience import ResilientClient
from config.validation import validate_prediction_market

logger = logging.getLogger(__name__)


class PolymarketFetcher:
    """Fetch live market data from Polymarket's public CLOB API."""

    def __init__(self):
        self.client = ResilientClient(
            name="polymarket",
            base_url=POLYMARKET_CONFIG.base_url,
            timeout=POLYMARKET_CONFIG.timeout_s,
            retry_policy=POLYMARKET_CONFIG.retry_policy,
            rate_limit_per_min=POLYMARKET_CONFIG.rate_limit_per_min,
        )

    def get_markets(self, limit: int = 100, active_only: bool = True) -> pd.DataFrame:
        """Return a validated DataFrame of current prediction markets."""
        resp = self.client.get("markets", params={"limit": limit, "active": active_only})
        if resp is None:
            return pd.DataFrame()

        try:
            markets = resp.json()
        except ValueError:
            logger.error("Polymarket returned invalid JSON")
            return pd.DataFrame()

        if not isinstance(markets, list) or not markets:
            return pd.DataFrame()

        rows = []
        for m in markets:
            tokens = m.get("tokens", [])
            yes_price = None
            no_price = None
            if tokens and len(tokens) >= 1:
                try:
                    yes_price = float(tokens[0].get("price", 0))
                except (ValueError, TypeError):
                    pass
            if tokens and len(tokens) >= 2:
                try:
                    no_price = float(tokens[1].get("price", 0))
                except (ValueError, TypeError):
                    pass

            rows.append({
                "market_id": str(m.get("condition_id", "")),
                "question": str(m.get("question", "")),
                "outcome_yes": yes_price,
                "outcome_no": no_price,
                "volume": float(m.get("volume", 0) or 0),
                "liquidity": float(m.get("liquidity", 0) or 0),
                "end_date": str(m.get("end_date_iso", "")),
                "source": "polymarket",
            })

        df = pd.DataFrame(rows)
        df, report = validate_prediction_market(df, "polymarket")
        if not report.passed:
            logger.warning("Polymarket data quality: %s", report)
        return df

    def get_market_orderbook(self, token_id: str) -> dict:
        """Get orderbook for a specific market token."""
        resp = self.client.get("book", params={"token_id": token_id})
        if resp is None:
            return {}
        try:
            return resp.json()
        except ValueError:
            return {}

    def get_market_trades(self, token_id: str, limit: int = 100) -> pd.DataFrame:
        """Get recent trades for a market."""
        resp = self.client.get("trades", params={"token_id": token_id, "limit": limit})
        if resp is None:
            return pd.DataFrame()
        try:
            trades = resp.json()
        except ValueError:
            return pd.DataFrame()

        if not isinstance(trades, list) or not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        return df
