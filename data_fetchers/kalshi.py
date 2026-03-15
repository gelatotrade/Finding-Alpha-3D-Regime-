"""
Kalshi prediction market data fetcher with resilient HTTP client.
"""
import logging
from typing import Optional
import pandas as pd
from config.settings import KALSHI_CONFIG
from config.resilience import ResilientClient
from config.validation import validate_prediction_market

logger = logging.getLogger(__name__)


class KalshiFetcher:
    """Fetch live market data from Kalshi's public API."""

    def __init__(self):
        headers = {"Accept": "application/json"}
        if KALSHI_CONFIG.api_key:
            headers["Authorization"] = f"Bearer {KALSHI_CONFIG.api_key}"
        self.client = ResilientClient(
            name="kalshi",
            base_url=KALSHI_CONFIG.base_url,
            timeout=KALSHI_CONFIG.timeout_s,
            retry_policy=KALSHI_CONFIG.retry_policy,
            rate_limit_per_min=KALSHI_CONFIG.rate_limit_per_min,
            default_headers=headers,
        )

    def get_events(self, limit: int = 50, status: str = "open") -> pd.DataFrame:
        """Return a validated DataFrame of current Kalshi events."""
        resp = self.client.get("events", params={"limit": limit, "status": status})
        if resp is None:
            return pd.DataFrame()

        try:
            data = resp.json()
        except ValueError:
            logger.error("Kalshi returned invalid JSON")
            return pd.DataFrame()

        events = data.get("events", [])
        if not isinstance(events, list) or not events:
            return pd.DataFrame()

        rows = []
        for ev in events:
            for mkt in ev.get("markets", []):
                yes_raw = mkt.get("yes_price")
                no_raw = mkt.get("no_price")
                try:
                    yes_price = float(yes_raw) / 100.0 if yes_raw is not None else None
                except (ValueError, TypeError):
                    yes_price = None
                try:
                    no_price = float(no_raw) / 100.0 if no_raw is not None else None
                except (ValueError, TypeError):
                    no_price = None

                rows.append({
                    "event_id": str(ev.get("event_ticker", "")),
                    "market_id": str(mkt.get("ticker", "")),
                    "title": str(mkt.get("title", "")),
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "volume": int(mkt.get("volume", 0) or 0),
                    "open_interest": int(mkt.get("open_interest", 0) or 0),
                    "close_time": str(mkt.get("close_time", "")),
                    "source": "kalshi",
                })

        df = pd.DataFrame(rows)
        df, report = validate_prediction_market(df, "kalshi")
        if not report.passed:
            logger.warning("Kalshi data quality: %s", report)
        return df

    def get_market_history(self, ticker: str, limit: int = 500) -> pd.DataFrame:
        """Get price history for a specific Kalshi market."""
        resp = self.client.get(f"markets/{ticker}/history", params={"limit": limit})
        if resp is None:
            return pd.DataFrame()

        try:
            history = resp.json().get("history", [])
        except (ValueError, AttributeError):
            return pd.DataFrame()

        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)
        if "ts" in df.columns:
            df["timestamp"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
            df.sort_values("timestamp", inplace=True)
        return df
