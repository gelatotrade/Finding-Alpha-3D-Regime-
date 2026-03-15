"""
Kalshi prediction market data fetcher.
Uses the public v2 API (API key optional for public market reads).
"""
import logging
from typing import Optional
import requests
import pandas as pd
from config.settings import KALSHI_API_URL, KALSHI_API_KEY

logger = logging.getLogger(__name__)


class KalshiFetcher:
    """Fetch live market data from Kalshi's public API."""

    BASE = KALSHI_API_URL

    def _headers(self) -> dict:
        h = {"Accept": "application/json"}
        if KALSHI_API_KEY:
            h["Authorization"] = f"Bearer {KALSHI_API_KEY}"
        return h

    def get_events(self, limit: int = 50, status: str = "open") -> pd.DataFrame:
        """Return a DataFrame of current Kalshi events."""
        try:
            resp = requests.get(
                f"{self.BASE}/events",
                headers=self._headers(),
                params={"limit": limit, "status": status},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            events = data.get("events", [])
            if not events:
                return pd.DataFrame()

            rows = []
            for ev in events:
                for mkt in ev.get("markets", []):
                    rows.append({
                        "event_id": ev.get("event_ticker", ""),
                        "market_id": mkt.get("ticker", ""),
                        "title": mkt.get("title", ""),
                        "yes_price": mkt.get("yes_price", 0) / 100.0
                        if mkt.get("yes_price") else None,
                        "no_price": mkt.get("no_price", 0) / 100.0
                        if mkt.get("no_price") else None,
                        "volume": mkt.get("volume", 0),
                        "open_interest": mkt.get("open_interest", 0),
                        "close_time": mkt.get("close_time", ""),
                        "source": "kalshi",
                    })
            return pd.DataFrame(rows)
        except Exception as e:
            logger.error("Kalshi fetch failed: %s", e)
            return pd.DataFrame()

    def get_market_history(
        self, ticker: str, limit: int = 500
    ) -> pd.DataFrame:
        """Get price history for a specific Kalshi market."""
        try:
            resp = requests.get(
                f"{self.BASE}/markets/{ticker}/history",
                headers=self._headers(),
                params={"limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            history = resp.json().get("history", [])
            if not history:
                return pd.DataFrame()
            df = pd.DataFrame(history)
            if "ts" in df.columns:
                df["timestamp"] = pd.to_datetime(df["ts"], unit="s")
            return df
        except Exception as e:
            logger.error("Kalshi history fetch failed: %s", e)
            return pd.DataFrame()
