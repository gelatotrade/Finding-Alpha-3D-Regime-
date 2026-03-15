"""
Stock market data fetchers using free APIs.
  - yfinance: free, no key required
  - Finnhub: free tier with API key
  - Alpha Vantage: free tier with API key
"""
import logging
from typing import List, Optional
import requests
import pandas as pd
from config.settings import (
    ALPHA_VANTAGE_KEY,
    FINNHUB_KEY,
    STOCK_WATCHLIST,
)

logger = logging.getLogger(__name__)


class YFinanceFetcher:
    """Yahoo Finance via yfinance (free, no key)."""

    def get_ohlcv(
        self, symbol: str, period: str = "3mo", interval: str = "1d"
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                return pd.DataFrame()
            df["symbol"] = symbol
            df.index.name = "timestamp"
            return df[["Open", "High", "Low", "Close", "Volume", "symbol"]].rename(
                columns={"Open": "open", "High": "high", "Low": "low",
                         "Close": "close", "Volume": "volume"}
            )
        except Exception as e:
            logger.error("yfinance fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()

    def get_batch_ohlcv(
        self, symbols: Optional[List[str]] = None, period: str = "3mo"
    ) -> dict:
        symbols = symbols or STOCK_WATCHLIST
        result = {}
        for sym in symbols:
            result[sym] = self.get_ohlcv(sym, period)
        return result

    def get_live_quote(self, symbol: str) -> dict:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            return {
                "symbol": symbol,
                "price": info.get("lastPrice", None),
                "market_cap": info.get("marketCap", None),
                "volume": info.get("lastVolume", None),
                "source": "yfinance",
            }
        except Exception as e:
            logger.error("yfinance quote failed for %s: %s", symbol, e)
            return {}


class FinnhubFetcher:
    """Finnhub.io free tier (API key required)."""

    BASE = "https://finnhub.io/api/v1"

    def _params(self, **kwargs) -> dict:
        return {"token": FINNHUB_KEY, **kwargs}

    def get_quote(self, symbol: str) -> dict:
        if not FINNHUB_KEY:
            return {}
        try:
            resp = requests.get(
                f"{self.BASE}/quote",
                params=self._params(symbol=symbol),
                timeout=10,
            )
            resp.raise_for_status()
            q = resp.json()
            return {
                "symbol": symbol,
                "price": q.get("c"),
                "change": q.get("d"),
                "change_pct": q.get("dp"),
                "high": q.get("h"),
                "low": q.get("l"),
                "open": q.get("o"),
                "prev_close": q.get("pc"),
                "source": "finnhub",
            }
        except Exception as e:
            logger.error("Finnhub quote failed for %s: %s", symbol, e)
            return {}

    def get_company_news(self, symbol: str, from_date: str, to_date: str) -> list:
        if not FINNHUB_KEY:
            return []
        try:
            resp = requests.get(
                f"{self.BASE}/company-news",
                params=self._params(symbol=symbol, **{"from": from_date, "to": to_date}),
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("Finnhub news failed for %s: %s", symbol, e)
            return []


class AlphaVantageFetcher:
    """Alpha Vantage free tier (API key required, 25 requests/day)."""

    BASE = "https://www.alphavantage.co/query"

    def get_daily(self, symbol: str, outputsize: str = "compact") -> pd.DataFrame:
        if not ALPHA_VANTAGE_KEY:
            return pd.DataFrame()
        try:
            resp = requests.get(
                self.BASE,
                params={
                    "function": "TIME_SERIES_DAILY",
                    "symbol": symbol,
                    "outputsize": outputsize,
                    "apikey": ALPHA_VANTAGE_KEY,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            ts = data.get("Time Series (Daily)", {})
            if not ts:
                return pd.DataFrame()
            df = pd.DataFrame.from_dict(ts, orient="index").astype(float)
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index = pd.to_datetime(df.index)
            df.index.name = "timestamp"
            df.sort_index(inplace=True)
            df["symbol"] = symbol
            return df
        except Exception as e:
            logger.error("Alpha Vantage fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()
