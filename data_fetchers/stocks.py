"""
Stock market data fetchers with resilience, validation, and quality scoring.
  - yfinance: free, no key required (primary)
  - Finnhub: free tier with API key (live quotes)
  - Alpha Vantage: free tier with API key (backup)
"""
import logging
from typing import Dict, List, Optional
import pandas as pd
from config.settings import ALPHA_VANTAGE_CONFIG, FINNHUB_CONFIG, STOCK_WATCHLIST
from config.resilience import ResilientClient
from config.validation import validate_ohlcv

logger = logging.getLogger(__name__)

# Import yfinance once at module level
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    logger.warning("yfinance not installed — stock data from yfinance unavailable")


class YFinanceFetcher:
    """Yahoo Finance via yfinance with validation."""

    def get_ohlcv(self, symbol: str, period: str = "3mo",
                  interval: str = "1d") -> pd.DataFrame:
        if not _YF_AVAILABLE:
            return pd.DataFrame()
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                return pd.DataFrame()

            df["symbol"] = symbol
            df.index.name = "timestamp"
            df = df[["Open", "High", "Low", "Close", "Volume", "symbol"]].rename(
                columns={"Open": "open", "High": "high", "Low": "low",
                         "Close": "close", "Volume": "volume"}
            )
            # Validate
            df, report = validate_ohlcv(df, f"yfinance_{symbol}")
            if not report.passed:
                logger.warning("yfinance %s quality: %s", symbol, report)
            return df
        except Exception as e:
            logger.error("yfinance fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()

    def get_batch_ohlcv(self, symbols: Optional[List[str]] = None,
                        period: str = "3mo") -> Dict[str, pd.DataFrame]:
        symbols = symbols or STOCK_WATCHLIST
        result = {}
        for sym in symbols:
            df = self.get_ohlcv(sym, period)
            if not df.empty:
                result[sym] = df
        return result

    def get_live_quote(self, symbol: str) -> dict:
        if not _YF_AVAILABLE:
            return {}
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            return {
                "symbol": symbol,
                "price": getattr(info, "last_price", None),
                "market_cap": getattr(info, "market_cap", None),
                "volume": getattr(info, "last_volume", None),
                "source": "yfinance",
            }
        except Exception as e:
            logger.error("yfinance quote failed for %s: %s", symbol, e)
            return {}


class FinnhubFetcher:
    """Finnhub.io with resilient client."""

    def __init__(self):
        self.available = bool(FINNHUB_CONFIG.api_key)
        if self.available:
            self.client = ResilientClient(
                name="finnhub", base_url=FINNHUB_CONFIG.base_url,
                timeout=FINNHUB_CONFIG.timeout_s,
                retry_policy=FINNHUB_CONFIG.retry_policy,
                rate_limit_per_min=FINNHUB_CONFIG.rate_limit_per_min,
            )

    def get_quote(self, symbol: str) -> dict:
        if not self.available:
            return {}
        resp = self.client.get("quote", params={"symbol": symbol,
                                                 "token": FINNHUB_CONFIG.api_key})
        if resp is None:
            return {}
        try:
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
        except (ValueError, AttributeError):
            return {}

    def get_company_news(self, symbol: str, from_date: str,
                         to_date: str) -> list:
        if not self.available:
            return []
        resp = self.client.get("company-news",
                               params={"symbol": symbol, "from": from_date,
                                       "to": to_date, "token": FINNHUB_CONFIG.api_key})
        if resp is None:
            return []
        try:
            return resp.json()
        except ValueError:
            return []


class AlphaVantageFetcher:
    """Alpha Vantage with resilient client and rate-limit awareness."""

    def __init__(self):
        self.available = bool(ALPHA_VANTAGE_CONFIG.api_key)
        if self.available:
            self.client = ResilientClient(
                name="alphavantage", base_url=ALPHA_VANTAGE_CONFIG.base_url,
                timeout=ALPHA_VANTAGE_CONFIG.timeout_s,
                retry_policy=ALPHA_VANTAGE_CONFIG.retry_policy,
                rate_limit_per_min=ALPHA_VANTAGE_CONFIG.rate_limit_per_min,
            )

    def get_daily(self, symbol: str, outputsize: str = "compact") -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()

        resp = self.client.get("", params={
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": ALPHA_VANTAGE_CONFIG.api_key,
        })
        if resp is None:
            return pd.DataFrame()

        try:
            data = resp.json()
        except ValueError:
            return pd.DataFrame()

        # Alpha Vantage returns errors in JSON body, not HTTP status
        if "Error Message" in data or "Note" in data:
            logger.error("Alpha Vantage error for %s: %s",
                         symbol, data.get("Error Message") or data.get("Note"))
            return pd.DataFrame()

        ts = data.get("Time Series (Daily)", {})
        if not ts:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(ts, orient="index").astype(float)
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index)
        df.index.name = "timestamp"
        df.sort_index(inplace=True)
        df["symbol"] = symbol

        df, report = validate_ohlcv(df, f"alphavantage_{symbol}")
        return df
