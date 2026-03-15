"""
Alpha Signal Engine — combines market data with news sentiment
to generate cross-market alpha signals.
"""
import logging
import re
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Simple keyword-based sentiment (no ML dependency) ────────────────────
POSITIVE_WORDS = {
    "surge", "soar", "rally", "bullish", "gain", "profit", "boom", "beat",
    "upgrade", "breakout", "high", "record", "growth", "positive", "strong",
    "outperform", "buy", "upside", "recover", "accelerate",
}
NEGATIVE_WORDS = {
    "crash", "plunge", "bearish", "loss", "drop", "decline", "selloff",
    "downgrade", "risk", "fear", "recession", "default", "bankruptcy",
    "warning", "miss", "cut", "weak", "fraud", "collapse", "crisis",
}


def simple_sentiment_score(text: str) -> float:
    """Return sentiment score in [-1, 1] from keyword frequency."""
    if not text:
        return 0.0
    words = set(re.findall(r"\w+", text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


class AlphaSignalEngine:
    """Generate alpha signals by cross-referencing markets, prices, and news."""

    def compute_news_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment scores to a news DataFrame."""
        if news_df.empty:
            return news_df
        news_df = news_df.copy()
        news_df["sentiment"] = (
            news_df["title"].fillna("") + " " + news_df.get("description", pd.Series(dtype=str)).fillna("")
        ).apply(simple_sentiment_score)
        return news_df

    def aggregate_asset_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sentiment per asset symbol."""
        if news_df.empty or "asset_symbol" not in news_df.columns:
            return pd.DataFrame()
        if "sentiment" not in news_df.columns:
            news_df = self.compute_news_sentiment(news_df)
        agg = (
            news_df.groupby("asset_symbol")
            .agg(
                mean_sentiment=("sentiment", "mean"),
                article_count=("sentiment", "count"),
                max_sentiment=("sentiment", "max"),
                min_sentiment=("sentiment", "min"),
            )
            .reset_index()
        )
        return agg

    def momentum_signal(self, prices: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """Compute momentum signals across multiple lookback windows."""
        windows = windows or [5, 10, 21, 63]
        if prices.empty or "close" not in prices.columns:
            return pd.DataFrame()
        signals = pd.DataFrame(index=prices.index)
        for w in windows:
            signals[f"mom_{w}d"] = prices["close"].pct_change(w)
        signals["mom_composite"] = signals.mean(axis=1)
        return signals

    def mean_reversion_signal(self, prices: pd.DataFrame, window: int = 20) -> pd.Series:
        """Z-score based mean reversion signal."""
        if prices.empty or "close" not in prices.columns:
            return pd.Series(dtype=float)
        rolling_mean = prices["close"].rolling(window).mean()
        rolling_std = prices["close"].rolling(window).std()
        z_score = (prices["close"] - rolling_mean) / rolling_std
        return -z_score  # negative z = buy signal

    def prediction_market_edge(
        self, pred_df: pd.DataFrame, sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Find alpha where prediction market prices diverge from news sentiment.
        If sentiment is strongly positive but market implies low probability → potential long.
        """
        if pred_df.empty or sentiment_df.empty:
            return pd.DataFrame()

        rows = []
        for _, market in pred_df.iterrows():
            yes_price = market.get("outcome_yes") or market.get("yes_price")
            if yes_price is None:
                continue

            # Match sentiment by keyword overlap
            question = str(market.get("question", "") or market.get("title", ""))
            question_words = set(re.findall(r"\w+", question.lower()))

            best_match_sent = 0.0
            best_match_count = 0
            for _, row in sentiment_df.iterrows():
                asset_words = set(re.findall(r"\w+", str(row.get("asset_symbol", "")).lower()))
                if asset_words & question_words:
                    best_match_sent = row.get("mean_sentiment", 0)
                    best_match_count = row.get("article_count", 0)
                    break

            # Edge = sentiment - implied probability
            implied_prob = float(yes_price)
            edge = best_match_sent - (implied_prob - 0.5) * 2  # normalize to [-1,1]

            rows.append({
                "market": question[:80],
                "implied_prob": implied_prob,
                "news_sentiment": best_match_sent,
                "edge_signal": edge,
                "news_coverage": best_match_count,
                "source": market.get("source", ""),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df.sort_values("edge_signal", ascending=False, inplace=True, key=abs)
        return df

    def cross_market_signals(
        self,
        stock_prices: Dict[str, pd.DataFrame],
        crypto_prices: Dict[str, pd.DataFrame],
        sentiment_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate composite alpha signals across all asset classes."""
        rows = []
        for asset_type, prices_dict in [("stock", stock_prices), ("crypto", crypto_prices)]:
            for symbol, prices in prices_dict.items():
                if prices.empty:
                    continue
                mom = self.momentum_signal(prices)
                mr = self.mean_reversion_signal(prices)

                # Latest values
                latest_mom = mom["mom_composite"].iloc[-1] if not mom.empty else 0
                latest_mr = mr.iloc[-1] if not mr.empty else 0

                # Sentiment
                sent = 0.0
                if not sentiment_df.empty and "asset_symbol" in sentiment_df.columns:
                    match = sentiment_df[sentiment_df["asset_symbol"] == symbol]
                    if not match.empty:
                        sent = match["mean_sentiment"].iloc[0]

                # Composite signal: weighted sum
                composite = 0.4 * latest_mom + 0.3 * latest_mr + 0.3 * sent
                last_price = prices["close"].iloc[-1] if "close" in prices.columns else None
                vol_20d = prices["close"].pct_change().rolling(20).std().iloc[-1] if "close" in prices.columns else None

                rows.append({
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "momentum": latest_mom,
                    "mean_reversion": latest_mr,
                    "sentiment": sent,
                    "composite_alpha": composite,
                    "last_price": last_price,
                    "volatility_20d": vol_20d,
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df.sort_values("composite_alpha", ascending=False, inplace=True)
        return df
