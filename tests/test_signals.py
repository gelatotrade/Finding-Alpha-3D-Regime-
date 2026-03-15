"""Tests for alpha signal engine."""
import numpy as np
import pandas as pd
import pytest
from signals.alpha_engine import (
    AlphaSignalEngine,
    enhanced_sentiment_score,
)


class TestSentiment:
    def test_positive_sentiment(self):
        result = enhanced_sentiment_score("stock price surged to record high growth")
        assert result["score"] > 0
        assert result["pos_count"] > 0

    def test_negative_sentiment(self):
        result = enhanced_sentiment_score("market crash fear recession decline")
        assert result["score"] < 0
        assert result["neg_count"] > 0

    def test_neutral_text(self):
        result = enhanced_sentiment_score("the cat sat on the mat")
        assert result["score"] == 0
        assert result["confidence"] == 0

    def test_negation_handling(self):
        result_pos = enhanced_sentiment_score("the stock is bullish")
        result_neg = enhanced_sentiment_score("the stock is not bullish")
        # Negated positive should be less positive (or negative)
        assert result_neg["score"] <= result_pos["score"]

    def test_empty_text(self):
        result = enhanced_sentiment_score("")
        assert result["score"] == 0

    def test_credibility_weighting(self):
        text = "stock surged to record high"
        high_cred = enhanced_sentiment_score(text, 1.0)
        low_cred = enhanced_sentiment_score(text, 0.3)
        assert abs(high_cred["score"]) >= abs(low_cred["score"])


class TestAlphaEngine:
    def _make_prices(self, n=252):
        np.random.seed(42)
        dates = pd.bdate_range("2024-01-01", periods=n)
        returns = np.random.normal(0.0005, 0.015, n)
        return pd.DataFrame({
            "close": 100 * np.exp(np.cumsum(returns)),
        }, index=dates)

    def test_momentum_signal(self):
        engine = AlphaSignalEngine()
        prices = self._make_prices()
        signals = engine.momentum_signal(prices)
        assert "mom_composite" in signals.columns
        assert len(signals) == len(prices)

    def test_mean_reversion_signal(self):
        engine = AlphaSignalEngine()
        prices = self._make_prices()
        signal = engine.mean_reversion_signal(prices)
        assert len(signal) == len(prices)

    def test_hurst_exponent(self):
        engine = AlphaSignalEngine()
        # Trending series
        np.random.seed(42)
        trending = pd.Series(np.cumsum(np.random.randn(500)))
        h = engine.compute_hurst_exponent(trending)
        assert 0 <= h <= 1

    def test_information_coefficient(self):
        engine = AlphaSignalEngine()
        np.random.seed(42)
        signal = pd.Series(np.random.randn(100))
        returns = signal * 0.3 + np.random.randn(100) * 0.7  # partially predictive
        ic = engine.compute_ic(signal, returns)
        assert "ic" in ic
        assert "p_value" in ic
        assert -1 <= ic["ic"] <= 1

    def test_alpha_decay(self):
        engine = AlphaSignalEngine()
        prices = self._make_prices(300)
        signal = engine.momentum_signal(prices)["mom_composite"]
        decay = engine.alpha_decay_analysis(signal, prices, horizons=[1, 5, 10])
        assert not decay.empty
        assert "horizon_days" in decay.columns
        assert "ic" in decay.columns

    def test_cross_market_signals(self):
        engine = AlphaSignalEngine()
        prices = self._make_prices()
        stock_prices = {"TEST": prices}
        crypto_prices = {}
        sentiment = pd.DataFrame()

        signals = engine.cross_market_signals(stock_prices, crypto_prices, sentiment)
        assert not signals.empty
        assert "composite_alpha" in signals.columns
        assert "hurst_exponent" in signals.columns
        assert "regime" in signals.columns
