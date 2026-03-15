"""
Built-in trading strategies for backtesting and screening.
Each strategy is a signal function: prices DataFrame → position Series.
"""
import numpy as np
import pandas as pd


def momentum_crossover(fast: int = 10, slow: int = 30):
    """Moving average crossover: long when fast > slow, else flat."""
    def signal_fn(prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        fast_ma = close.rolling(fast).mean()
        slow_ma = close.rolling(slow).mean()
        signal = pd.Series(0.0, index=close.index)
        signal[fast_ma > slow_ma] = 1.0
        signal[fast_ma < slow_ma] = -1.0
        return signal
    return signal_fn


def mean_reversion_bollinger(window: int = 20, num_std: float = 2.0):
    """Bollinger Band mean reversion: buy at lower band, sell at upper band."""
    def signal_fn(prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        ma = close.rolling(window).mean()
        std = close.rolling(window).std()
        upper = ma + num_std * std
        lower = ma - num_std * std

        signal = pd.Series(0.0, index=close.index)
        signal[close < lower] = 1.0   # buy
        signal[close > upper] = -1.0  # sell
        return signal
    return signal_fn


def rsi_strategy(period: int = 14, oversold: float = 30, overbought: float = 70):
    """RSI-based: buy when oversold, sell when overbought."""
    def signal_fn(prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        signal = pd.Series(0.0, index=close.index)
        signal[rsi < oversold] = 1.0
        signal[rsi > overbought] = -1.0
        return signal
    return signal_fn


def breakout_strategy(window: int = 20):
    """Donchian channel breakout: long above high, short below low."""
    def signal_fn(prices: pd.DataFrame) -> pd.Series:
        high_ch = prices["high"].rolling(window).max() if "high" in prices.columns else prices["close"].rolling(window).max()
        low_ch = prices["low"].rolling(window).min() if "low" in prices.columns else prices["close"].rolling(window).min()
        close = prices["close"]

        signal = pd.Series(0.0, index=close.index)
        signal[close >= high_ch] = 1.0
        signal[close <= low_ch] = -1.0
        return signal
    return signal_fn


def volatility_targeting(target_vol: float = 0.15, lookback: int = 21):
    """Scale position size to target a specific annualized volatility."""
    def signal_fn(prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        returns = close.pct_change()
        realized_vol = returns.rolling(lookback).std() * np.sqrt(252)
        # Position size = target / realized (capped at 2x)
        position = (target_vol / realized_vol.replace(0, np.nan)).clip(0, 2.0)
        return position.fillna(0)
    return signal_fn


# ── Strategy factories for parameter sweeps ──────────────────────────────

def momentum_crossover_factory(fast: int = 10, slow: int = 30):
    """Factory for parameter sweep over momentum crossover."""
    return momentum_crossover(fast=fast, slow=slow)


def bollinger_factory(window: int = 20, num_std: float = 2.0):
    """Factory for parameter sweep over Bollinger bands."""
    return mean_reversion_bollinger(window=window, num_std=num_std)


def rsi_factory(period: int = 14, oversold: float = 30):
    """Factory for parameter sweep over RSI."""
    return rsi_strategy(period=period, oversold=oversold, overbought=100 - oversold)
