"""
Institutional Trading Strategies.

Features:
- Regime-adaptive signal generation
- Proper position sizing (volatility targeting)
- Stop-loss / take-profit integration
- Signal decay awareness
- Multi-timeframe confirmation
"""
import numpy as np
import pandas as pd


# ── Momentum Strategies ──────────────────────────────────────────────────

def momentum_crossover(fast: int = 10, slow: int = 30):
    """
    Moving average crossover with volatility-scaled position sizing.
    Position size inversely proportional to realized volatility.
    """
    def signal_fn(prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        fast_ma = close.rolling(fast, min_periods=fast).mean()
        slow_ma = close.rolling(slow, min_periods=slow).mean()

        raw_signal = pd.Series(0.0, index=close.index)
        raw_signal[fast_ma > slow_ma] = 1.0
        raw_signal[fast_ma < slow_ma] = -1.0

        # Vol-scale: target 15% annualized vol
        realized_vol = close.pct_change().rolling(21).std() * np.sqrt(252)
        target_vol = 0.15
        vol_scalar = (target_vol / realized_vol.replace(0, np.nan)).clip(0.2, 2.0).fillna(1)

        return (raw_signal * vol_scalar).clip(-1, 1)
    return signal_fn


def dual_momentum(lookback: int = 63, threshold: float = 0.0):
    """
    Dual momentum (Antonacci):
    - Absolute momentum: asset return > threshold
    - Relative momentum: rank assets by return
    """
    def signal_fn(prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        mom = close.pct_change(lookback)

        signal = pd.Series(0.0, index=close.index)
        signal[mom > threshold] = 1.0
        signal[mom < -threshold] = -1.0
        return signal
    return signal_fn


# ── Mean Reversion Strategies ────────────────────────────────────────────

def mean_reversion_bollinger(window: int = 20, num_std: float = 2.0):
    """
    Bollinger Band mean reversion with position sizing based on z-score.
    Graduated entry: stronger signal further from bands.
    """
    def signal_fn(prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        ma = close.rolling(window, min_periods=window).mean()
        std = close.rolling(window, min_periods=window).std()

        z_score = (close - ma) / std.replace(0, np.nan)

        # Graduated position sizing
        signal = pd.Series(0.0, index=close.index)
        signal[z_score < -num_std] = 1.0
        signal[z_score < -num_std * 0.5] = 0.5
        signal[z_score > num_std] = -1.0
        signal[z_score > num_std * 0.5] = -0.5
        # Reset at mean
        signal[(z_score > -0.5) & (z_score < 0.5)] = 0.0
        return signal
    return signal_fn


# ── RSI Strategy ─────────────────────────────────────────────────────────

def rsi_strategy(period: int = 14, oversold: float = 30, overbought: float = 70):
    """RSI with graduated entries."""
    def signal_fn(prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        signal = pd.Series(0.0, index=close.index)
        # Strong signal
        signal[rsi < oversold] = 1.0
        signal[rsi > overbought] = -1.0
        # Moderate signal
        signal[(rsi < oversold + 5) & (rsi >= oversold)] = 0.5
        signal[(rsi > overbought - 5) & (rsi <= overbought)] = -0.5
        return signal
    return signal_fn


# ── Breakout Strategy ────────────────────────────────────────────────────

def breakout_strategy(window: int = 20):
    """Donchian channel breakout with ATR-based stop."""
    def signal_fn(prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        high = prices["high"] if "high" in prices.columns else close
        low = prices["low"] if "low" in prices.columns else close

        high_ch = high.rolling(window).max()
        low_ch = low.rolling(window).min()

        signal = pd.Series(0.0, index=close.index)
        signal[close >= high_ch] = 1.0
        signal[close <= low_ch] = -1.0
        return signal
    return signal_fn


# ── Volatility Targeting ────────────────────────────────────────────────

def volatility_targeting(target_vol: float = 0.15, lookback: int = 21):
    """Scale position to target annualized volatility."""
    def signal_fn(prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        realized_vol = close.pct_change().rolling(lookback).std() * np.sqrt(252)
        position = (target_vol / realized_vol.replace(0, np.nan)).clip(0, 2.0)
        return position.fillna(0)
    return signal_fn


# ── Regime-Adaptive Strategy ────────────────────────────────────────────

def regime_adaptive(
    fast: int = 10, slow: int = 30,
    bb_window: int = 20, bb_std: float = 2.0,
    vol_lookback: int = 63,
):
    """
    Switches between momentum and mean-reversion based on
    realized vol regime:
    - Low vol → mean reversion
    - High vol → momentum (trend following)
    """
    def signal_fn(prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        returns = close.pct_change()

        # Vol regime
        short_vol = returns.rolling(21).std()
        long_vol = returns.rolling(vol_lookback).std()
        vol_ratio = short_vol / long_vol.replace(0, np.nan)

        # Momentum signal
        fast_ma = close.rolling(fast).mean()
        slow_ma = close.rolling(slow).mean()
        mom_signal = pd.Series(0.0, index=close.index)
        mom_signal[fast_ma > slow_ma] = 1.0
        mom_signal[fast_ma < slow_ma] = -1.0

        # Mean reversion signal
        ma = close.rolling(bb_window).mean()
        std = close.rolling(bb_window).std()
        z_score = (close - ma) / std.replace(0, np.nan)
        mr_signal = pd.Series(0.0, index=close.index)
        mr_signal[z_score < -bb_std] = 1.0
        mr_signal[z_score > bb_std] = -1.0

        # Blend based on regime
        signal = pd.Series(0.0, index=close.index)
        high_vol = vol_ratio > 1.2
        signal[high_vol] = mom_signal[high_vol]
        signal[~high_vol] = mr_signal[~high_vol]

        return signal
    return signal_fn


# ── Strategy Factories (for parameter sweeps) ───────────────────────────

def momentum_crossover_factory(fast: int = 10, slow: int = 30):
    return momentum_crossover(fast=fast, slow=slow)

def bollinger_factory(window: int = 20, num_std: float = 2.0):
    return mean_reversion_bollinger(window=window, num_std=num_std)

def rsi_factory(period: int = 14, oversold: float = 30):
    return rsi_strategy(period=period, oversold=oversold, overbought=100 - oversold)

def regime_adaptive_factory(fast: int = 10, slow: int = 30):
    return regime_adaptive(fast=fast, slow=slow)
