"""
Institutional ARIMA-based trading strategies.

All strategies enforce strict no-lookahead: at time t, the signal
can only see data strictly before t.
"""
import logging
from typing import Callable
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Helper: cached ARIMA rolling forecast ────────────────────────────────

_FORECAST_CACHE = {}


def _cached_forecast_series(
    price_id: int,
    returns: pd.Series,
    train_window: int,
    refit_every: int,
    order: tuple,
) -> pd.Series:
    """Cache rolling ARIMA forecasts to avoid recomputation across param sweeps."""
    key = (price_id, train_window, refit_every, order, len(returns))
    if key in _FORECAST_CACHE:
        return _FORECAST_CACHE[key]

    from signals.arima_forecaster import ArimaForecaster
    fc = ArimaForecaster()
    forecasts = fc.rolling_forecast(
        returns, train_window=train_window, refit_every=refit_every, order=order,
    )
    if forecasts.empty:
        result = pd.Series(0.0, index=returns.index)
    else:
        result = forecasts["forecast_mean"].reindex(returns.index).fillna(0)

    _FORECAST_CACHE[key] = result
    return result


# ── Strategy: ARIMA Direction ────────────────────────────────────────────

def arima_direction(
    train_window: int = 252,
    refit_every: int = 42,
    threshold_std: float = 0.5,
    order: tuple = (2, 0, 2),
):
    """
    Trade based on direction of ARIMA one-step-ahead mean forecast.
    Position = sign(forecast) if |forecast| > threshold_std * forecast_std.
    """
    def signal_fn(prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        returns = close.pct_change().fillna(0)

        from signals.arima_forecaster import ArimaForecaster
        fc = ArimaForecaster()
        forecasts = fc.rolling_forecast(
            returns, train_window=train_window,
            refit_every=refit_every, order=order,
        )
        if forecasts.empty:
            return pd.Series(0.0, index=close.index)

        signal = pd.Series(0.0, index=close.index)
        thresh = forecasts["forecast_std"] * threshold_std
        signal.loc[forecasts.index] = np.where(
            forecasts["forecast_mean"] > thresh, 1.0,
            np.where(forecasts["forecast_mean"] < -thresh, -1.0, 0.0),
        )
        return signal
    return signal_fn


# ── Strategy: ARIMA Confidence-Scaled ───────────────────────────────────

def arima_confidence(
    train_window: int = 252,
    refit_every: int = 42,
    order: tuple = (2, 0, 2),
    confidence_cap: float = 2.0,
):
    """
    Position sized by forecast signal-to-noise ratio (|mean| / std).
    Higher confidence → larger position.
    """
    def signal_fn(prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        returns = close.pct_change().fillna(0)

        from signals.arima_forecaster import ArimaForecaster
        fc = ArimaForecaster()
        forecasts = fc.rolling_forecast(
            returns, train_window=train_window,
            refit_every=refit_every, order=order,
        )
        if forecasts.empty:
            return pd.Series(0.0, index=close.index)

        signal = pd.Series(0.0, index=close.index)
        # Position = forecast / std, capped at ±confidence_cap
        snr = forecasts["forecast_mean"] / forecasts["forecast_std"].replace(0, np.nan)
        snr = snr.clip(-confidence_cap, confidence_cap) / confidence_cap
        signal.loc[forecasts.index] = snr.fillna(0).values
        return signal
    return signal_fn


# ── Strategy: ARIMA + Momentum Filter ───────────────────────────────────

def arima_momentum_filter(
    train_window: int = 252,
    refit_every: int = 42,
    order: tuple = (1, 0, 1),
    momentum_window: int = 63,
    threshold_std: float = 0.3,
):
    """
    ARIMA signal gated by longer-term momentum.
    Only take ARIMA long if momentum positive (and vice versa).
    This combines short-term prediction with trend confirmation.
    """
    def signal_fn(prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        returns = close.pct_change().fillna(0)

        from signals.arima_forecaster import ArimaForecaster
        fc = ArimaForecaster()
        forecasts = fc.rolling_forecast(
            returns, train_window=train_window,
            refit_every=refit_every, order=order,
        )
        if forecasts.empty:
            return pd.Series(0.0, index=close.index)

        # Momentum filter
        momentum = close.pct_change(momentum_window)

        signal = pd.Series(0.0, index=close.index)
        thresh = forecasts["forecast_std"] * threshold_std

        for idx in forecasts.index:
            arima_sign = (1.0 if forecasts.loc[idx, "forecast_mean"] > thresh.loc[idx]
                          else -1.0 if forecasts.loc[idx, "forecast_mean"] < -thresh.loc[idx]
                          else 0.0)
            mom = momentum.loc[idx] if idx in momentum.index else 0
            if np.isnan(mom):
                continue
            # Only take signal if aligned with momentum direction
            if arima_sign > 0 and mom > 0:
                signal.loc[idx] = 1.0
            elif arima_sign < 0 and mom < 0:
                signal.loc[idx] = -1.0

        return signal
    return signal_fn


# ── Strategy: ARIMA + GARCH Vol-Scaled ──────────────────────────────────

def arima_garch_scaled(
    train_window: int = 252,
    refit_every: int = 42,
    order: tuple = (2, 0, 2),
    target_vol: float = 0.15,
    threshold_std: float = 0.3,
):
    """
    ARIMA direction, sized by GARCH conditional volatility.
    Reduces position in high-vol regimes, increases in low-vol.
    """
    def signal_fn(prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        returns = close.pct_change().fillna(0)

        from signals.arima_forecaster import ArimaForecaster
        fc = ArimaForecaster()
        forecasts = fc.rolling_forecast(
            returns, train_window=train_window,
            refit_every=refit_every, order=order,
        )
        if forecasts.empty:
            return pd.Series(0.0, index=close.index)

        signal = pd.Series(0.0, index=close.index)
        thresh = forecasts["forecast_std"] * threshold_std
        arima_dir = np.where(
            forecasts["forecast_mean"] > thresh, 1.0,
            np.where(forecasts["forecast_mean"] < -thresh, -1.0, 0.0),
        )

        # Realized vol as GARCH proxy (faster than full GARCH fit per step)
        realized_vol = returns.rolling(21).std() * np.sqrt(252)
        vol_scalar = (target_vol / realized_vol.replace(0, np.nan)).clip(0.2, 2.0)

        raw_signal = pd.Series(arima_dir, index=forecasts.index)
        sized = raw_signal * vol_scalar.reindex(forecasts.index).fillna(1)
        signal.loc[forecasts.index] = sized.clip(-2, 2).values
        return signal
    return signal_fn


# ── Factories for parameter sweep ───────────────────────────────────────

def arima_direction_factory(
    train_window: int = 252,
    refit_every: int = 42,
    threshold_std: float = 0.5,
    order: tuple = (2, 0, 2),
):
    return arima_direction(train_window, refit_every, threshold_std, order)


def arima_confidence_factory(
    train_window: int = 252,
    refit_every: int = 42,
    order: tuple = (2, 0, 2),
):
    return arima_confidence(train_window, refit_every, order)


def arima_momentum_factory(
    train_window: int = 252,
    refit_every: int = 42,
    order: tuple = (1, 0, 1),
    momentum_window: int = 63,
    threshold_std: float = 0.3,
):
    return arima_momentum_filter(train_window, refit_every, order,
                                  momentum_window, threshold_std)


def arima_garch_factory(
    train_window: int = 252,
    refit_every: int = 42,
    order: tuple = (2, 0, 2),
    target_vol: float = 0.15,
    threshold_std: float = 0.3,
):
    return arima_garch_scaled(train_window, refit_every, order, target_vol, threshold_std)
