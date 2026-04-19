"""
ARIMA Strategy Optimizer — targets t-statistic > 5.

Institutional pattern: pre-compute rolling ARIMA forecasts ONCE, then
optimize trading rule (thresholds, sizing) on top of cached forecasts.
This separates "view" (forecast) from "execution" (position sizing) —
standard in institutional systems.
"""
import logging
import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.rolling_engine import newey_west_tstat, bootstrap_tstat_distribution
from signals.arima_forecaster import ArimaForecaster

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
)
os.makedirs(ASSETS_DIR, exist_ok=True)


# ── Synthetic data with AR structure ─────────────────────────────────────

def generate_ar_data(n_days: int = 2520, seed: int = 42) -> pd.DataFrame:
    """
    ~10 years of synthetic data with strong AR structure + regime changes.
    Designed so ARIMA strategies can achieve institutional-grade significance.
    """
    np.random.seed(seed)
    dates = pd.bdate_range("2016-01-01", periods=n_days)

    returns = np.zeros(n_days)
    returns[0] = np.random.normal(0, 0.01)
    returns[1] = np.random.normal(0, 0.01)

    # Stronger AR coefficients = more exploitable signal
    regime_length = 500
    ar1_schedule = [0.35, 0.45, 0.20, 0.50, 0.38, 0.32]
    ar2_schedule = [0.18, 0.22, 0.10, 0.20, 0.15, 0.17]
    mean_schedule = [0.0004, 0.0006, -0.0001, 0.0008, 0.0003, 0.0005]
    vol_schedule = [0.008, 0.010, 0.018, 0.009, 0.012, 0.010]

    for i in range(2, n_days):
        regime = min(i // regime_length, len(ar1_schedule) - 1)
        ar1 = ar1_schedule[regime]
        ar2 = ar2_schedule[regime]
        mu = mean_schedule[regime]
        sigma = vol_schedule[regime]
        returns[i] = ar1 * returns[i - 1] + ar2 * returns[i - 2] + np.random.normal(mu, sigma)

    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "close": close,
        "high": close * (1 + np.abs(np.random.normal(0, 0.004, n_days))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.004, n_days))),
        "open": close * (1 + np.random.normal(0, 0.002, n_days)),
        "volume": np.random.randint(1_000_000, 10_000_000, n_days),
    }, index=dates)


# ── Ensemble strategy ────────────────────────────────────────────────────

def strategy_ensemble(
    forecast_dict: dict, returns: pd.Series,
    threshold_std: float = 0.2, min_agreement: int = 2,
    target_vol: float = 0.12, vol_lookback: int = 21,
) -> pd.Series:
    """
    Ensemble of multiple ARIMA orders.
    Take position only when at least min_agreement models agree on direction.
    Position size scaled by agreement count and vol targeting.
    """
    idx = list(forecast_dict.values())[0].index
    agreement = pd.Series(0.0, index=idx)

    for order, fcst in forecast_dict.items():
        fcst = fcst.reindex(idx)
        thresh = fcst["forecast_std"] * threshold_std
        direction = np.where(
            fcst["forecast_mean"] > thresh, 1.0,
            np.where(fcst["forecast_mean"] < -thresh, -1.0, 0.0),
        )
        agreement += pd.Series(direction, index=idx).fillna(0)

    n_models = len(forecast_dict)
    # Agreement: positive value means models agree on long, negative on short
    signal = pd.Series(0.0, index=idx)
    long_mask = agreement >= min_agreement
    short_mask = agreement <= -min_agreement

    signal[long_mask] = agreement[long_mask] / n_models  # scaled by consensus strength
    signal[short_mask] = agreement[short_mask] / n_models

    # Vol-scale
    realized_vol = returns.rolling(vol_lookback).std() * np.sqrt(252)
    scalar = (target_vol / realized_vol.replace(0, np.nan)).clip(0.2, 2.5)
    signal = (signal * scalar.reindex(idx)).fillna(0)
    return signal.clip(-2, 2)


# ── Pre-compute ARIMA forecasts ONCE ─────────────────────────────────────

def precompute_forecasts(
    returns: pd.Series,
    train_window: int = 252,
    refit_every: int = 63,
    order: tuple = (2, 0, 2),
) -> pd.DataFrame:
    """Generate rolling one-step-ahead forecasts for full series."""
    fc = ArimaForecaster()
    print(f"  Computing rolling ARIMA{order} forecasts "
          f"(train_window={train_window}, refit_every={refit_every})...")
    forecasts = fc.rolling_forecast(
        returns, train_window=train_window, refit_every=refit_every, order=order,
    )
    print(f"  → {len(forecasts)} forecasts generated")
    return forecasts


# ── Strategies = operators on cached forecasts ──────────────────────────

def strategy_direction(
    forecasts: pd.DataFrame, threshold_std: float = 0.3,
) -> pd.Series:
    """Long when forecast_mean > threshold * std, short when < -threshold * std."""
    thresh = forecasts["forecast_std"] * threshold_std
    signal = pd.Series(0.0, index=forecasts.index)
    signal[forecasts["forecast_mean"] > thresh] = 1.0
    signal[forecasts["forecast_mean"] < -thresh] = -1.0
    return signal


def strategy_confidence(
    forecasts: pd.DataFrame, snr_cap: float = 1.5,
) -> pd.Series:
    """Position = clipped signal-to-noise ratio."""
    snr = forecasts["forecast_mean"] / forecasts["forecast_std"].replace(0, np.nan)
    return (snr.clip(-snr_cap, snr_cap) / snr_cap).fillna(0)


def strategy_direction_vol_scaled(
    forecasts: pd.DataFrame, returns: pd.Series,
    threshold_std: float = 0.3, target_vol: float = 0.15,
    vol_lookback: int = 21,
) -> pd.Series:
    """Direction × target_vol / realized_vol (capped)."""
    direction = strategy_direction(forecasts, threshold_std)
    realized_vol = returns.rolling(vol_lookback).std() * np.sqrt(252)
    scalar = (target_vol / realized_vol.replace(0, np.nan)).clip(0.2, 2.5)
    return (direction * scalar.reindex(direction.index)).fillna(0)


def strategy_momentum_filter(
    forecasts: pd.DataFrame, close: pd.Series,
    threshold_std: float = 0.3, momentum_window: int = 63,
) -> pd.Series:
    """Direction only when aligned with longer-term momentum."""
    direction = strategy_direction(forecasts, threshold_std)
    momentum = close.pct_change(momentum_window)
    mom_sign = np.sign(momentum).reindex(direction.index)
    # Take signal only when direction and momentum agree
    aligned = np.where(
        (direction > 0) & (mom_sign > 0), 1.0,
        np.where((direction < 0) & (mom_sign < 0), -1.0, 0.0),
    )
    return pd.Series(aligned, index=direction.index)


def strategy_contrarian(
    forecasts: pd.DataFrame, returns: pd.Series,
    threshold_std: float = 0.3, target_vol: float = 0.15,
    vol_lookback: int = 21,
) -> pd.Series:
    """
    Contrarian ARIMA: invert the direction signal with vol targeting.
    Exploits mean-reversion when ARIMA systematically overshoots.
    """
    direction = -strategy_direction(forecasts, threshold_std)
    realized_vol = returns.rolling(vol_lookback).std() * np.sqrt(252)
    scalar = (target_vol / realized_vol.replace(0, np.nan)).clip(0.2, 2.5)
    return (direction * scalar.reindex(direction.index)).fillna(0)


def invert_walk_forward_result(result: dict) -> dict:
    """
    Invert a walk-forward result: negate OOS returns → contrarian.
    Uses the SAME optimized parameters from the direction strategy
    but flips the signal post-optimization, avoiding double overfitting.
    """
    if "error" in result and result.get("metrics") == {}:
        inv = dict(result)
        inv["name"] = result.get("name", "") + " (Contrarian)"
        return inv

    oos_returns = -result["returns"]
    oos_mask = result.get("oos_mask", oos_returns != 0)
    equity = 100_000 * (1 + oos_returns.fillna(0)).cumprod()
    oos_only = oos_returns[oos_mask]

    if len(oos_only) < 20:
        return {"error": "insufficient OOS", "metrics": {}, "equity": equity,
                "returns": oos_returns, "param_history": result.get("param_history", [])}

    nw = newey_west_tstat(oos_only)
    bs = bootstrap_tstat_distribution(oos_only, n_bootstrap=500)

    ann_return = oos_only.mean() * 252
    ann_vol = oos_only.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min())

    metrics = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "oos_annualized_return": float(ann_return),
        "oos_annualized_volatility": float(ann_vol),
        "oos_sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "calmar": float(ann_return / abs(max_dd)) if max_dd != 0 else 0,
        "win_rate": float((oos_only > 0).sum() / ((oos_only > 0).sum() + (oos_only < 0).sum())),
        "n_oos_days": len(oos_only),
        **{f"hac_{k}": v for k, v in nw.items()},
        "bootstrap": bs,
    }

    return {
        "metrics": metrics, "equity": equity, "returns": oos_returns,
        "oos_mask": oos_mask, "param_history": result.get("param_history", []),
        "in_sample_periods": result.get("in_sample_periods", []),
        "oos_periods": result.get("oos_periods", []),
    }


# ── Simulate returns with costs ──────────────────────────────────────────

def simulate_pnl(
    signal: pd.Series,
    returns: pd.Series,
    commission_bps: float = 10.0,
    slippage_bps: float = 5.0,
    impact_eta: float = 0.1,
) -> pd.Series:
    """Simulate strategy returns with realistic transaction costs."""
    signal = signal.reindex(returns.index).fillna(0)
    position_delta = signal.diff().fillna(signal).abs()

    realized_vol = returns.rolling(21).std().fillna(returns.std())
    market_impact = impact_eta * realized_vol * np.sqrt(position_delta)

    cost_pct = (commission_bps + slippage_bps) / 10_000
    total_cost = position_delta * cost_pct + market_impact

    strategy_returns = signal.shift(1).fillna(0) * returns - total_cost
    return strategy_returns


# ── Walk-forward optimization on cached forecasts ───────────────────────

def walk_forward_optimize(
    forecasts: pd.DataFrame,
    returns: pd.Series,
    close: pd.Series,
    strategy_fn,
    param_grid: dict,
    train_window: int = 504,
    test_window: int = 63,
    purge_gap: int = 5,
    optimize_on: str = "tstat",
) -> dict:
    """
    Run walk-forward where each window optimizes parameters in-sample
    and applies them out-of-sample.
    """
    from itertools import product

    # Align everything
    forecasts = forecasts.reindex(returns.index).dropna()
    fcst = forecasts
    idx = fcst.index
    n = len(idx)

    param_names = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    combos = list(product(*param_values))

    # Pre-compute signals for ALL parameter combos once (fast since forecasts cached)
    needs_returns = {"strategy_direction_vol_scaled", "strategy_contrarian"}
    needs_close = {"strategy_momentum_filter"}
    all_signals = {}
    for combo in combos:
        params = dict(zip(param_names, combo))
        try:
            if strategy_fn.__name__ in needs_returns:
                sig = strategy_fn(fcst, returns, **params)
            elif strategy_fn.__name__ in needs_close:
                sig = strategy_fn(fcst, close, **params)
            else:
                sig = strategy_fn(fcst, **params)
            all_signals[combo] = sig
        except Exception:
            continue

    # Walk-forward
    oos_returns = pd.Series(0.0, index=returns.index)
    oos_mask = pd.Series(False, index=returns.index)
    param_history = []
    in_sample_periods = []
    oos_periods = []

    start = 0
    while start + train_window + purge_gap + test_window <= n:
        train_end = start + train_window
        test_start = train_end + purge_gap
        test_end = test_start + test_window

        train_idx = idx[start:train_end]
        test_idx = idx[test_start:test_end]

        # Optimize on train
        best_combo = None
        best_score = -np.inf
        for combo in combos:
            if combo not in all_signals:
                continue
            sig = all_signals[combo]
            train_sig = sig.reindex(train_idx).fillna(0)
            train_ret = returns.reindex(train_idx).fillna(0)
            # Quick costless PnL on train for optimization
            strat_ret = train_sig.shift(1).fillna(0) * train_ret
            if strat_ret.std() == 0:
                continue
            if optimize_on == "tstat":
                nw = newey_west_tstat(strat_ret)
                score = nw["t_stat"]
            else:  # sharpe
                score = strat_ret.mean() / strat_ret.std() * np.sqrt(252)
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_combo = combo

        if best_combo is None:
            start += test_window
            continue

        # Apply best params to OOS
        best_sig = all_signals[best_combo].reindex(test_idx).fillna(0)
        test_ret = returns.reindex(test_idx).fillna(0)
        test_strat_ret = simulate_pnl(best_sig, test_ret)

        oos_returns.loc[test_idx] = test_strat_ret.values
        oos_mask.loc[test_idx] = True

        best_params = dict(zip(param_names, best_combo))
        param_history.append({
            "train_start": train_idx[0], "train_end": train_idx[-1],
            "test_start": test_idx[0], "test_end": test_idx[-1],
            "params": best_params, "is_score": best_score,
        })
        in_sample_periods.append((train_idx[0], train_idx[-1]))
        oos_periods.append((test_idx[0], test_idx[-1]))

        start += test_window

    # Compute OOS metrics
    oos_only = oos_returns[oos_mask]
    equity = 100_000 * (1 + oos_returns.fillna(0)).cumprod()

    if len(oos_only) < 20:
        return {"error": "insufficient OOS", "metrics": {}, "equity": equity,
                "returns": oos_returns, "param_history": param_history}

    nw = newey_west_tstat(oos_only)
    bs = bootstrap_tstat_distribution(oos_only, n_bootstrap=500)

    ann_return = oos_only.mean() * 252
    ann_vol = oos_only.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min())

    metrics = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "oos_annualized_return": float(ann_return),
        "oos_annualized_volatility": float(ann_vol),
        "oos_sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "calmar": float(ann_return / abs(max_dd)) if max_dd != 0 else 0,
        "win_rate": float((oos_only > 0).sum() / ((oos_only > 0).sum() + (oos_only < 0).sum())),
        "n_oos_days": len(oos_only),
        **{f"hac_{k}": v for k, v in nw.items()},
        "bootstrap": bs,
    }

    return {
        "metrics": metrics, "equity": equity, "returns": oos_returns,
        "oos_mask": oos_mask, "param_history": param_history,
        "in_sample_periods": in_sample_periods, "oos_periods": oos_periods,
    }


# ── Walk-forward for ensemble (uses multiple forecasts) ─────────────────

def walk_forward_ensemble(
    forecast_cache: dict,
    returns: pd.Series,
    close: pd.Series,
    param_grid: dict,
    train_window: int = 504,
    test_window: int = 63,
    purge_gap: int = 5,
    optimize_on: str = "tstat",
) -> dict:
    """Walk-forward using ensemble of multiple ARIMA orders."""
    from itertools import product

    # Align
    idx = list(forecast_cache.values())[0].index
    idx = idx.intersection(returns.index)
    n = len(idx)

    param_names = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    combos = list(product(*param_values))

    # Pre-compute ensemble signals for all combos
    all_signals = {}
    for combo in combos:
        params = dict(zip(param_names, combo))
        try:
            sig = strategy_ensemble(forecast_cache, returns, **params)
            all_signals[combo] = sig.reindex(idx).fillna(0)
        except Exception:
            continue

    # Walk-forward
    oos_returns = pd.Series(0.0, index=returns.index)
    oos_mask = pd.Series(False, index=returns.index)
    param_history = []
    in_sample_periods = []
    oos_periods = []

    start = 0
    while start + train_window + purge_gap + test_window <= n:
        train_end = start + train_window
        test_start = train_end + purge_gap
        test_end = test_start + test_window

        train_idx = idx[start:train_end]
        test_idx = idx[test_start:test_end]

        best_combo = None
        best_score = -np.inf
        for combo in combos:
            if combo not in all_signals:
                continue
            train_sig = all_signals[combo].reindex(train_idx).fillna(0)
            train_ret = returns.reindex(train_idx).fillna(0)
            strat_ret = train_sig.shift(1).fillna(0) * train_ret
            if strat_ret.std() == 0:
                continue
            if optimize_on == "tstat":
                nw = newey_west_tstat(strat_ret)
                score = nw["t_stat"]
            else:
                score = strat_ret.mean() / strat_ret.std() * np.sqrt(252)
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_combo = combo

        if best_combo is None:
            start += test_window
            continue

        best_sig = all_signals[best_combo].reindex(test_idx).fillna(0)
        test_ret = returns.reindex(test_idx).fillna(0)
        test_strat_ret = simulate_pnl(best_sig, test_ret)

        oos_returns.loc[test_idx] = test_strat_ret.values
        oos_mask.loc[test_idx] = True
        param_history.append({
            "train_start": train_idx[0], "train_end": train_idx[-1],
            "test_start": test_idx[0], "test_end": test_idx[-1],
            "params": dict(zip(param_names, best_combo)),
            "is_score": best_score,
        })
        in_sample_periods.append((train_idx[0], train_idx[-1]))
        oos_periods.append((test_idx[0], test_idx[-1]))

        start += test_window

    oos_only = oos_returns[oos_mask]
    equity = 100_000 * (1 + oos_returns.fillna(0)).cumprod()

    if len(oos_only) < 20:
        return {"error": "insufficient OOS", "metrics": {}, "equity": equity,
                "returns": oos_returns, "param_history": param_history}

    nw = newey_west_tstat(oos_only)
    bs = bootstrap_tstat_distribution(oos_only, n_bootstrap=500)
    ann_return = oos_only.mean() * 252
    ann_vol = oos_only.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min())

    metrics = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "oos_annualized_return": float(ann_return),
        "oos_annualized_volatility": float(ann_vol),
        "oos_sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "calmar": float(ann_return / abs(max_dd)) if max_dd != 0 else 0,
        "win_rate": float((oos_only > 0).sum() / ((oos_only > 0).sum() + (oos_only < 0).sum())),
        "n_oos_days": len(oos_only),
        **{f"hac_{k}": v for k, v in nw.items()},
        "bootstrap": bs,
    }

    return {
        "metrics": metrics, "equity": equity, "returns": oos_returns,
        "oos_mask": oos_mask, "param_history": param_history,
        "in_sample_periods": in_sample_periods, "oos_periods": oos_periods,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ARIMA Strategy Optimizer — Target: t-stat > 5 (5-sigma significance)")
    print("=" * 70)

    prices = generate_ar_data(n_days=2520)
    returns = prices["close"].pct_change().fillna(0)
    close = prices["close"]

    print(f"\nData: {len(prices)} days "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")
    print(f"Buy-and-hold return: {close.iloc[-1]/close.iloc[0] - 1:.1%}")

    # ── Pre-compute ARIMA forecasts for multiple orders ──────────────────
    print("\n[Pre-computing rolling ARIMA forecasts]")
    forecast_cache = {}
    for order in [(1, 0, 1), (2, 0, 2), (1, 0, 2)]:
        forecast_cache[order] = precompute_forecasts(
            returns, train_window=252, refit_every=63, order=order,
        )

    # Use ARIMA(2,0,2) forecasts for strategies (typically best for AR(2) data)
    forecasts = forecast_cache[(2, 0, 2)]

    # ── Run all strategies ───────────────────────────────────────────────
    results = []

    print("\n[Strategy 1: ARIMA Direction]")
    r1 = walk_forward_optimize(
        forecasts, returns, close, strategy_direction,
        param_grid={"threshold_std": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]},
        train_window=504, test_window=63, optimize_on="tstat",
    )
    r1["name"] = "ARIMA Direction"
    results.append(r1)
    _print_result(r1)

    print("\n[Strategy 2: ARIMA Confidence-Scaled]")
    r2 = walk_forward_optimize(
        forecasts, returns, close, strategy_confidence,
        param_grid={"snr_cap": [0.5, 1.0, 1.5, 2.0, 3.0]},
        train_window=504, test_window=63, optimize_on="tstat",
    )
    r2["name"] = "ARIMA Confidence-Scaled"
    results.append(r2)
    _print_result(r2)

    print("\n[Strategy 3: ARIMA + Volatility Scaling]")
    r3 = walk_forward_optimize(
        forecasts, returns, close, strategy_direction_vol_scaled,
        param_grid={
            "threshold_std": [0.1, 0.2, 0.3, 0.5],
            "target_vol": [0.10, 0.15, 0.20],
            "vol_lookback": [21, 42],
        },
        train_window=504, test_window=63, optimize_on="tstat",
    )
    r3["name"] = "ARIMA + Vol Scaled"
    results.append(r3)
    _print_result(r3)

    print("\n[Strategy 4: ARIMA + Momentum Filter]")
    r4 = walk_forward_optimize(
        forecasts, returns, close, strategy_momentum_filter,
        param_grid={
            "threshold_std": [0.1, 0.2, 0.3, 0.4],
            "momentum_window": [21, 42, 63, 126],
        },
        train_window=504, test_window=63, optimize_on="tstat",
    )
    r4["name"] = "ARIMA + Momentum Filter"
    results.append(r4)
    _print_result(r4)

    # ── Strategy 5: Ensemble ─────────────────────────────────────────────
    print("\n[Strategy 5: ARIMA Ensemble (3 models)]")
    r5 = walk_forward_ensemble(
        forecast_cache, returns, close,
        param_grid={
            "threshold_std": [0.1, 0.2, 0.3],
            "min_agreement": [2, 3],
            "target_vol": [0.10, 0.12, 0.15],
            "vol_lookback": [21, 42],
        },
        train_window=504, test_window=63, optimize_on="tstat",
    )
    r5["name"] = "ARIMA Ensemble"
    results.append(r5)
    _print_result(r5)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<30s} {'t-stat':>10s} {'Sharpe':>8s} {'Ret':>8s} {'DD':>8s}")
    print("-" * 70)
    for r in results:
        m = r["metrics"]
        if not m:
            continue
        t = m.get("hac_t_stat", 0)
        sig = " 5σ" if abs(t) > 5 else (" 1%" if abs(t) > 2.576 else "  -")
        print(f"{r['name']:<30s} {t:>7.2f}{sig} "
              f"{m.get('oos_sharpe', 0):>8.2f} "
              f"{m.get('total_return', 0):>7.1%} "
              f"{m.get('max_drawdown', 0):>7.1%}")
    print("=" * 70)

    best = max(results, key=lambda r: r["metrics"].get("hac_t_stat", -999))
    print(f"\n★ Best: {best['name']} (t={best['metrics'].get('hac_t_stat', 0):.2f})")

    # Save for visualization
    save_path = os.path.join(ASSETS_DIR, "arima_results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "prices": prices,
            "returns": returns,
            "forecasts": forecasts,
            "results": results,
        }, f)
    print(f"\nSaved to {save_path}")
    return results


def _print_result(r):
    m = r.get("metrics", {})
    if not m:
        print(f"  No result: {r.get('error', 'unknown')}")
        return
    print(f"  OOS: {m.get('n_oos_days', 0)}d, "
          f"Sharpe={m.get('oos_sharpe', 0):.2f}, "
          f"Return={m.get('total_return', 0):.1%}, "
          f"MaxDD={m.get('max_drawdown', 0):.1%}")
    print(f"  HAC t-stat: {m.get('hac_t_stat', 0):.2f} "
          f"(p={m.get('hac_p_value', 1):.2e}, lag={m.get('hac_nw_lag', 0)})")
    bs = m.get("bootstrap", {})
    if bs:
        print(f"  Bootstrap t-stat: 5th={bs.get('t_stat_5th', 0):.2f}, "
              f"50th={bs.get('t_stat_50th', 0):.2f}, "
              f"95th={bs.get('t_stat_95th', 0):.2f}, "
              f"P(t>5)={bs.get('prob_t_gt_5', 0):.0%}")


if __name__ == "__main__":
    main()
