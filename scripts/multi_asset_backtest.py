"""
Multi-Asset ARIMA Backtest — 10 assets across different categories.

Tests the ARIMA walk-forward engine on realistic synthetic data calibrated
to each asset class's statistical properties: volatility, autocorrelation
structure, regime dynamics, and mean return.

Asset Categories:
  Stocks:      AAPL, MSFT, TSLA
  Crypto:      BTC, ETH
  Indices:     SPY, QQQ
  Commodities: GLD (Gold)
  Bonds:       TLT (Long Treasury)
  Forex:       EURUSD
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
from scripts.arima_optimizer import (
    precompute_forecasts, strategy_direction, strategy_confidence,
    strategy_direction_vol_scaled, strategy_momentum_filter,
    strategy_contrarian, strategy_ensemble, simulate_pnl,
    walk_forward_optimize, walk_forward_ensemble,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
)
os.makedirs(ASSETS_DIR, exist_ok=True)


# ── Asset data generators calibrated to real-world statistics ─────────────

ASSET_CONFIGS = {
    # ── Stocks ────────────────────────────────────────────────────────────
    "AAPL": {
        "category": "Stocks",
        "n_days": 1260,
        "ar1_schedule": [0.08, 0.12, 0.05, 0.15, 0.10, 0.07],
        "ar2_schedule": [0.03, 0.05, 0.02, 0.06, 0.04, 0.03],
        "mean_schedule": [0.0005, 0.0007, -0.0002, 0.0006, 0.0004, 0.0005],
        "vol_schedule": [0.015, 0.018, 0.028, 0.016, 0.020, 0.017],
        "seed": 100,
    },
    "MSFT": {
        "category": "Stocks",
        "n_days": 1260,
        "ar1_schedule": [0.06, 0.10, 0.04, 0.12, 0.08, 0.06],
        "ar2_schedule": [0.02, 0.04, 0.01, 0.05, 0.03, 0.02],
        "mean_schedule": [0.0004, 0.0006, -0.0001, 0.0005, 0.0003, 0.0004],
        "vol_schedule": [0.014, 0.016, 0.025, 0.015, 0.018, 0.015],
        "seed": 101,
    },
    "TSLA": {
        "category": "Stocks",
        "n_days": 1260,
        "ar1_schedule": [0.10, 0.18, 0.06, 0.20, 0.14, 0.09],
        "ar2_schedule": [0.04, 0.08, 0.02, 0.09, 0.06, 0.04],
        "mean_schedule": [0.0008, 0.0012, -0.0005, 0.0010, 0.0006, 0.0007],
        "vol_schedule": [0.030, 0.040, 0.055, 0.035, 0.042, 0.032],
        "seed": 102,
    },
    # ── Crypto ────────────────────────────────────────────────────────────
    "BTC": {
        "category": "Crypto",
        "n_days": 1260,
        "ar1_schedule": [0.05, 0.10, 0.03, 0.12, 0.08, 0.06],
        "ar2_schedule": [0.02, 0.04, 0.01, 0.05, 0.03, 0.02],
        "mean_schedule": [0.0006, 0.0010, -0.0004, 0.0008, 0.0005, 0.0006],
        "vol_schedule": [0.035, 0.045, 0.060, 0.038, 0.050, 0.040],
        "seed": 200,
    },
    "ETH": {
        "category": "Crypto",
        "n_days": 1260,
        "ar1_schedule": [0.07, 0.14, 0.04, 0.16, 0.10, 0.08],
        "ar2_schedule": [0.03, 0.06, 0.01, 0.07, 0.04, 0.03],
        "mean_schedule": [0.0007, 0.0012, -0.0006, 0.0010, 0.0006, 0.0007],
        "vol_schedule": [0.042, 0.055, 0.075, 0.048, 0.060, 0.045],
        "seed": 201,
    },
    # ── Indices / ETFs ────────────────────────────────────────────────────
    "SPY": {
        "category": "Indices",
        "n_days": 1260,
        "ar1_schedule": [0.05, 0.08, 0.03, 0.10, 0.06, 0.05],
        "ar2_schedule": [0.02, 0.03, 0.01, 0.04, 0.02, 0.02],
        "mean_schedule": [0.0003, 0.0004, -0.0001, 0.0004, 0.0003, 0.0003],
        "vol_schedule": [0.010, 0.012, 0.022, 0.011, 0.014, 0.011],
        "seed": 300,
    },
    "QQQ": {
        "category": "Indices",
        "n_days": 1260,
        "ar1_schedule": [0.06, 0.10, 0.04, 0.12, 0.07, 0.06],
        "ar2_schedule": [0.02, 0.04, 0.01, 0.05, 0.03, 0.02],
        "mean_schedule": [0.0004, 0.0005, -0.0002, 0.0005, 0.0004, 0.0004],
        "vol_schedule": [0.013, 0.015, 0.026, 0.014, 0.018, 0.014],
        "seed": 301,
    },
    # ── Commodities ───────────────────────────────────────────────────────
    "GLD": {
        "category": "Commodities",
        "n_days": 1260,
        "ar1_schedule": [0.04, 0.06, 0.02, 0.08, 0.05, 0.04],
        "ar2_schedule": [0.01, 0.02, 0.005, 0.03, 0.02, 0.01],
        "mean_schedule": [0.0002, 0.0003, 0.0001, 0.0003, 0.0002, 0.0002],
        "vol_schedule": [0.009, 0.011, 0.016, 0.010, 0.013, 0.010],
        "seed": 400,
    },
    # ── Bonds ─────────────────────────────────────────────────────────────
    "TLT": {
        "category": "Bonds",
        "n_days": 1260,
        "ar1_schedule": [0.03, 0.05, 0.02, 0.06, 0.04, 0.03],
        "ar2_schedule": [0.01, 0.02, 0.005, 0.02, 0.01, 0.01],
        "mean_schedule": [0.0001, 0.0002, 0.0001, 0.0002, 0.0001, 0.0001],
        "vol_schedule": [0.008, 0.010, 0.015, 0.009, 0.012, 0.009],
        "seed": 500,
    },
    # ── Forex ─────────────────────────────────────────────────────────────
    "EURUSD": {
        "category": "Forex",
        "n_days": 1260,
        "ar1_schedule": [0.03, 0.05, 0.02, 0.07, 0.04, 0.03],
        "ar2_schedule": [0.01, 0.02, 0.005, 0.03, 0.02, 0.01],
        "mean_schedule": [0.0000, 0.0001, -0.0001, 0.0001, 0.0000, 0.0000],
        "vol_schedule": [0.005, 0.006, 0.010, 0.006, 0.007, 0.006],
        "seed": 600,
    },
}


def generate_asset_data(config: dict) -> pd.DataFrame:
    """Generate synthetic OHLCV data calibrated to asset-class statistics."""
    np.random.seed(config["seed"])
    n_days = config["n_days"]
    dates = pd.bdate_range("2016-01-01", periods=n_days)

    returns = np.zeros(n_days)
    returns[0] = np.random.normal(0, config["vol_schedule"][0])
    returns[1] = np.random.normal(0, config["vol_schedule"][0])

    regime_length = 250
    for i in range(2, n_days):
        regime = min(i // regime_length, len(config["ar1_schedule"]) - 1)
        ar1 = config["ar1_schedule"][regime]
        ar2 = config["ar2_schedule"][regime]
        mu = config["mean_schedule"][regime]
        sigma = config["vol_schedule"][regime]
        returns[i] = (ar1 * returns[i - 1] + ar2 * returns[i - 2]
                      + np.random.normal(mu, sigma))

    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "close": close,
        "high": close * (1 + np.abs(np.random.normal(0, 0.004, n_days))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.004, n_days))),
        "open": close * (1 + np.random.normal(0, 0.002, n_days)),
        "volume": np.random.randint(1_000_000, 10_000_000, n_days),
    }, index=dates)


def backtest_single_asset(
    asset_name: str,
    config: dict,
    verbose: bool = True,
) -> dict:
    """Run full ARIMA walk-forward backtest on a single asset."""
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  {asset_name} ({config['category']})")
        print(f"{'─' * 60}")

    prices = generate_asset_data(config)
    returns = prices["close"].pct_change().fillna(0)
    close = prices["close"]
    bh_return = close.iloc[-1] / close.iloc[0] - 1

    if verbose:
        ann_vol = returns.std() * np.sqrt(252)
        print(f"  Data: {len(prices)} days, "
              f"ann. vol={ann_vol:.1%}, B&H={bh_return:+.1%}")

    # Pre-compute ARIMA forecasts (fixed order, no auto-selection for speed)
    if verbose:
        print("  Computing ARIMA forecasts...")
    forecast_cache = {}
    for order in [(1, 0, 1), (2, 0, 2)]:
        forecast_cache[order] = precompute_forecasts(
            returns, train_window=252, refit_every=126, order=order,
        )
    forecasts = forecast_cache[(2, 0, 2)]

    if verbose:
        print(f"  Running 4 strategies + ensemble...")

    results = {}

    # Strategy 1: ARIMA Direction
    r1 = walk_forward_optimize(
        forecasts, returns, close, strategy_direction,
        param_grid={"threshold_std": [0.1, 0.2, 0.3, 0.5, 0.8]},
        train_window=252, test_window=42, optimize_on="tstat",
    )
    r1["name"] = "Direction"
    results["Direction"] = r1

    # Strategy 2: ARIMA + Vol Scaling
    r2 = walk_forward_optimize(
        forecasts, returns, close, strategy_direction_vol_scaled,
        param_grid={
            "threshold_std": [0.1, 0.2, 0.3, 0.5],
            "target_vol": [0.10, 0.15, 0.20],
            "vol_lookback": [21, 42],
        },
        train_window=252, test_window=42, optimize_on="tstat",
    )
    r2["name"] = "Vol-Scaled"
    results["Vol-Scaled"] = r2

    # Strategy 3: ARIMA + Momentum Filter
    r3 = walk_forward_optimize(
        forecasts, returns, close, strategy_momentum_filter,
        param_grid={
            "threshold_std": [0.1, 0.2, 0.3, 0.4],
            "momentum_window": [21, 42, 63, 126],
        },
        train_window=252, test_window=42, optimize_on="tstat",
    )
    r3["name"] = "Momentum"
    results["Momentum"] = r3

    # Strategy 4: Contrarian (inverted direction, vol-scaled)
    r_c = walk_forward_optimize(
        forecasts, returns, close, strategy_contrarian,
        param_grid={
            "threshold_std": [0.1, 0.2, 0.3, 0.5],
            "target_vol": [0.10, 0.15, 0.20],
            "vol_lookback": [21, 42],
        },
        train_window=252, test_window=42, optimize_on="tstat",
    )
    r_c["name"] = "Contrarian"
    results["Contrarian"] = r_c

    # Strategy 5: Ensemble
    r4 = walk_forward_ensemble(
        forecast_cache, returns, close,
        param_grid={
            "threshold_std": [0.1, 0.2, 0.3],
            "min_agreement": [2],
            "target_vol": [0.12, 0.18],
            "vol_lookback": [21],
        },
        train_window=252, test_window=42, optimize_on="tstat",
    )
    r4["name"] = "Ensemble"
    results["Ensemble"] = r4

    # Pick best strategy by t-stat
    best_name = max(
        results, key=lambda k: results[k].get("metrics", {}).get("hac_t_stat", -999)
    )
    best = results[best_name]
    best_m = best.get("metrics", {})

    if verbose:
        print(f"\n  {'Strategy':<16s} {'t-stat':>8s} {'Sharpe':>8s} "
              f"{'Return':>9s} {'MaxDD':>8s} {'WinR':>6s}")
        print(f"  {'-' * 56}")
        for name, r in results.items():
            m = r.get("metrics", {})
            if not m:
                print(f"  {name:<16s} {'N/A':>8s}")
                continue
            t = m.get("hac_t_stat", 0)
            marker = " *" if name == best_name else ""
            print(f"  {name:<16s} {t:>8.2f} {m.get('oos_sharpe', 0):>8.2f} "
                  f"{m.get('total_return', 0):>8.1%} "
                  f"{m.get('max_drawdown', 0):>8.1%} "
                  f"{m.get('win_rate', 0):>5.1%}{marker}")

        if best_m:
            sig = "5-sigma" if abs(best_m.get("hac_t_stat", 0)) > 5 else \
                  "1% sig." if abs(best_m.get("hac_t_stat", 0)) > 2.576 else \
                  "5% sig." if abs(best_m.get("hac_t_stat", 0)) > 1.96 else \
                  "not sig."
            print(f"\n  Best: {best_name} → t={best_m.get('hac_t_stat', 0):.2f} ({sig})")

    return {
        "asset": asset_name,
        "category": config["category"],
        "prices": prices,
        "returns": returns,
        "buy_hold_return": bh_return,
        "results": results,
        "best_strategy": best_name,
        "best_metrics": best_m,
        "forecasts": forecasts,
        "forecast_cache": forecast_cache,
    }


def main():
    print("=" * 70)
    print("Multi-Asset ARIMA Backtest — 10 Assets × 4 Strategies")
    print("Walk-forward with Newey-West HAC t-statistic")
    print("=" * 70)

    all_results = {}
    for asset_name, config in ASSET_CONFIGS.items():
        result = backtest_single_asset(asset_name, config)
        all_results[asset_name] = result

    # ── Cross-Asset Summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-ASSET SUMMARY")
    print("=" * 70)
    print(f"\n{'Asset':<10s} {'Category':<12s} {'Best Strategy':<16s} "
          f"{'t-stat':>8s} {'Sharpe':>8s} {'Return':>9s} {'MaxDD':>8s} {'Sig':>8s}")
    print("-" * 82)

    for asset_name, result in all_results.items():
        m = result["best_metrics"]
        if not m:
            print(f"{asset_name:<10s} {result['category']:<12s} {'N/A':<16s}")
            continue
        t = m.get("hac_t_stat", 0)
        sig = "★ 5σ" if abs(t) > 5 else \
              "✓ 1%" if abs(t) > 2.576 else \
              "~ 5%" if abs(t) > 1.96 else \
              "✗ n.s."
        print(f"{asset_name:<10s} {result['category']:<12s} "
              f"{result['best_strategy']:<16s} "
              f"{t:>8.2f} {m.get('oos_sharpe', 0):>8.2f} "
              f"{m.get('total_return', 0):>8.1%} "
              f"{m.get('max_drawdown', 0):>8.1%} {sig:>8s}")

    # Category-level aggregation
    print(f"\n{'Category':<12s} {'Avg t-stat':>10s} {'Avg Sharpe':>10s} "
          f"{'Avg Return':>10s} {'Assets':>8s}")
    print("-" * 52)
    categories = {}
    for asset_name, result in all_results.items():
        m = result["best_metrics"]
        if not m:
            continue
        cat = result["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(m)

    for cat, metrics_list in categories.items():
        avg_t = np.mean([m.get("hac_t_stat", 0) for m in metrics_list])
        avg_s = np.mean([m.get("oos_sharpe", 0) for m in metrics_list])
        avg_r = np.mean([m.get("total_return", 0) for m in metrics_list])
        print(f"{cat:<12s} {avg_t:>10.2f} {avg_s:>10.2f} "
              f"{avg_r:>9.1%} {len(metrics_list):>8d}")

    print("=" * 70)

    # Save results
    save_data = {}
    for asset_name, result in all_results.items():
        save_data[asset_name] = {
            "asset": result["asset"],
            "category": result["category"],
            "prices": result["prices"],
            "returns": result["returns"],
            "buy_hold_return": result["buy_hold_return"],
            "results": result["results"],
            "best_strategy": result["best_strategy"],
            "best_metrics": result["best_metrics"],
        }

    save_path = os.path.join(ASSETS_DIR, "multi_asset_results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"\nSaved to {save_path}")

    return all_results


if __name__ == "__main__":
    main()
