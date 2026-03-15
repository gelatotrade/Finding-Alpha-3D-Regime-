#!/usr/bin/env python3
"""
Finding Alpha 3D Regime — Institutional Multi-Market Screener
=============================================================

Usage:
    python main.py scan          # Full multi-market scan
    python main.py backtest      # Backtest with 3D surfaces + Deflated Sharpe + PBO
    python main.py risk          # Risk analysis (VaR/CVaR/EVT, drift, tail risk)
    python main.py portfolio     # Portfolio optimization (risk parity, Kelly, MV)
    python main.py continuous    # Continuous screener
    python main.py demo          # Demo with synthetic data (no API keys needed)
"""
import argparse
import logging
import sys
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_scan():
    from screener.multi_market_screener import MultiMarketScreener
    screener = MultiMarketScreener()
    return screener.full_scan()


def run_backtest():
    from screener.multi_market_screener import MultiMarketScreener
    screener = MultiMarketScreener()

    logger.info("Fetching data for backtesting...")
    stock_prices = screener.fetch_stock_prices()

    for symbol, prices in stock_prices.items():
        if prices.empty or len(prices) < 50:
            continue

        print(f"\n{'='*50}")
        print(f"BACKTESTING: {symbol}")
        print(f"{'='*50}")

        for strat in ["momentum", "mean_reversion", "rsi", "regime_adaptive"]:
            result = screener.run_strategy_backtest(prices, strat)
            if not result:
                continue
            m = result["metrics"]
            pbo = result.get("pbo", {})
            mc = result.get("monte_carlo", {})

            print(f"\n  [{strat}]")
            print(f"    Sharpe={m['sharpe_ratio']:.2f}, Sortino={m['sortino_ratio']:.2f}, "
                  f"Calmar={m['calmar_ratio']:.2f}")
            print(f"    Return={m['total_return']:.1%}, MaxDD={m['max_drawdown']:.1%} "
                  f"({m['max_dd_duration_days']}d)")
            print(f"    WinRate={m['win_rate']:.0%}, PF={m['profit_factor']:.2f}, "
                  f"Trades={m['total_trades']}, Costs=${m['total_costs']:.0f}")
            print(f"    Skew={m['skewness']:.2f}, Kurt={m['kurtosis']:.1f}, "
                  f"TailRatio={m['tail_ratio']:.2f}")

            if pbo:
                print(f"    PBO={pbo.get('pbo', 0):.0%} "
                      f"(OOS Sharpe: mean={pbo.get('mean_oos_sharpe', 0):.2f}, "
                      f"median={pbo.get('median_oos_sharpe', 0):.2f})")
            if mc:
                print(f"    MC: P(profit)={mc.get('prob_positive_return', 0):.0%}, "
                      f"5th-95th return=[{mc.get('return_5th', 0):.1%}, "
                      f"{mc.get('return_95th', 0):.1%}]")

        # Parameter sweep + 3D surface
        sweep = screener.run_parameter_sweep(prices)
        if sweep:
            bm = sweep["best_metrics"]
            dsr = bm.get("deflated_sharpe", {})
            print(f"\n  [Parameter Sweep]")
            print(f"    Best: {bm['params']}, Sharpe={bm['sharpe_ratio']:.2f}")
            if dsr:
                print(f"    Deflated Sharpe: z={dsr.get('deflated_sharpe', 0):.2f}, "
                      f"p={dsr.get('p_value', 1):.4f}, "
                      f"{'SIGNIFICANT' if dsr.get('significant') else 'NOT significant'}")

            if sweep.get("pnl_surface") is not None:
                fig = screener.generate_3d_surface(sweep["pnl_surface"])
                out = f"pnl_surface_{symbol}.html"
                fig.write_html(out)
                print(f"    3D surface → {out}")

        ts = screener.run_time_evolving_sweep(prices, n_periods=5)
        if ts:
            fig = screener.generate_animated_surface(ts)
            out = f"pnl_animated_{symbol}.html"
            fig.write_html(out)
            print(f"    Animated surface → {out}")


def run_risk():
    from screener.multi_market_screener import MultiMarketScreener
    screener = MultiMarketScreener()

    stock_prices = screener.fetch_stock_prices()
    returns_dict = {}
    for sym, prices in stock_prices.items():
        if not prices.empty and "close" in prices.columns:
            returns_dict[sym] = prices["close"].pct_change().dropna()

    if not returns_dict:
        print("No data available")
        return

    combined = pd.DataFrame(returns_dict).dropna()
    portfolio_ret = combined.mean(axis=1)
    portfolio_eq = 100000 * (1 + portfolio_ret).cumprod()

    risk_result = screener.run_risk_scan(portfolio_ret, portfolio_eq, returns_dict)
    metrics = risk_result.get("risk_metrics", {})

    print("\n" + "=" * 60)
    print("INSTITUTIONAL RISK REPORT")
    print("=" * 60)

    if metrics:
        print(f"\n  VaR (99%)")
        print(f"    Historical:    {metrics.get('var_99_hist', 0):.2%}")
        print(f"    Parametric:    {metrics.get('var_99_param', 0):.2%}")
        print(f"    Cornish-Fisher:{metrics.get('var_99_cf', 0):.2%}")
        print(f"    EVT (GPD):     {metrics.get('var_99_evt', 0):.2%}")
        print(f"    CVaR (97.5%):  {metrics.get('cvar_975', 0):.2%}")
        print(f"\n  Volatility")
        print(f"    EWMA (λ=0.94): {metrics.get('ewma_vol', 0):.1%}")
        print(f"    21d Realized:  {metrics.get('rolling_vol_21d', 0):.1%}")
        print(f"    63d Realized:  {metrics.get('rolling_vol_63d', 0):.1%}")
        print(f"\n  Tail Risk")
        print(f"    Kurtosis:      {metrics.get('kurtosis', 0):.1f}")
        print(f"    Skewness:      {metrics.get('skewness', 0):.2f}")
        print(f"    Hill α:        {metrics.get('hill_tail_index', 0):.1f}")
        print(f"    Worst day:     {metrics.get('worst_day', 0):.2%}")

    alerts = risk_result.get("drift_alerts", []) + risk_result.get("tail_risk_alerts", [])
    if alerts:
        print(f"\n  ALERTS ({len(alerts)})")
        for a in alerts:
            print(f"    [{a.severity.upper()}] {a.message}")
    else:
        print("\n  No risk alerts.")

    fig = screener.visualizer.plot_risk_dashboard(
        portfolio_ret, portfolio_eq, metrics, "Institutional Risk Dashboard"
    )
    fig.write_html("risk_dashboard.html")
    print(f"\n  Dashboard → risk_dashboard.html")


def run_portfolio():
    from screener.multi_market_screener import MultiMarketScreener
    screener = MultiMarketScreener()

    stock_prices = screener.fetch_stock_prices()
    returns_dict = {}
    for sym, prices in stock_prices.items():
        if not prices.empty and "close" in prices.columns:
            returns_dict[sym] = prices["close"].pct_change().dropna()

    if len(returns_dict) < 2:
        print("Need >= 2 assets for portfolio optimization")
        return

    print("\n" + "=" * 60)
    print("PORTFOLIO OPTIMIZATION")
    print("=" * 60)

    for method in ["risk_parity", "kelly", "mean_variance", "max_diversification"]:
        weights = screener.construct_portfolio(returns_dict, method)
        if weights:
            sorted_w = sorted(weights.items(), key=lambda x: -x[1])
            print(f"\n  [{method}]")
            for asset, w in sorted_w:
                print(f"    {asset:6s}: {w:6.1%}")


def run_demo():
    from backtesting.engine import BacktestEngine, Strategy
    from risk.drift_detector import DriftDetector
    from risk.tail_risk import TailRiskScreener
    from risk.portfolio import PortfolioConstructor
    from visualization.pnl_surfaces import PnLSurfaceVisualizer
    from screener.strategies import (
        momentum_crossover_factory, bollinger_factory, regime_adaptive,
    )

    print("\n" + "=" * 70)
    print("INSTITUTIONAL DEMO — Synthetic Data")
    print("=" * 70)

    # Generate realistic synthetic data with regime change
    np.random.seed(42)
    n_days = 504  # 2 years
    dates = pd.bdate_range("2024-01-01", periods=n_days)

    # Multi-asset with correlations
    n_assets = 4
    asset_names = ["ALPHA", "BETA", "GAMMA", "DELTA"]

    # Correlated returns via Cholesky
    corr = np.array([
        [1.0, 0.6, 0.3, 0.1],
        [0.6, 1.0, 0.4, 0.2],
        [0.3, 0.4, 1.0, 0.5],
        [0.1, 0.2, 0.5, 1.0],
    ])
    vols = np.array([0.015, 0.020, 0.018, 0.012])
    cov = np.outer(vols, vols) * corr
    L = np.linalg.cholesky(cov)

    all_returns = {}
    all_prices = {}

    for i, name in enumerate(asset_names):
        z = np.random.randn(n_days, n_assets)
        corr_z = (z @ L.T)[:, i]

        # Regime: trending first 300d, volatile last 200d
        returns = np.zeros(n_days)
        returns[:300] = 0.0004 + corr_z[:300]
        returns[300:] = -0.0002 + corr_z[300:] * 1.8  # vol spike

        prices = pd.DataFrame({
            "close": 100 * np.exp(np.cumsum(returns)),
            "high": 100 * np.exp(np.cumsum(returns)) * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
            "low": 100 * np.exp(np.cumsum(returns)) * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
            "open": 100 * np.exp(np.cumsum(returns)) * (1 + np.random.normal(0, 0.003, n_days)),
            "volume": np.random.randint(1_000_000, 10_000_000, n_days),
        }, index=dates)

        all_returns[name] = pd.Series(returns, index=dates)
        all_prices[name] = prices

    engine = BacktestEngine()
    viz = PnLSurfaceVisualizer()
    portfolio = PortfolioConstructor()

    # 1. Backtest strategies
    print("\n── BACKTEST RESULTS ──")
    strategies = {
        "Momentum 10/30": Strategy("Momentum 10/30", momentum_crossover_factory(10, 30)),
        "Bollinger 20/2": Strategy("Bollinger 20/2", bollinger_factory(20, 2.0)),
        "Regime Adaptive": Strategy("Regime Adaptive", regime_adaptive()),
    }

    equity_curves = {}
    strategy_returns = {}

    for name, strat in strategies.items():
        prices = all_prices["ALPHA"]
        result = engine.run(prices, strat)
        m = result.metrics
        mc = engine.monte_carlo_simulation(result.returns)

        print(f"\n  {name}:")
        print(f"    Sharpe={m['sharpe_ratio']:.2f}, Sortino={m['sortino_ratio']:.2f}, "
              f"Calmar={m['calmar_ratio']:.2f}")
        print(f"    Return={m['total_return']:.1%}, MaxDD={m['max_drawdown']:.1%} "
              f"({m['max_dd_duration_days']}d)")
        print(f"    PF={m['profit_factor']:.2f}, TailRatio={m['tail_ratio']:.2f}, "
              f"Costs=${m['total_costs']:.0f}")
        print(f"    MC: P(profit)={mc.get('prob_positive_return', 0):.0%}, "
              f"5th-95th=[{mc.get('return_5th', 0):.1%}, {mc.get('return_95th', 0):.1%}]")

        # PBO
        pbo = engine.combinatorial_purged_cv(prices, strat)
        print(f"    PBO={pbo['pbo']:.0%}, OOS Sharpe: "
              f"mean={pbo['mean_oos_sharpe']:.2f}, "
              f"median={pbo['median_oos_sharpe']:.2f}")

        equity_curves[name] = result.equity_curve
        strategy_returns[name] = result.returns

    # 2. Parameter sweep + 3D surface
    print("\n── PARAMETER SWEEP + DEFLATED SHARPE ──")
    sweep = engine.parameter_sweep(
        all_prices["ALPHA"],
        signal_fn_factory=momentum_crossover_factory,
        param_grid={"fast": [3, 5, 8, 10, 15, 20], "slow": [20, 25, 30, 40, 50, 60]},
    )
    bm = sweep.metrics
    dsr = bm.get("deflated_sharpe", {})
    print(f"  Best: {bm['params']}, Sharpe={bm['sharpe_ratio']:.2f}")
    if dsr:
        print(f"  Deflated Sharpe: z={dsr.get('deflated_sharpe', 0):.2f}, "
              f"p={dsr.get('p_value', 1):.4f}, "
              f"{'SIGNIFICANT' if dsr.get('significant') else 'NOT significant'}")
        print(f"  E[max SR under null]={dsr.get('e_max_sharpe', 0):.2f} "
              f"({dsr.get('n_trials', 0)} trials)")

    fig = viz.plot_static_pnl_surface(
        sweep.pnl_by_param, "fast", "slow", "sharpe",
        "Momentum Sharpe Surface (Deflated Sharpe corrected)",
    )
    fig.write_html("demo_pnl_surface.html")
    print("  3D surface → demo_pnl_surface.html")

    # 3. Time-evolving surface
    print("\n── TIME-EVOLVING PNL SURFACE ──")
    n_periods = 6
    window = len(all_prices["ALPHA"]) // 2
    step = (len(all_prices["ALPHA"]) - window) // (n_periods - 1)
    time_surfaces = {}

    for i in range(n_periods):
        start = i * step
        end = start + window
        if end > len(all_prices["ALPHA"]):
            break
        wp = all_prices["ALPHA"].iloc[start:end]
        label = str(wp.index[-1].date())
        try:
            res = engine.parameter_sweep(
                wp, signal_fn_factory=momentum_crossover_factory,
                param_grid={"fast": [5, 10, 15, 20], "slow": [25, 35, 45, 55]},
            )
            time_surfaces[label] = res.pnl_by_param
        except ValueError:
            pass

    if time_surfaces:
        fig = viz.plot_animated_pnl_surface(
            time_surfaces, "fast", "slow", "sharpe",
            "Time-Evolving PnL Surface",
        )
        fig.write_html("demo_animated_surface.html")
        print("  Animated surface → demo_animated_surface.html")

    # 4. Portfolio optimization
    print("\n── PORTFOLIO OPTIMIZATION ──")
    ret_df = pd.DataFrame(all_returns)
    for method in ["risk_parity", "kelly", "mean_variance", "max_diversification"]:
        weights = portfolio.optimize(ret_df, method=method)
        sorted_w = sorted(weights.items(), key=lambda x: -x[1])
        print(f"  [{method}]: " + ", ".join(f"{k}={v:.1%}" for k, v in sorted_w))

    # 5. Risk analysis
    print("\n── RISK ANALYSIS ──")
    portfolio_ret = ret_df.mean(axis=1)
    portfolio_eq = 100000 * (1 + portfolio_ret).cumprod()

    drift = DriftDetector()
    drift_alerts = drift.full_drift_scan(portfolio_ret, corr_data=ret_df)

    tail = TailRiskScreener()
    tail_alerts = tail.full_tail_risk_scan(
        portfolio_ret, portfolio_eq,
        {k: v for k, v in all_returns.items()},
    )

    risk_data = tail.get_risk_dashboard_data(portfolio_ret)
    print(f"  VaR(99%): hist={risk_data.get('var_99_hist', 0):.2%}, "
          f"EVT={risk_data.get('var_99_evt', 0):.2%}")
    print(f"  CVaR(97.5%): {risk_data.get('cvar_975', 0):.2%}")
    print(f"  EWMA Vol: {risk_data.get('ewma_vol', 0):.1%}")
    print(f"  Tail: kurt={risk_data.get('kurtosis', 0):.1f}, "
          f"skew={risk_data.get('skewness', 0):.2f}, "
          f"Hill α={risk_data.get('hill_tail_index', 0):.1f}")

    total_alerts = len(drift_alerts) + len(tail_alerts)
    print(f"\n  Alerts: {total_alerts}")
    for a in (drift_alerts + tail_alerts)[:5]:
        sev = getattr(a, 'severity', 'unknown')
        msg = getattr(a, 'message', str(a))
        print(f"    [{sev.upper()}] {msg}")

    # 6. Visualizations
    fig = viz.plot_equity_comparison(equity_curves, "Strategy Equity Curves")
    fig.write_html("demo_equity_curves.html")

    fig = viz.plot_risk_dashboard(portfolio_ret, portfolio_eq, risk_data,
                                  "Institutional Risk Dashboard")
    fig.write_html("demo_risk_dashboard.html")

    print("\n  Equity curves → demo_equity_curves.html")
    print("  Risk dashboard → demo_risk_dashboard.html")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE — Open .html files in browser to explore")
    print("=" * 70 + "\n")


def run_continuous():
    from screener.multi_market_screener import MultiMarketScreener
    screener = MultiMarketScreener()
    screener.run_continuous()


def main():
    parser = argparse.ArgumentParser(
        description="Finding Alpha 3D Regime — Institutional Multi-Market Screener"
    )
    parser.add_argument(
        "command",
        choices=["scan", "backtest", "risk", "portfolio", "continuous", "demo"],
        help="Command to execute",
    )
    args = parser.parse_args()

    commands = {
        "scan": run_scan,
        "backtest": run_backtest,
        "risk": run_risk,
        "portfolio": run_portfolio,
        "continuous": run_continuous,
        "demo": run_demo,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
