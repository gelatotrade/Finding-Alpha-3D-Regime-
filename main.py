#!/usr/bin/env python3
"""
Finding Alpha 3D Regime — Multi-Market Screener
================================================
Entry point for the multi-market alpha screener with:
- Prediction markets (Polymarket, Kalshi)
- Crypto oracles (Chainlink, Pyth) + CoinGecko OHLCV
- Stock markets (yfinance, Finnhub, Alpha Vantage)
- News API sentiment-driven alpha signals
- Backtesting engine with parameter sweeps
- Time-evolving animated 3D PnL surfaces
- Drift detection constraints
- Tail risk early warning system

Usage:
    python main.py scan          # Run a single full scan
    python main.py backtest      # Run backtest with 3D surfaces
    python main.py risk          # Run risk analysis only
    python main.py continuous    # Run continuous screener
    python main.py demo          # Run demo with synthetic data
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
    """Execute a single full market scan."""
    from screener.multi_market_screener import MultiMarketScreener
    screener = MultiMarketScreener()
    results = screener.full_scan()
    return results


def run_backtest():
    """Run backtests with parameter sweeps and generate 3D surfaces."""
    from screener.multi_market_screener import MultiMarketScreener
    screener = MultiMarketScreener()

    logger.info("Fetching stock data for backtesting...")
    stock_prices = screener.fetch_stock_prices()

    for symbol, prices in stock_prices.items():
        if prices.empty or len(prices) < 50:
            continue

        logger.info("Backtesting %s...", symbol)

        # Run strategy comparison
        for strat_name in ["momentum", "mean_reversion", "rsi"]:
            result = screener.run_strategy_backtest(prices, strat_name)
            m = result["metrics"]
            print(f"  {symbol} [{strat_name}]: Sharpe={m['sharpe_ratio']:.2f}, "
                  f"Return={m['total_return']:.1%}, MaxDD={m['max_drawdown']:.1%}")

        # Parameter sweep + 3D surface
        sweep = screener.run_parameter_sweep(prices)
        print(f"\n  Best params: {sweep['best_metrics']['params']}")
        print(f"  Best Sharpe: {sweep['best_metrics']['sharpe_ratio']:.2f}")

        if sweep["pnl_surface"] is not None:
            fig = screener.generate_3d_surface(sweep["pnl_surface"])
            output = f"pnl_surface_{symbol}.html"
            fig.write_html(output)
            print(f"  3D surface saved: {output}")

        # Time-evolving surface
        ts = screener.run_time_evolving_sweep(prices, n_periods=5)
        if ts:
            fig = screener.generate_animated_surface(ts)
            output = f"pnl_animated_{symbol}.html"
            fig.write_html(output)
            print(f"  Animated surface saved: {output}")


def run_risk():
    """Run risk analysis on portfolio."""
    from screener.multi_market_screener import MultiMarketScreener
    screener = MultiMarketScreener()

    stock_prices = screener.fetch_stock_prices()
    returns_dict = {}
    for sym, prices in stock_prices.items():
        if not prices.empty and "close" in prices.columns:
            returns_dict[sym] = prices["close"].pct_change().dropna()

    if not returns_dict:
        print("No data available for risk analysis")
        return

    combined = pd.DataFrame(returns_dict)
    portfolio_ret = combined.mean(axis=1)
    portfolio_eq = 100000 * (1 + portfolio_ret).cumprod()

    risk_result = screener.run_risk_scan(
        portfolio_ret, portfolio_eq, returns_dict
    )

    print("\n" + "=" * 60)
    print("RISK ANALYSIS REPORT")
    print("=" * 60)

    metrics = risk_result.get("risk_metrics", {})
    if metrics:
        print(f"\n  VaR (99%, historical): {metrics.get('var_99_hist', 0):.2%}")
        print(f"  VaR (99%, Cornish-Fisher): {metrics.get('var_99_cf', 0):.2%}")
        print(f"  CVaR (97.5%): {metrics.get('cvar_975', 0):.2%}")
        print(f"  Current Vol (21d ann.): {metrics.get('current_vol', 0):.1%}")
        print(f"  Kurtosis: {metrics.get('kurtosis', 0):.1f}")
        print(f"  Skewness: {metrics.get('skewness', 0):.2f}")
        print(f"  Worst Day: {metrics.get('worst_day', 0):.2%}")

    alerts = risk_result.get("drift_alerts", []) + risk_result.get("tail_risk_alerts", [])
    if alerts:
        print(f"\n  ALERTS ({len(alerts)}):")
        for a in alerts:
            print(f"    [{a.severity.upper()}] {a.message}")
    else:
        print("\n  No risk alerts. All clear.")

    # Risk dashboard visualization
    fig = screener.visualizer.plot_risk_dashboard(
        portfolio_ret, portfolio_eq, "Portfolio Risk Dashboard"
    )
    fig.write_html("risk_dashboard.html")
    print(f"\n  Risk dashboard saved: risk_dashboard.html")


def run_demo():
    """
    Run a demo with synthetic data to showcase all features
    without requiring API keys.
    """
    from backtesting.engine import BacktestEngine, Strategy
    from risk.drift_detector import DriftDetector
    from risk.tail_risk import TailRiskScreener
    from visualization.pnl_surfaces import PnLSurfaceVisualizer
    from screener.strategies import momentum_crossover_factory, bollinger_factory

    print("\n" + "=" * 60)
    print("DEMO MODE — Synthetic Data")
    print("=" * 60)

    # Generate synthetic price data
    np.random.seed(42)
    n_days = 252
    dates = pd.bdate_range("2025-01-01", periods=n_days)

    # Trending + mean-reverting price
    returns = np.random.normal(0.0005, 0.015, n_days)
    # Add regime change at day 180
    returns[180:] = np.random.normal(-0.001, 0.025, n_days - 180)
    prices = pd.DataFrame({
        "close": 100 * np.exp(np.cumsum(returns)),
        "high": 100 * np.exp(np.cumsum(returns)) * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
        "low": 100 * np.exp(np.cumsum(returns)) * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
        "open": 100 * np.exp(np.cumsum(returns)) * (1 + np.random.normal(0, 0.003, n_days)),
        "volume": np.random.randint(1_000_000, 10_000_000, n_days),
    }, index=dates)

    engine = BacktestEngine()
    viz = PnLSurfaceVisualizer()

    # 1. Backtest multiple strategies
    print("\n── BACKTEST RESULTS ──")
    strategies = {
        "Momentum 10/30": Strategy("Momentum 10/30", momentum_crossover_factory(fast=10, slow=30)),
        "Momentum 5/20": Strategy("Momentum 5/20", momentum_crossover_factory(fast=5, slow=20)),
        "Bollinger 20/2": Strategy("Bollinger 20/2", bollinger_factory(window=20, num_std=2.0)),
    }
    equity_curves = {}
    all_returns = {}

    for name, strat in strategies.items():
        result = engine.run(prices, strat)
        m = result.metrics
        print(f"  {name}: Sharpe={m['sharpe_ratio']:.2f}, "
              f"Return={m['total_return']:.1%}, MaxDD={m['max_drawdown']:.1%}")
        equity_curves[name] = result.equity_curve
        all_returns[name] = result.returns

    # 2. Parameter sweep → 3D surface
    print("\n── PARAMETER SWEEP ──")
    sweep = engine.parameter_sweep(
        prices,
        signal_fn_factory=momentum_crossover_factory,
        param_grid={"fast": [3, 5, 8, 10, 15, 20], "slow": [20, 25, 30, 40, 50, 60]},
    )
    print(f"  Best: {sweep.metrics['params']}, Sharpe={sweep.metrics['sharpe_ratio']:.2f}")

    fig = viz.plot_static_pnl_surface(
        sweep.pnl_by_param, "fast", "slow", "sharpe",
        "Momentum Strategy: Sharpe Ratio Surface",
    )
    fig.write_html("demo_pnl_surface.html")
    print("  Static 3D surface → demo_pnl_surface.html")

    # 3. Time-evolving surface
    print("\n── TIME-EVOLVING PNL SURFACE ──")
    n_periods = 6
    window = len(prices) // 2
    step = (len(prices) - window) // (n_periods - 1)
    time_surfaces = {}

    for i in range(n_periods):
        start = i * step
        end = start + window
        if end > len(prices):
            break
        wp = prices.iloc[start:end]
        label = str(wp.index[-1].date())
        try:
            res = engine.parameter_sweep(
                wp,
                signal_fn_factory=momentum_crossover_factory,
                param_grid={"fast": [5, 10, 15, 20], "slow": [25, 35, 45, 55]},
            )
            time_surfaces[label] = res.pnl_by_param
        except Exception:
            pass

    if time_surfaces:
        fig = viz.plot_animated_pnl_surface(
            time_surfaces, "fast", "slow", "sharpe",
            "Time-Evolving PnL Surface (Sharpe)",
        )
        fig.write_html("demo_animated_surface.html")
        print("  Animated surface → demo_animated_surface.html")

    # 4. Risk analysis
    print("\n── RISK ANALYSIS ──")
    portfolio_ret = pd.DataFrame(all_returns).mean(axis=1)
    portfolio_eq = 100000 * (1 + portfolio_ret).cumprod()

    drift = DriftDetector()
    drift_alerts = drift.full_drift_scan(portfolio_ret)

    tail = TailRiskScreener()
    tail_alerts = tail.full_tail_risk_scan(
        portfolio_ret, portfolio_eq, all_returns
    )

    risk_data = tail.get_risk_dashboard_data(portfolio_ret)
    print(f"  VaR (99%): {risk_data.get('var_99_hist', 0):.2%}")
    print(f"  CVaR (97.5%): {risk_data.get('cvar_975', 0):.2%}")
    print(f"  Current Vol: {risk_data.get('current_vol', 0):.1%}")
    print(f"  Kurtosis: {risk_data.get('kurtosis', 0):.1f}")

    total_alerts = len(drift_alerts) + len(tail_alerts)
    print(f"\n  Total Alerts: {total_alerts}")
    for a in (drift_alerts + tail_alerts)[:5]:
        sev = getattr(a, 'severity', 'unknown')
        msg = getattr(a, 'message', str(a))
        print(f"    [{sev.upper()}] {msg}")

    # 5. Equity comparison + risk dashboard
    fig = viz.plot_equity_comparison(equity_curves, "Strategy Equity Curves (Demo)")
    fig.write_html("demo_equity_curves.html")
    print("\n  Equity curves → demo_equity_curves.html")

    fig = viz.plot_risk_dashboard(portfolio_ret, portfolio_eq, "Risk Dashboard (Demo)")
    fig.write_html("demo_risk_dashboard.html")
    print("  Risk dashboard → demo_risk_dashboard.html")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE — Open the .html files in a browser to explore")
    print("=" * 60 + "\n")


def run_continuous():
    """Run continuous screener."""
    from screener.multi_market_screener import MultiMarketScreener
    screener = MultiMarketScreener()
    screener.run_continuous()


def main():
    parser = argparse.ArgumentParser(
        description="Finding Alpha 3D Regime — Multi-Market Screener"
    )
    parser.add_argument(
        "command",
        choices=["scan", "backtest", "risk", "continuous", "demo"],
        help="Command to execute",
    )
    args = parser.parse_args()

    commands = {
        "scan": run_scan,
        "backtest": run_backtest,
        "risk": run_risk,
        "continuous": run_continuous,
        "demo": run_demo,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
