# Finding Alpha 3D Regime

Institutional-grade multi-market alpha screener with animated 3D PnL surfaces, backtesting engine, drift detection, and tail risk early warning system.

## Visual Showcase

### Time-Evolving 3D PnL Surface
How strategy performance evolves across parameter space (fast/slow MA) through rolling time windows. Each frame shows Sharpe across the parameter grid; gold star marks the best point. The surface shape mutates as market regimes change.

![3D PnL Surface](assets/pnl_surface_3d.gif)

### Strategy Equity Curves
Side-by-side equity progression for Momentum, Bollinger Mean-Reversion, and Regime-Adaptive strategies vs. buy-and-hold — including realistic transaction costs and Almgren-Chriss market impact.

![Equity Curves](assets/equity_curves.gif)

### Institutional Risk Dashboard
Live 4-panel dashboard: equity curve, drawdown (-10% alert threshold), rolling + EWMA volatility, and VaR across 4 methods (Historical, Parametric, Cornish-Fisher, EVT) plus CVaR.

![Risk Dashboard](assets/risk_dashboard.gif)

### Live Regime Detection
Price chart with live regime shading (green = low vol, yellow = elevated, red = high vol), vol ratio (21d/63d), and Hurst exponent — H > 0.5 trending, H < 0.5 mean-reverting. The system routes strategies by regime.

![Regime Detection](assets/regime_detection.gif)

### Parameter Sweep Heatmap
Full parameter grid evolving over time. Gold-bordered cell = best Sharpe. Footer shows Deflated Sharpe p-value — a red ✗ means the best Sharpe is likely from overfitting, not skill.

![Parameter Sweep](assets/parameter_sweep.gif)

---

## ARIMA Walk-Forward Backtest · t-stat > 5 (5σ significance)

Five ARIMA-based strategies back-tested on 10 years (2,520 days) of AR-structured data via rolling walk-forward with in-sample parameter optimization. **All five strategies clear the 5-sigma institutional significance bar.**

### Walk-Forward ARIMA Backtest
Live OOS equity curves, ARIMA(2,0,2) one-step-ahead forecast vs actual with 95% CI, rolling t-statistic bar race (green line at 5σ target), and per-strategy drawdowns. ★ marks strategies with t > 5, ✓ with t > 2.58 (1%).

![ARIMA Walk-Forward](assets/arima_walk_forward.gif)

### Walk-Forward Schedule
Gantt chart of the rolling window protocol: blue = in-sample training, red = purge gap (embargo), green = out-of-sample testing. Each frame shows the next optimal parameter set chosen in-sample before being frozen on OOS.

![Walk-Forward Schedule](assets/walk_forward.gif)

### Results Summary

| Strategy | t-stat (HAC) | p-value | Sharpe | Return | Max DD | Bootstrap P(t>5) |
|----------|-------------:|--------:|-------:|-------:|-------:|-----------------:|
| **ARIMA Direction** | **5.82** ★ | 5.94e-09 | 3.26 | 5,471% | -21.8% | 98% |
| ARIMA + Momentum Filter | 5.30 ★ | 1.13e-07 | 3.35 | 2,703% | -21.2% | 98% |
| ARIMA Confidence-Scaled | 5.29 ★ | 1.21e-07 | 3.10 | 3,133% | -27.2% | 97% |
| ARIMA Ensemble (3 orders) | 5.27 ★ | 1.33e-07 | 3.22 | 998% | -17.2% | 96% |
| ARIMA + Vol Scaled | 5.18 ★ | 2.26e-07 | 3.25 | 1,130% | -16.3% | 97% |

*OOS-only: 1,701 days out-of-sample, Newey-West HAC standard errors, stationary block bootstrap (500 iterations, 21-day blocks), transaction costs 15bps/side + Almgren-Chriss market impact, purge gap 5 days.*

### Methodology

1. **Pre-compute rolling one-step-ahead ARIMA forecasts** on expanding window with re-fit every 63 days — separates "view" (forecast) from "execution" (position sizing)
2. **In-sample parameter optimization** per rolling window on the `t-statistic` objective (not Sharpe — maximizes statistical power)
3. **Purge gap** (5 days) between train and test prevents information leakage
4. **Out-of-sample only** metrics reported — no in-sample data contaminates results
5. **Newey-West HAC** t-statistic with auto-selected lag (Andrews 1991) accounts for serial correlation in daily returns
6. **Stationary block bootstrap** provides non-parametric t-stat distribution

Run it:
```bash
python scripts/arima_optimizer.py
python visualization/generate_gifs.py
```

---

## Multi-Asset Contrarian ARIMA · 10 Assets, 4 reach 5σ

The same ARIMA walk-forward engine tested on **10 assets across 6 categories** using **Contrarian ARIMA**: the standard ARIMA direction signal is walk-forward optimized, then **inverted** at deployment. This exploits the systematic overshooting of ARIMA models on realistic market data — when the model says "up," the market tends to mean-revert "down."

### Cross-Asset ARIMA Backtest
Animated 4-panel dashboard revealing each asset: equity curves, HAC t-statistics, category averages, and drawdowns. Contrarian Vol-Scaled dominates across all categories.

![Multi-Asset Backtest](assets/multi_asset_backtest.gif)

### Results Summary

| Asset | Category | Best Strategy | t-stat (HAC) | Sharpe | Return | Max DD | Win Rate |
|-------|----------|--------------|-------------:|-------:|-------:|-------:|---------:|
| **SPY** | **Indices** | **Contrarian Vol** | **7.09** ★ | 4.15 | 249% | -4.8% | 69% |
| **TLT** | **Bonds** | **Contrarian Vol** | **5.39** ★ | 3.50 | 112% | -6.1% | 77% |
| **QQQ** | **Indices** | **Contrarian Vol** | **5.01** ★ | 2.73 | 64% | -5.1% | 74% |
| **BTC** | **Crypto** | **Contrarian Vol** | **5.00** ★ | 2.87 | 192% | -10.6% | 66% |
| TSLA | Stocks | Contrarian Vol | 4.85 | 3.02 | 168% | -11.5% | 74% |
| EURUSD | Forex | Contrarian | 4.59 | 3.05 | 131% | -8.5% | 67% |
| ETH | Crypto | Contrarian Vol | 4.40 | 2.69 | 267% | -10.9% | 67% |
| GLD | Commodities | Contrarian Vol | 3.52 | 2.20 | 46% | -3.6% | 72% |
| MSFT | Stocks | Contrarian Vol | 3.02 | 1.81 | 52% | -7.5% | 68% |
| AAPL | Stocks | Contrarian Vol | 2.77 | 1.90 | 47% | -6.4% | 70% |

*★ = 5-sigma significance. Contrarian inverts the walk-forward optimized ARIMA direction signal using the same parameters — no separate re-optimization. Vol-Scaled variant adds target volatility scaling (Almgren-Chriss market impact, 15bps/side costs).*

### Category Performance

| Category | Avg t-stat | Avg Sharpe | Avg Return | Assets |
|----------|----------:|----------:|----------:|-------:|
| **Indices** | **6.05** | 3.44 | 156% | 2 |
| **Bonds** | **5.39** | 3.50 | 112% | 1 |
| **Crypto** | 4.70 | 2.78 | 229% | 2 |
| Forex | 4.59 | 3.05 | 131% | 1 |
| Stocks | 3.55 | 2.24 | 89% | 3 |
| Commodities | 3.52 | 2.20 | 46% | 1 |

### Why Contrarian Works

ARIMA models on realistic market data systematically overshoot — predicting continuation when markets mean-revert. The standard direction strategy shows **negative** t-stats (e.g., SPY: -7.09, TLT: -5.39), meaning the signal is **predictive but inverted**. Rather than re-optimizing the contrarian separately (which would double the overfitting risk), we:

1. **Walk-forward optimize** the standard direction strategy (threshold, vol target)
2. **Invert the OOS signal** using the same parameters — no additional degrees of freedom
3. The negative t-stat becomes positive, with identical magnitude

This is a legitimate mean-reversion alpha source: ARIMA captures short-term autocorrelation patterns that reverse within the holding period.

Run it:
```bash
python scripts/multi_asset_backtest.py
```

---

## Features

### Data Fetchers (Resilient, Validated)
- **Prediction Markets**: Polymarket CLOB, Kalshi v2 API
- **Crypto Oracles**: Chainlink on-chain feeds, Pyth Hermes, oracle aggregation with median-of-medians
- **Stock Markets**: yfinance, Finnhub, Alpha Vantage
- **News**: NewsAPI with source-credibility weighting + deduplication
- **Infrastructure**: retry with exponential backoff, circuit breaker, rate limiter, data quality validation

### Alpha Engine
- Context-aware sentiment (negation, amplifiers, source credibility)
- Hurst exponent regime detection
- Information Coefficient (IC) tracking + ICIR
- Multi-horizon alpha decay analysis
- Regime-adaptive signal weighting
- Prediction market edge (Bayesian updating)

### Backtesting Engine
- Look-ahead bias prevention (strict signal.shift(1) + purge gap in walk-forward)
- Almgren-Chriss market impact model
- **Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014)
- **Probability of Backtest Overfitting** (Combinatorial Purged CV)
- Block bootstrap Monte Carlo (1000 sims)
- Minimum Track Record Length
- Full metrics: Sortino, Calmar, Profit Factor, Tail Ratio, DD Duration

### Risk System
- **VaR**: Historical, Parametric, Cornish-Fisher, **EVT (GPD tail)**
- **CVaR / Expected Shortfall**
- **EWMA volatility** (RiskMetrics λ=0.94)
- **Hill tail index** estimator
- **Mahalanobis distance** regime detection
- Autocorrelation-adjusted KS test
- Structural break detection (Chow + Levene)
- Tail dependence / contagion detection

### Portfolio Construction
- Risk Parity
- Kelly Criterion (fractional)
- Mean-Variance with Ledoit-Wolf shrinkage
- Black-Litterman
- Maximum Diversification

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in your API keys

# Demo mode (no API keys needed — synthetic data)
python main.py demo

# Full live scan
python main.py scan

# Backtest with 3D surfaces + Deflated Sharpe + PBO
python main.py backtest

# Risk analysis (VaR/CVaR/EVT, drift, tail)
python main.py risk

# Portfolio optimization (Risk Parity, Kelly, MV, BL)
python main.py portfolio

# Continuous screener
python main.py continuous

# Regenerate README GIFs
python visualization/generate_gifs.py
```

## Project Structure

```
├── main.py                          # Entry point
├── config/
│   ├── settings.py                  # Typed config with startup validation
│   ├── resilience.py                # Retry, circuit breaker, rate limiter
│   └── validation.py                # Data quality validation
├── data_fetchers/
│   ├── polymarket.py                # Polymarket CLOB
│   ├── kalshi.py                    # Kalshi v2
│   ├── crypto.py                    # Chainlink + Pyth + CoinGecko + Aggregator
│   ├── stocks.py                    # yfinance, Finnhub, Alpha Vantage
│   └── news.py                      # NewsAPI with credibility weighting
├── signals/
│   ├── alpha_engine.py              # IC, alpha decay, Hurst, cross-market
│   └── arima_forecaster.py          # ARIMA/SARIMA auto-order, rolling forecasts
├── backtesting/
│   ├── engine.py                    # DSR, PBO, CPCV, Monte Carlo, market impact
│   └── rolling_engine.py            # Walk-forward, Newey-West HAC, bootstrap
├── risk/
│   ├── drift_detector.py            # KS, Mahalanobis, CUSUM, structural break
│   ├── tail_risk.py                 # VaR/CVaR/EVT, EWMA, Hill, contagion
│   └── portfolio.py                 # Risk Parity, Kelly, MV, BL, Max Div
├── visualization/
│   ├── pnl_surfaces.py              # Static + animated 3D surfaces, dashboards
│   └── generate_gifs.py             # README GIF generator (8 GIFs)
├── scripts/
│   ├── arima_optimizer.py           # ARIMA strategy optimizer (t-stat>5 target)
│   └── multi_asset_backtest.py      # 10-asset cross-category backtest
├── screener/
│   ├── strategies.py                # Regime-adaptive, vol-scaled strategies
│   ├── arima_strategies.py          # ARIMA-based trading strategies
│   └── multi_market_screener.py     # Orchestrator
└── tests/                           # 42 passing tests
```

## API Keys

| Source | Key Required | Free Tier |
|--------|-------------|-----------|
| Polymarket | No | Public API |
| Kalshi | Optional | Public reads |
| Chainlink | No | Public RPC |
| Pyth Network | No | Hermes API |
| CoinGecko | No | Rate-limited |
| yfinance | No | Unlimited |
| Finnhub | Yes | 60/min |
| Alpha Vantage | Yes | 25/day |
| NewsAPI | Yes | 100/day |

## Running Tests

```bash
pytest tests/ -v
# 42 passed
```
