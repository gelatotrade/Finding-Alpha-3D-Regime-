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

## Multi-Asset Cross-Category Backtest · Reality Check

The same ARIMA walk-forward engine tested on **10 assets across 6 categories** with realistic market-calibrated autocorrelation structure. Five strategies per asset (Direction, Vol-Scaled, Momentum, Contrarian, Ensemble) compete via walk-forward with in-sample t-stat optimization and strict OOS evaluation.

### Cross-Asset ARIMA Backtest
Animated 4-panel dashboard revealing each asset one by one: equity curves, HAC t-statistics, category averages, and drawdowns. Every asset shows negative alpha after costs and walk-forward parameter overfitting — the honest result at realistic autocorrelation levels.

![Multi-Asset Backtest](assets/multi_asset_backtest.gif)

### Results Summary

| Asset | Category | Best Strategy | t-stat (HAC) | Sharpe | Return | Max DD | Sig. |
|-------|----------|--------------|-------------:|-------:|-------:|-------:|-----:|
| AAPL | Stocks | Ensemble | -0.59 | -0.44 | -5.9% | -12.8% | n.s. |
| MSFT | Stocks | Direction | -1.33 | -0.71 | -41.7% | -55.8% | n.s. |
| TSLA | Stocks | Contrarian | -1.69 | -1.14 | -46.3% | -48.4% | n.s. |
| BTC | Crypto | Ensemble | -1.02 | -0.62 | -16.6% | -22.4% | n.s. |
| ETH | Crypto | Momentum | -0.71 | -0.43 | -75.8% | -90.4% | n.s. |
| SPY | Indices | Ensemble | -2.26 | -1.46 | -18.9% | -21.8% | 5% |
| QQQ | Indices | Direction | -2.80 | -1.53 | -43.8% | -45.9% | 1% |
| GLD | Commodities | Ensemble | -2.41 | -1.52 | -19.1% | -22.5% | 5% |
| TLT | Bonds | Contrarian | -3.46 | -2.17 | -48.0% | -50.9% | 1% |
| EURUSD | Forex | Momentum | -3.63 | -2.15 | -36.5% | -38.2% | 1% |

*5 strategies tested per asset. Contrarian inverts the ARIMA direction signal with vol targeting. Ensemble aggregates ARIMA(1,0,1) + ARIMA(2,0,2).*

### Key Insight — Academic Honesty

The contrast validates the engine's integrity:
- **Synthetic AR(2) data** (strong coefficients 0.35–0.50) → **t > 5** on all 5 strategies (5-sigma significance)
- **Realistic market data** (weak coefficients 0.03–0.20) → **negative alpha** across all 10 assets

Even the contrarian variant (which inverts the signal) fails — the negative t-stats aren't from a systematic inverse; they're from **walk-forward parameter overfitting**. When the signal-to-noise is low enough, t-stat maximization in-sample picks parameters that actively hurt out-of-sample. This is the classic symptom of an efficient market: no amount of parameter tuning can manufacture alpha from weak autocorrelation.

The engine does **not** hallucinate alpha where none exists. This is the correct outcome.

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
