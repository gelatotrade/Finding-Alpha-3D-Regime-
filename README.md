# Finding Alpha 3D Regime

Multi-market alpha screener with animated 3D PnL surfaces, backtesting engine, drift detection, and tail risk early warning system.

## Features

- **Multi-Market Data Fetchers**: Polymarket, Kalshi (prediction markets), Chainlink/Pyth (crypto oracles), CoinGecko, yfinance, Finnhub, Alpha Vantage
- **News Sentiment Alpha**: NewsAPI integration with keyword-based sentiment scoring and cross-market signal generation
- **Backtesting Engine**: Vectorized backtester with parameter sweeps, walk-forward analysis, and transaction cost modeling
- **3D Animated PnL Surfaces**: Time-evolving Plotly/Matplotlib surfaces showing strategy performance across parameter space
- **Drift Detection**: KS-test return distribution drift, allocation drift, regime change detection, CUSUM parameter stability
- **Tail Risk Early Warning**: VaR/CVaR (historical + Cornish-Fisher), volatility spike detection, drawdown monitoring, cross-asset contagion alerts

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in your API keys

# Demo mode (no API keys needed — synthetic data)
python main.py demo

# Full scan with live data
python main.py scan

# Backtest with 3D surfaces
python main.py backtest

# Risk analysis
python main.py risk

# Continuous screener
python main.py continuous
```

## Project Structure

```
├── main.py                          # Entry point (scan/backtest/risk/demo/continuous)
├── config/settings.py               # API keys, watchlists, risk thresholds
├── data_fetchers/
│   ├── polymarket.py                # Polymarket CLOB API
│   ├── kalshi.py                    # Kalshi v2 API
│   ├── crypto.py                    # Chainlink, Pyth, CoinGecko
│   ├── stocks.py                    # yfinance, Finnhub, Alpha Vantage
│   └── news.py                      # NewsAPI.org
├── signals/alpha_engine.py          # Sentiment, momentum, mean-reversion, prediction edge
├── backtesting/engine.py            # Vectorized backtest, parameter sweep, walk-forward
├── risk/
│   ├── drift_detector.py            # Allocation/return/regime/parameter drift
│   └── tail_risk.py                 # VaR, CVaR, vol spikes, drawdown, contagion
├── visualization/pnl_surfaces.py    # 3D surfaces (static + animated), risk dashboard
└── screener/
    ├── strategies.py                # Built-in trading strategies
    └── multi_market_screener.py     # Main orchestrator
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
| Finnhub | Yes | Free tier |
| Alpha Vantage | Yes | 25 req/day |
| NewsAPI | Yes | 100 req/day |
