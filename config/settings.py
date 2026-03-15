"""
Central configuration for the Multi-Market Alpha Screener.
API keys are loaded from environment variables for security.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys (set via .env file or environment) ──────────────────────────
POLYMARKET_API_URL = "https://clob.polymarket.com"
KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_API_KEY = os.getenv("KALSHI_API_KEY", "")

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")  # https://newsapi.org (free tier)
NEWS_API_URL = "https://newsapi.org/v2"

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")  # free key from alphavantage.co
FINNHUB_KEY = os.getenv("FINNHUB_KEY", "")  # free key from finnhub.io

# Chainlink price feed (public, no key needed)
CHAINLINK_ETH_RPC = os.getenv("ETH_RPC_URL", "https://eth.llamarpc.com")

# Pyth Network (public, no key needed)
PYTH_API_URL = "https://hermes.pyth.network"

# ── Screener Settings ────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = 60
LOOKBACK_DAYS = 90

# Watchlists
STOCK_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "JNJ", "SPY", "QQQ", "IWM", "GLD", "TLT",
]
CRYPTO_WATCHLIST = [
    "BTC", "ETH", "SOL", "LINK", "AVAX", "MATIC", "ARB",
]

# ── Risk Constraints ─────────────────────────────────────────────────────
MAX_PORTFOLIO_DRIFT_PCT = 15.0   # max allowed drift from target allocation
TAIL_RISK_VAR_CONFIDENCE = 0.99  # 99% VaR
TAIL_RISK_CVAR_CONFIDENCE = 0.975
MAX_DRAWDOWN_ALERT_PCT = 10.0
VOLATILITY_SPIKE_THRESHOLD = 2.5  # z-score for vol spike alert

# ── Backtesting ──────────────────────────────────────────────────────────
BACKTEST_INITIAL_CAPITAL = 100_000
BACKTEST_COMMISSION_PCT = 0.001   # 10 bps
BACKTEST_SLIPPAGE_PCT = 0.0005   # 5 bps
