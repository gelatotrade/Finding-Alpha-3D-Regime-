"""
Central configuration for the Multi-Market Alpha Screener.
Institutional-grade: validated at startup, typed, documented.
"""
import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ── Retry / Resilience ───────────────────────────────────────────────────

@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    exponential_base: float = 2.0
    retry_on_status: tuple = (429, 500, 502, 503, 504)


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout_s: float = 60.0
    half_open_max_calls: int = 1


# ── API Configurations ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ApiConfig:
    """Per-endpoint API configuration."""
    base_url: str
    api_key: str = ""
    timeout_s: float = 15.0
    rate_limit_per_min: int = 60
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)


# Build from environment
POLYMARKET_CONFIG = ApiConfig(
    base_url="https://clob.polymarket.com",
    timeout_s=15.0,
    rate_limit_per_min=30,
)

KALSHI_CONFIG = ApiConfig(
    base_url="https://api.elections.kalshi.com/trade-api/v2",
    api_key=os.getenv("KALSHI_API_KEY", ""),
    timeout_s=15.0,
    rate_limit_per_min=30,
)

NEWS_API_CONFIG = ApiConfig(
    base_url="https://newsapi.org/v2",
    api_key=os.getenv("NEWS_API_KEY", ""),
    timeout_s=15.0,
    rate_limit_per_min=50,  # free tier ~100/day
)

ALPHA_VANTAGE_CONFIG = ApiConfig(
    base_url="https://www.alphavantage.co/query",
    api_key=os.getenv("ALPHA_VANTAGE_KEY", ""),
    timeout_s=20.0,
    rate_limit_per_min=5,  # free tier 25/day
)

FINNHUB_CONFIG = ApiConfig(
    base_url="https://finnhub.io/api/v1",
    api_key=os.getenv("FINNHUB_KEY", ""),
    timeout_s=10.0,
    rate_limit_per_min=60,
)

# Chainlink / Pyth — public, no key
CHAINLINK_ETH_RPC = os.getenv("ETH_RPC_URL", "https://eth.llamarpc.com")
PYTH_API_URL = "https://hermes.pyth.network"

# ── Watchlists ───────────────────────────────────────────────────────────

STOCK_WATCHLIST: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "JNJ", "SPY", "QQQ", "IWM", "GLD", "TLT",
]
CRYPTO_WATCHLIST: List[str] = [
    "BTC", "ETH", "SOL", "LINK", "AVAX", "MATIC", "ARB",
]

# ── Screener ─────────────────────────────────────────────────────────────

SCAN_INTERVAL_SECONDS: int = 60
LOOKBACK_DAYS: int = 90

# ── Risk Constraints (documented thresholds) ─────────────────────────────

# Allocation drift: rebalance trigger when any position deviates >15%
# from target weight.  Industry standard range: 5–20%.
MAX_PORTFOLIO_DRIFT_PCT: float = 15.0

# VaR / CVaR confidence levels per Basel III / FRTB guidelines
TAIL_RISK_VAR_CONFIDENCE: float = 0.99
TAIL_RISK_CVAR_CONFIDENCE: float = 0.975

# Drawdown: alert when portfolio drops >10% from peak.
# Institutional norm: 5–15% depending on strategy mandate.
MAX_DRAWDOWN_ALERT_PCT: float = 10.0

# Volatility spike: z-score threshold for short-term vol vs long-term.
# 2.5σ ≈ 99.4% confidence that vol is abnormal.
VOLATILITY_SPIKE_THRESHOLD: float = 2.5

# ── Backtesting ──────────────────────────────────────────────────────────

BACKTEST_INITIAL_CAPITAL: float = 100_000.0
BACKTEST_COMMISSION_BPS: float = 10.0     # 10 bps per side
BACKTEST_SLIPPAGE_BPS: float = 5.0        # 5 bps per side

# Market impact (Almgren-Chriss square-root model)
MARKET_IMPACT_ETA: float = 0.1            # permanent impact coefficient
MARKET_IMPACT_GAMMA: float = 0.1          # temporary impact coefficient

# Walk-forward
WALK_FORWARD_TRAIN_DAYS: int = 126        # 6 months
WALK_FORWARD_TEST_DAYS: int = 21          # 1 month
PURGE_DAYS: int = 5                       # embargo between train/test

# ── Statistical Testing ──────────────────────────────────────────────────

MIN_OBSERVATIONS_FOR_STATS: int = 30      # absolute minimum
BONFERRONI_CORRECTION: bool = True        # correct for multiple tests
FDR_LEVEL: float = 0.05                   # false discovery rate
DEFLATED_SHARPE_ENABLED: bool = True      # Bailey & Lopez de Prado (2014)


# ── Startup Validation ───────────────────────────────────────────────────

def validate_config() -> List[str]:
    """Validate configuration at startup. Returns list of warnings."""
    warnings = []

    # Check API key availability
    if not NEWS_API_CONFIG.api_key:
        warnings.append("NEWS_API_KEY not set — news sentiment will be unavailable")
    if not FINNHUB_CONFIG.api_key:
        warnings.append("FINNHUB_KEY not set — Finnhub data unavailable")
    if not ALPHA_VANTAGE_CONFIG.api_key:
        warnings.append("ALPHA_VANTAGE_KEY not set — Alpha Vantage unavailable")

    # Sanity checks
    if BACKTEST_COMMISSION_BPS < 0:
        warnings.append("BACKTEST_COMMISSION_BPS is negative — check config")
    if MAX_PORTFOLIO_DRIFT_PCT <= 0 or MAX_PORTFOLIO_DRIFT_PCT > 100:
        warnings.append(f"MAX_PORTFOLIO_DRIFT_PCT={MAX_PORTFOLIO_DRIFT_PCT} is out of range (0,100]")
    if TAIL_RISK_VAR_CONFIDENCE <= 0.5 or TAIL_RISK_VAR_CONFIDENCE >= 1.0:
        warnings.append(f"TAIL_RISK_VAR_CONFIDENCE={TAIL_RISK_VAR_CONFIDENCE} should be in (0.5, 1.0)")

    for w in warnings:
        logger.warning("CONFIG: %s", w)

    return warnings
