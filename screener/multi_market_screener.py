"""
Multi-Market Alpha Screener — Main Orchestrator.
Combines prediction markets, crypto, stocks, and news
to generate a unified alpha dashboard.
"""
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from data_fetchers.polymarket import PolymarketFetcher
from data_fetchers.kalshi import KalshiFetcher
from data_fetchers.crypto import ChainlinkFetcher, PythFetcher, CoinGeckoFetcher
from data_fetchers.stocks import YFinanceFetcher, FinnhubFetcher
from data_fetchers.news import NewsApiFetcher
from signals.alpha_engine import AlphaSignalEngine
from backtesting.engine import BacktestEngine, Strategy
from risk.drift_detector import DriftDetector
from risk.tail_risk import TailRiskScreener
from visualization.pnl_surfaces import PnLSurfaceVisualizer
from screener.strategies import (
    momentum_crossover,
    mean_reversion_bollinger,
    rsi_strategy,
    momentum_crossover_factory,
    bollinger_factory,
)
from config.settings import STOCK_WATCHLIST, CRYPTO_WATCHLIST, SCAN_INTERVAL_SECONDS

logger = logging.getLogger(__name__)


class MultiMarketScreener:
    """
    Unified screener that:
    1. Fetches live data from prediction markets, crypto, and stocks
    2. Pulls news and computes sentiment-based alpha signals
    3. Runs backtests with parameter sweeps
    4. Monitors drift and tail risk
    5. Generates animated 3D PnL surfaces
    """

    def __init__(self):
        # Data fetchers
        self.polymarket = PolymarketFetcher()
        self.kalshi = KalshiFetcher()
        self.chainlink = ChainlinkFetcher()
        self.pyth = PythFetcher()
        self.coingecko = CoinGeckoFetcher()
        self.yfinance = YFinanceFetcher()
        self.finnhub = FinnhubFetcher()
        self.news = NewsApiFetcher()

        # Engines
        self.alpha_engine = AlphaSignalEngine()
        self.backtest_engine = BacktestEngine()
        self.drift_detector = DriftDetector()
        self.tail_risk = TailRiskScreener()
        self.visualizer = PnLSurfaceVisualizer()

        # State
        self.last_scan = None
        self.scan_results = {}

    # ── Data Collection ──────────────────────────────────────────────────

    def fetch_prediction_markets(self) -> pd.DataFrame:
        """Fetch and combine prediction market data."""
        logger.info("Fetching prediction markets...")
        poly = self.polymarket.get_markets(limit=30)
        kalshi = self.kalshi.get_events(limit=30)

        frames = []
        if not poly.empty:
            frames.append(poly)
        if not kalshi.empty:
            frames.append(kalshi)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            logger.info("Fetched %d prediction markets", len(combined))
            return combined
        return pd.DataFrame()

    def fetch_crypto_prices(self) -> Dict[str, pd.DataFrame]:
        """Fetch crypto OHLCV + live oracle prices."""
        logger.info("Fetching crypto prices...")

        # Live oracle prices
        chainlink_prices = self.chainlink.get_prices()
        pyth_prices = self.pyth.get_prices()
        if not chainlink_prices.empty:
            logger.info("Chainlink prices:\n%s", chainlink_prices.to_string())
        if not pyth_prices.empty:
            logger.info("Pyth prices:\n%s", pyth_prices.to_string())

        # Historical OHLCV
        ohlcv = self.coingecko.get_all_ohlcv(CRYPTO_WATCHLIST[:3], days=90)
        return ohlcv

    def fetch_stock_prices(self) -> Dict[str, pd.DataFrame]:
        """Fetch stock OHLCV data."""
        logger.info("Fetching stock prices...")
        return self.yfinance.get_batch_ohlcv(STOCK_WATCHLIST[:5], period="3mo")

    def fetch_news_sentiment(
        self, stock_symbols: List[str], crypto_symbols: List[str]
    ) -> pd.DataFrame:
        """Fetch news and compute sentiment for all assets."""
        logger.info("Fetching news sentiment...")
        stock_news = self.news.get_news_for_assets(stock_symbols[:3], "stock")
        crypto_news = self.news.get_news_for_assets(crypto_symbols[:3], "crypto")

        all_news = pd.concat(
            [df for df in [stock_news, crypto_news] if not df.empty],
            ignore_index=True,
        ) if not stock_news.empty or not crypto_news.empty else pd.DataFrame()

        if not all_news.empty:
            all_news = self.alpha_engine.compute_news_sentiment(all_news)

        return all_news

    # ── Alpha Generation ─────────────────────────────────────────────────

    def generate_alpha_signals(
        self,
        stock_prices: Dict[str, pd.DataFrame],
        crypto_prices: Dict[str, pd.DataFrame],
        news_df: pd.DataFrame,
        pred_markets: pd.DataFrame,
    ) -> dict:
        """Generate cross-market alpha signals."""
        logger.info("Generating alpha signals...")

        # Per-asset sentiment
        sentiment_agg = self.alpha_engine.aggregate_asset_sentiment(news_df)

        # Cross-market composite signals
        cross_signals = self.alpha_engine.cross_market_signals(
            stock_prices, crypto_prices, sentiment_agg
        )

        # Prediction market edge
        pred_edge = self.alpha_engine.prediction_market_edge(pred_markets, sentiment_agg)

        return {
            "cross_market_signals": cross_signals,
            "prediction_edge": pred_edge,
            "sentiment_summary": sentiment_agg,
        }

    # ── Backtesting ──────────────────────────────────────────────────────

    def run_strategy_backtest(
        self, prices: pd.DataFrame, strategy_name: str = "momentum"
    ) -> dict:
        """Run a backtest for a given strategy on price data."""
        strategies = {
            "momentum": Strategy("Momentum 10/30", momentum_crossover(10, 30)),
            "mean_reversion": Strategy("Bollinger 20/2", mean_reversion_bollinger(20, 2.0)),
            "rsi": Strategy("RSI 14", rsi_strategy(14)),
        }
        strategy = strategies.get(strategy_name, strategies["momentum"])
        result = self.backtest_engine.run(prices, strategy)
        return {
            "metrics": result.metrics,
            "equity_curve": result.equity_curve,
            "returns": result.returns,
        }

    def run_parameter_sweep(
        self, prices: pd.DataFrame
    ) -> dict:
        """Run 2D parameter sweep for 3D PnL surface."""
        result = self.backtest_engine.parameter_sweep(
            prices,
            signal_fn_factory=momentum_crossover_factory,
            param_grid={"fast": [5, 10, 15, 20, 25], "slow": [20, 30, 40, 50, 60]},
        )
        return {
            "best_metrics": result.metrics,
            "pnl_surface": result.pnl_by_param,
        }

    def run_time_evolving_sweep(
        self, prices: pd.DataFrame, n_periods: int = 6
    ) -> Dict[str, pd.DataFrame]:
        """
        Run parameter sweeps on rolling windows to create
        time-evolving PnL surface data.
        """
        total_len = len(prices)
        window_size = total_len // 2
        step = (total_len - window_size) // max(n_periods - 1, 1)

        time_surfaces = {}
        for i in range(n_periods):
            start = i * step
            end = start + window_size
            if end > total_len:
                break
            window_prices = prices.iloc[start:end]
            period_label = str(window_prices.index[-1].date()) if hasattr(window_prices.index[-1], 'date') else f"Period {i+1}"

            try:
                result = self.backtest_engine.parameter_sweep(
                    window_prices,
                    signal_fn_factory=momentum_crossover_factory,
                    param_grid={"fast": [5, 10, 15, 20], "slow": [25, 35, 45, 55]},
                )
                if result.pnl_by_param is not None:
                    time_surfaces[period_label] = result.pnl_by_param
            except Exception as e:
                logger.warning("Sweep failed for period %s: %s", period_label, e)

        return time_surfaces

    # ── Risk Monitoring ──────────────────────────────────────────────────

    def run_risk_scan(
        self,
        returns: pd.Series,
        equity: pd.Series,
        returns_dict: Optional[Dict[str, pd.Series]] = None,
        current_weights: Optional[Dict[str, float]] = None,
        target_weights: Optional[Dict[str, float]] = None,
    ) -> dict:
        """Run full drift + tail risk scan."""
        drift_alerts = self.drift_detector.full_drift_scan(
            returns, current_weights, target_weights
        )
        tail_alerts = self.tail_risk.full_tail_risk_scan(
            returns, equity, returns_dict
        )
        risk_dashboard = self.tail_risk.get_risk_dashboard_data(returns)

        return {
            "drift_alerts": drift_alerts,
            "tail_risk_alerts": tail_alerts,
            "risk_metrics": risk_dashboard,
            "total_alerts": len(drift_alerts) + len(tail_alerts),
        }

    # ── Visualization ────────────────────────────────────────────────────

    def generate_3d_surface(self, pnl_df: pd.DataFrame):
        """Generate interactive 3D PnL surface."""
        return self.visualizer.plot_static_pnl_surface(
            pnl_df, param1="fast", param2="slow", metric="sharpe",
            title="Strategy PnL Surface: Sharpe Ratio",
        )

    def generate_animated_surface(self, time_surfaces: Dict[str, pd.DataFrame]):
        """Generate animated time-evolving PnL surface."""
        return self.visualizer.plot_animated_pnl_surface(
            time_surfaces, param1="fast", param2="slow", metric="sharpe",
            title="Time-Evolving PnL Surface",
        )

    # ── Full Scan ────────────────────────────────────────────────────────

    def full_scan(self) -> dict:
        """
        Execute full multi-market scan:
        1. Fetch all market data
        2. Generate alpha signals
        3. Run backtests
        4. Check risk constraints
        5. Prepare visualizations
        """
        logger.info("=" * 60)
        logger.info("FULL MULTI-MARKET SCAN — %s", datetime.utcnow().isoformat())
        logger.info("=" * 60)

        # 1. Fetch data
        pred_markets = self.fetch_prediction_markets()
        crypto_prices = self.fetch_crypto_prices()
        stock_prices = self.fetch_stock_prices()
        news_df = self.fetch_news_sentiment(
            list(stock_prices.keys()), list(crypto_prices.keys())
        )

        # 2. Alpha signals
        alpha = self.generate_alpha_signals(
            stock_prices, crypto_prices, news_df, pred_markets
        )

        # 3. Backtest top signals on available data
        backtest_results = {}
        pnl_surfaces = {}
        time_surfaces = {}

        for symbol, prices in list(stock_prices.items())[:3]:
            if prices.empty or len(prices) < 50:
                continue
            # Run backtest
            bt = self.run_strategy_backtest(prices, "momentum")
            backtest_results[symbol] = bt

            # Parameter sweep
            sweep = self.run_parameter_sweep(prices)
            pnl_surfaces[symbol] = sweep["pnl_surface"]

            # Time-evolving sweep
            ts = self.run_time_evolving_sweep(prices, n_periods=4)
            if ts:
                time_surfaces[symbol] = ts

        # 4. Risk scan on combined returns
        combined_returns = pd.DataFrame({
            sym: bt["returns"]
            for sym, bt in backtest_results.items()
        })
        if not combined_returns.empty:
            portfolio_returns = combined_returns.mean(axis=1)
            portfolio_equity = 100000 * (1 + portfolio_returns).cumprod()
            risk_scan = self.run_risk_scan(
                portfolio_returns, portfolio_equity,
                returns_dict={sym: combined_returns[sym] for sym in combined_returns.columns},
            )
        else:
            risk_scan = {}

        self.last_scan = datetime.utcnow()
        self.scan_results = {
            "prediction_markets": pred_markets,
            "alpha_signals": alpha,
            "backtest_results": backtest_results,
            "pnl_surfaces": pnl_surfaces,
            "time_surfaces": time_surfaces,
            "risk": risk_scan,
            "scan_time": self.last_scan,
        }

        self._print_summary()
        return self.scan_results

    def _print_summary(self):
        """Print a concise scan summary."""
        r = self.scan_results
        print("\n" + "=" * 60)
        print(f"SCAN COMPLETE — {r.get('scan_time', '')}")
        print("=" * 60)

        # Alpha signals
        alpha = r.get("alpha_signals", {})
        cs = alpha.get("cross_market_signals", pd.DataFrame())
        if not cs.empty:
            print("\n── TOP ALPHA SIGNALS ──")
            print(cs.head(10).to_string(index=False))

        pe = alpha.get("prediction_edge", pd.DataFrame())
        if not pe.empty:
            print("\n── PREDICTION MARKET EDGE ──")
            print(pe.head(5).to_string(index=False))

        # Backtest
        bt = r.get("backtest_results", {})
        if bt:
            print("\n── BACKTEST RESULTS ──")
            for sym, res in bt.items():
                m = res["metrics"]
                print(f"  {sym}: Sharpe={m['sharpe_ratio']:.2f}, "
                      f"Return={m['total_return']:.1%}, "
                      f"MaxDD={m['max_drawdown']:.1%}, "
                      f"WinRate={m['win_rate']:.0%}")

        # Risk
        risk = r.get("risk", {})
        if risk:
            n_alerts = risk.get("total_alerts", 0)
            print(f"\n── RISK ALERTS: {n_alerts} ──")
            for alert in risk.get("drift_alerts", [])[:3]:
                print(f"  [{alert.severity.upper()}] {alert.message}")
            for alert in risk.get("tail_risk_alerts", [])[:3]:
                print(f"  [{alert.severity.upper()}] {alert.message}")

        print("=" * 60 + "\n")

    def run_continuous(self, interval: int = SCAN_INTERVAL_SECONDS):
        """Run screener continuously at a fixed interval."""
        logger.info("Starting continuous screener (interval=%ds)", interval)
        while True:
            try:
                self.full_scan()
            except Exception as e:
                logger.error("Scan failed: %s", e)
            time.sleep(interval)
