"""
Multi-Market Alpha Screener — Institutional Orchestrator.

Integrates all components:
- Multi-source data fetching with resilience
- Alpha signal generation with IC tracking
- Backtesting with Deflated Sharpe and PBO
- Portfolio construction (risk parity, Kelly, BL)
- Drift detection and tail risk monitoring
- 3D animated PnL surface generation
"""
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from data_fetchers.polymarket import PolymarketFetcher
from data_fetchers.kalshi import KalshiFetcher
from data_fetchers.crypto import ChainlinkFetcher, PythFetcher, CoinGeckoFetcher, OracleAggregator
from data_fetchers.stocks import YFinanceFetcher, FinnhubFetcher
from data_fetchers.news import NewsApiFetcher
from signals.alpha_engine import AlphaSignalEngine
from backtesting.engine import BacktestEngine, Strategy
from risk.drift_detector import DriftDetector
from risk.tail_risk import TailRiskScreener
from risk.portfolio import PortfolioConstructor
from visualization.pnl_surfaces import PnLSurfaceVisualizer
from screener.strategies import (
    momentum_crossover,
    mean_reversion_bollinger,
    rsi_strategy,
    regime_adaptive,
    momentum_crossover_factory,
    bollinger_factory,
)
from config.settings import (
    STOCK_WATCHLIST, CRYPTO_WATCHLIST, SCAN_INTERVAL_SECONDS,
    validate_config,
)

logger = logging.getLogger(__name__)


class MultiMarketScreener:
    """
    Institutional multi-market screener that orchestrates:
    data → signals → backtest → risk → portfolio → visualization.
    """

    def __init__(self):
        # Validate config at startup
        warnings = validate_config()
        if warnings:
            logger.info("Config validation: %d warnings", len(warnings))

        # Data fetchers
        self.polymarket = PolymarketFetcher()
        self.kalshi = KalshiFetcher()
        self.chainlink = ChainlinkFetcher()
        self.pyth = PythFetcher()
        self.coingecko = CoinGeckoFetcher()
        self.oracle_agg = OracleAggregator()
        self.yfinance = YFinanceFetcher()
        self.finnhub = FinnhubFetcher()
        self.news = NewsApiFetcher()

        # Engines
        self.alpha_engine = AlphaSignalEngine()
        self.backtest_engine = BacktestEngine()
        self.drift_detector = DriftDetector()
        self.tail_risk = TailRiskScreener()
        self.portfolio = PortfolioConstructor()
        self.visualizer = PnLSurfaceVisualizer()

        # State
        self.last_scan = None
        self.scan_results = {}

    # ── Data Collection ──────────────────────────────────────────────────

    def fetch_prediction_markets(self) -> pd.DataFrame:
        logger.info("Fetching prediction markets...")
        frames = []
        for name, fetcher, method in [
            ("polymarket", self.polymarket, "get_markets"),
            ("kalshi", self.kalshi, "get_events"),
        ]:
            try:
                df = getattr(fetcher, method)(limit=50)
                if not df.empty:
                    frames.append(df)
                    logger.info("  %s: %d markets", name, len(df))
            except Exception as e:
                logger.error("  %s failed: %s", name, e)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            return combined
        return pd.DataFrame()

    def fetch_crypto_prices(self) -> Dict[str, pd.DataFrame]:
        logger.info("Fetching crypto prices...")
        # Oracle aggregation
        agg_prices = self.oracle_agg.get_aggregated_prices()
        if not agg_prices.empty:
            logger.info("  Oracle aggregated: %d assets, max spread: %.2f%%",
                        len(agg_prices),
                        agg_prices["cross_source_spread_pct"].max())

        # Historical OHLCV
        ohlcv = self.coingecko.get_all_ohlcv(CRYPTO_WATCHLIST[:4], days=90)
        logger.info("  OHLCV: %d assets loaded", len(ohlcv))
        return ohlcv

    def fetch_stock_prices(self) -> Dict[str, pd.DataFrame]:
        logger.info("Fetching stock prices...")
        result = self.yfinance.get_batch_ohlcv(STOCK_WATCHLIST[:8], period="3mo")
        logger.info("  Stocks: %d assets loaded", len(result))
        return result

    def fetch_news_sentiment(
        self, stock_symbols: List[str], crypto_symbols: List[str]
    ) -> pd.DataFrame:
        logger.info("Fetching news sentiment...")
        frames = []
        for syms, atype in [(stock_symbols[:3], "stock"), (crypto_symbols[:3], "crypto")]:
            try:
                df = self.news.get_news_for_assets(syms, atype)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.error("  News fetch failed for %s: %s", atype, e)

        all_news = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not all_news.empty:
            all_news = self.alpha_engine.compute_news_sentiment(all_news)
            logger.info("  News: %d articles with sentiment", len(all_news))
        return all_news

    # ── Alpha Generation ─────────────────────────────────────────────────

    def generate_alpha_signals(
        self,
        stock_prices: Dict[str, pd.DataFrame],
        crypto_prices: Dict[str, pd.DataFrame],
        news_df: pd.DataFrame,
        pred_markets: pd.DataFrame,
    ) -> dict:
        logger.info("Generating alpha signals...")
        sentiment_agg = self.alpha_engine.aggregate_asset_sentiment(news_df)
        cross_signals = self.alpha_engine.cross_market_signals(
            stock_prices, crypto_prices, sentiment_agg
        )
        pred_edge = self.alpha_engine.prediction_market_edge(pred_markets, sentiment_agg)

        # Alpha decay analysis on top signals
        alpha_decay = {}
        for symbol, prices in list(stock_prices.items())[:3]:
            if prices.empty or len(prices) < 50:
                continue
            mom = self.alpha_engine.momentum_signal(prices)
            if not mom.empty and "mom_composite" in mom.columns:
                decay = self.alpha_engine.alpha_decay_analysis(
                    mom["mom_composite"], prices
                )
                if not decay.empty:
                    alpha_decay[symbol] = decay

        return {
            "cross_market_signals": cross_signals,
            "prediction_edge": pred_edge,
            "sentiment_summary": sentiment_agg,
            "alpha_decay": alpha_decay,
        }

    # ── Backtesting ──────────────────────────────────────────────────────

    def run_strategy_backtest(
        self, prices: pd.DataFrame, strategy_name: str = "momentum"
    ) -> dict:
        strategies = {
            "momentum": Strategy("Momentum 10/30", momentum_crossover(10, 30)),
            "mean_reversion": Strategy("Bollinger 20/2", mean_reversion_bollinger(20, 2.0)),
            "rsi": Strategy("RSI 14", rsi_strategy(14)),
            "regime_adaptive": Strategy("Regime Adaptive", regime_adaptive()),
        }
        strategy = strategies.get(strategy_name, strategies["momentum"])

        try:
            result = self.backtest_engine.run(prices, strategy)
            # Monte Carlo
            mc = self.backtest_engine.monte_carlo_simulation(result.returns)
            # CPCV / PBO
            pbo = self.backtest_engine.combinatorial_purged_cv(prices, strategy)

            return {
                "metrics": result.metrics,
                "equity_curve": result.equity_curve,
                "returns": result.returns,
                "monte_carlo": mc,
                "pbo": pbo,
            }
        except ValueError as e:
            logger.warning("Backtest failed for %s: %s", strategy_name, e)
            return {}

    def run_parameter_sweep(self, prices: pd.DataFrame) -> dict:
        try:
            result = self.backtest_engine.parameter_sweep(
                prices,
                signal_fn_factory=momentum_crossover_factory,
                param_grid={"fast": [5, 10, 15, 20, 25], "slow": [20, 30, 40, 50, 60]},
            )
            return {
                "best_metrics": result.metrics,
                "pnl_surface": result.pnl_by_param,
                "monte_carlo": result.monte_carlo,
            }
        except ValueError as e:
            logger.warning("Parameter sweep failed: %s", e)
            return {}

    def run_time_evolving_sweep(
        self, prices: pd.DataFrame, n_periods: int = 6
    ) -> Dict[str, pd.DataFrame]:
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
            if hasattr(window_prices.index[-1], 'date'):
                label = str(window_prices.index[-1].date())
            else:
                label = f"Period {i+1}"
            try:
                result = self.backtest_engine.parameter_sweep(
                    window_prices,
                    signal_fn_factory=momentum_crossover_factory,
                    param_grid={"fast": [5, 10, 15, 20], "slow": [25, 35, 45, 55]},
                )
                if result.pnl_by_param is not None:
                    time_surfaces[label] = result.pnl_by_param
            except ValueError:
                pass
        return time_surfaces

    # ── Portfolio Construction ────────────────────────────────────────────

    def construct_portfolio(
        self,
        returns_dict: Dict[str, pd.Series],
        method: str = "risk_parity",
    ) -> dict:
        if len(returns_dict) < 2:
            return {}

        df = pd.DataFrame(returns_dict).dropna()
        if len(df) < 30:
            return {}

        weights = self.portfolio.optimize(df, method=method)
        logger.info("Portfolio (%s): %s",
                     method, {k: f"{v:.1%}" for k, v in weights.items()})
        return weights

    # ── Risk Monitoring ──────────────────────────────────────────────────

    def run_risk_scan(
        self,
        returns: pd.Series,
        equity: pd.Series,
        returns_dict: Optional[Dict[str, pd.Series]] = None,
        current_weights: Optional[Dict[str, float]] = None,
        target_weights: Optional[Dict[str, float]] = None,
    ) -> dict:
        drift_alerts = self.drift_detector.full_drift_scan(
            returns, current_weights, target_weights,
            corr_data=pd.DataFrame(returns_dict) if returns_dict else None,
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
        return self.visualizer.plot_static_pnl_surface(
            pnl_df, "fast", "slow", "sharpe",
            "Strategy PnL Surface: Sharpe Ratio",
        )

    def generate_animated_surface(self, time_surfaces: Dict[str, pd.DataFrame]):
        return self.visualizer.plot_animated_pnl_surface(
            time_surfaces, "fast", "slow", "sharpe",
            "Time-Evolving PnL Surface",
        )

    # ── Full Scan ────────────────────────────────────────────────────────

    def full_scan(self) -> dict:
        scan_start = datetime.now(timezone.utc)
        logger.info("=" * 70)
        logger.info("FULL MULTI-MARKET SCAN — %s", scan_start.isoformat())
        logger.info("=" * 70)

        # 1. Fetch data (resilient — each source independent)
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

        # 3. Backtest top signals
        backtest_results = {}
        pnl_surfaces = {}
        time_surfaces = {}

        for symbol, prices in list(stock_prices.items())[:5]:
            if prices.empty or len(prices) < 50:
                continue

            for strat_name in ["momentum", "mean_reversion", "regime_adaptive"]:
                bt = self.run_strategy_backtest(prices, strat_name)
                if bt:
                    backtest_results[f"{symbol}_{strat_name}"] = bt

            sweep = self.run_parameter_sweep(prices)
            if sweep:
                pnl_surfaces[symbol] = sweep.get("pnl_surface")

            ts = self.run_time_evolving_sweep(prices, n_periods=4)
            if ts:
                time_surfaces[symbol] = ts

        # 4. Portfolio construction
        returns_dict = {}
        for sym, prices in stock_prices.items():
            if not prices.empty and "close" in prices.columns:
                returns_dict[sym] = prices["close"].pct_change().dropna()

        portfolio_weights = {}
        if len(returns_dict) >= 2:
            for method in ["risk_parity", "max_diversification"]:
                portfolio_weights[method] = self.construct_portfolio(
                    returns_dict, method
                )

        # 5. Risk scan
        risk_scan = {}
        if returns_dict:
            df = pd.DataFrame(returns_dict).dropna()
            if not df.empty:
                portfolio_ret = df.mean(axis=1)
                portfolio_eq = 100000 * (1 + portfolio_ret).cumprod()
                rp_weights = portfolio_weights.get("risk_parity", {})
                equal_weights = {s: 1.0/len(returns_dict) for s in returns_dict}

                risk_scan = self.run_risk_scan(
                    portfolio_ret, portfolio_eq,
                    returns_dict=returns_dict,
                    current_weights=equal_weights,
                    target_weights=rp_weights if rp_weights else equal_weights,
                )

        scan_duration = (datetime.now(timezone.utc) - scan_start).total_seconds()

        self.last_scan = scan_start
        self.scan_results = {
            "prediction_markets": pred_markets,
            "alpha_signals": alpha,
            "backtest_results": backtest_results,
            "pnl_surfaces": pnl_surfaces,
            "time_surfaces": time_surfaces,
            "portfolio_weights": portfolio_weights,
            "risk": risk_scan,
            "scan_time": scan_start,
            "scan_duration_s": scan_duration,
        }

        self._print_summary()
        return self.scan_results

    def _print_summary(self):
        r = self.scan_results
        print("\n" + "=" * 70)
        print(f"SCAN COMPLETE — {r.get('scan_time', '')} ({r.get('scan_duration_s', 0):.1f}s)")
        print("=" * 70)

        # Alpha signals
        alpha = r.get("alpha_signals", {})
        cs = alpha.get("cross_market_signals", pd.DataFrame())
        if not cs.empty:
            print("\n── TOP ALPHA SIGNALS ──")
            display_cols = [c for c in ["symbol", "asset_type", "composite_alpha",
                                         "momentum", "sentiment", "hurst_exponent",
                                         "regime", "volatility_20d"] if c in cs.columns]
            print(cs[display_cols].head(10).to_string(index=False))

        pe = alpha.get("prediction_edge", pd.DataFrame())
        if not pe.empty:
            print("\n── PREDICTION MARKET EDGE ──")
            print(pe.head(5).to_string(index=False))

        # Backtest
        bt = r.get("backtest_results", {})
        if bt:
            print("\n── BACKTEST RESULTS ──")
            for key, res in bt.items():
                m = res.get("metrics", {})
                pbo = res.get("pbo", {})
                dsr = m.get("deflated_sharpe", {})
                print(f"  {key}: Sharpe={m.get('sharpe_ratio', 0):.2f}, "
                      f"Return={m.get('total_return', 0):.1%}, "
                      f"MaxDD={m.get('max_drawdown', 0):.1%}, "
                      f"Sortino={m.get('sortino_ratio', 0):.2f}, "
                      f"PF={m.get('profit_factor', 0):.2f}")
                if pbo:
                    print(f"    PBO={pbo.get('pbo', 'N/A'):.0%}, "
                          f"OOS Sharpe={pbo.get('mean_oos_sharpe', 0):.2f}")

        # Portfolio
        pw = r.get("portfolio_weights", {})
        if pw:
            print("\n── OPTIMAL PORTFOLIO ──")
            for method, weights in pw.items():
                if weights:
                    print(f"  [{method}]: {', '.join(f'{k}={v:.1%}' for k, v in sorted(weights.items(), key=lambda x: -x[1])[:5])}")

        # Risk
        risk = r.get("risk", {})
        if risk:
            n_alerts = risk.get("total_alerts", 0)
            metrics = risk.get("risk_metrics", {})
            print(f"\n── RISK ({n_alerts} alerts) ──")
            if metrics:
                print(f"  VaR(99%): hist={metrics.get('var_99_hist', 0):.2%}, "
                      f"EVT={metrics.get('var_99_evt', 0):.2%}, "
                      f"CVaR={metrics.get('cvar_975', 0):.2%}")
                print(f"  Vol: EWMA={metrics.get('ewma_vol', 0):.1%}, "
                      f"21d={metrics.get('rolling_vol_21d', 0):.1%}")
                print(f"  Tail: kurtosis={metrics.get('kurtosis', 0):.1f}, "
                      f"skew={metrics.get('skewness', 0):.2f}, "
                      f"Hill α={metrics.get('hill_tail_index', 0):.1f}")

            for a in (risk.get("drift_alerts", []) + risk.get("tail_risk_alerts", []))[:5]:
                print(f"  [{a.severity.upper()}] {a.message}")

        print("=" * 70 + "\n")

    def run_continuous(self, interval: int = SCAN_INTERVAL_SECONDS):
        logger.info("Starting continuous screener (interval=%ds)", interval)
        while True:
            try:
                self.full_scan()
            except KeyboardInterrupt:
                logger.info("Screener stopped by user")
                break
            except Exception as e:
                logger.error("Scan failed: %s", e, exc_info=True)
            time.sleep(interval)
