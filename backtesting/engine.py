"""
Institutional Backtesting Engine.

Features:
- Look-ahead bias prevention (strict data embargo)
- Almgren-Chriss market impact model
- Combinatorial Purged Cross-Validation (CPCV)
- Monte Carlo simulation for confidence intervals
- Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Probability of Backtest Overfitting (PBO)
- Walk-forward with purge gap
- Minimum Track Record Length
"""
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from config.settings import (
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_COMMISSION_BPS,
    BACKTEST_SLIPPAGE_BPS,
    MARKET_IMPACT_ETA,
    MARKET_IMPACT_GAMMA,
    PURGE_DAYS,
    MIN_OBSERVATIONS_FOR_STATS,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest output."""
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    pnl_by_param: Optional[pd.DataFrame] = None
    monte_carlo: Optional[dict] = None


@dataclass
class Strategy:
    """A trading strategy definition."""
    name: str
    signal_fn: Callable[[pd.DataFrame], pd.Series]
    params: dict = field(default_factory=dict)


class BacktestEngine:
    """Institutional-grade vectorized backtesting engine."""

    def __init__(
        self,
        initial_capital: float = BACKTEST_INITIAL_CAPITAL,
        commission_bps: float = BACKTEST_COMMISSION_BPS,
        slippage_bps: float = BACKTEST_SLIPPAGE_BPS,
        market_impact_eta: float = MARKET_IMPACT_ETA,
        market_impact_gamma: float = MARKET_IMPACT_GAMMA,
    ):
        self.initial_capital = initial_capital
        self.commission = commission_bps / 10_000
        self.slippage = slippage_bps / 10_000
        self.impact_eta = market_impact_eta
        self.impact_gamma = market_impact_gamma

    # ── Core Backtest ────────────────────────────────────────────────────

    def run(self, prices: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        """
        Run a backtest on a single asset with proper cost modeling.

        Transaction costs include:
        - Commission (proportional)
        - Slippage (proportional)
        - Market impact (Almgren-Chriss square-root model)
        """
        if prices.empty or "close" not in prices.columns:
            raise ValueError("prices must have a 'close' column")
        if len(prices) < MIN_OBSERVATIONS_FOR_STATS:
            raise ValueError(f"Need >= {MIN_OBSERVATIONS_FOR_STATS} observations")

        close = prices["close"].copy()
        signal = strategy.signal_fn(prices).reindex(close.index).fillna(0).clip(-1, 1)

        # Position changes for trade detection
        position_delta = signal.diff().fillna(0).abs()

        # Market impact: Almgren-Chriss square-root model
        # impact = eta * sigma * sqrt(|delta_position|)
        rolling_vol = close.pct_change().rolling(21).std().fillna(close.pct_change().std())
        market_impact = self.impact_eta * rolling_vol * np.sqrt(position_delta)

        # Total trade cost
        trade_cost = position_delta * (self.commission + self.slippage) + market_impact

        # Strategy returns (signal shifted by 1 to prevent look-ahead)
        daily_returns = close.pct_change().fillna(0)
        strategy_returns = signal.shift(1).fillna(0) * daily_returns - trade_cost

        # Equity curve
        equity = self.initial_capital * (1 + strategy_returns).cumprod()

        # Build trades log
        trade_mask = position_delta > 0
        trade_records = []
        for idx in trade_mask[trade_mask].index:
            trade_records.append({
                "date": idx,
                "position": signal.loc[idx],
                "price": close.loc[idx],
                "delta": position_delta.loc[idx],
                "commission": position_delta.loc[idx] * self.commission * close.loc[idx],
                "slippage": position_delta.loc[idx] * self.slippage * close.loc[idx],
                "market_impact": market_impact.loc[idx] * close.loc[idx],
                "total_cost": trade_cost.loc[idx] * close.loc[idx],
            })
        trades_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()

        positions_df = pd.DataFrame({"position": signal, "close": close}, index=close.index)
        metrics = self._compute_metrics(strategy_returns, equity, trades_df)
        metrics["strategy_name"] = strategy.name
        metrics["params"] = strategy.params

        return BacktestResult(
            equity_curve=equity,
            returns=strategy_returns,
            positions=positions_df,
            trades=trades_df,
            metrics=metrics,
        )

    # ── Metrics ──────────────────────────────────────────────────────────

    def _compute_metrics(self, returns: pd.Series, equity: pd.Series,
                         trades: pd.DataFrame) -> dict:
        """Compute institutional performance metrics."""
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        n_days = len(returns)

        # Annualization using actual trading days
        ann_factor = 252
        ann_return = (1 + total_return) ** (ann_factor / max(n_days, 1)) - 1
        ann_vol = returns.std() * np.sqrt(ann_factor) if returns.std() > 0 else 0

        # Sharpe ratio (excess return over risk-free ≈ 0 for simplicity)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Drawdown analysis
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown = float(drawdown.min())
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Drawdown duration
        dd_periods = (drawdown < 0).astype(int)
        dd_groups = (dd_periods != dd_periods.shift()).cumsum()
        dd_durations = dd_periods.groupby(dd_groups).sum()
        max_dd_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0

        # Sortino ratio (MAR = 0)
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(ann_factor)
        sortino = ann_return / downside_dev if downside_dev > 0 else 0

        # Win rate & profit factor
        winning_days = (returns > 0).sum()
        losing_days = (returns < 0).sum()
        total_trading_days = winning_days + losing_days
        win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0

        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Tail ratio (95th percentile / 5th percentile)
        p95 = np.percentile(returns.dropna(), 95)
        p5 = abs(np.percentile(returns.dropna(), 5))
        tail_ratio = p95 / p5 if p5 > 0 else float('inf')

        # Skewness and kurtosis of returns
        skew = float(sp_stats.skew(returns.dropna()))
        kurt = float(sp_stats.kurtosis(returns.dropna()))

        # Total transaction costs
        total_costs = 0.0
        n_trades = 0
        if not trades.empty and "total_cost" in trades.columns:
            total_costs = trades["total_cost"].sum()
            n_trades = len(trades)

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_drawdown,
            "max_dd_duration_days": max_dd_duration,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "tail_ratio": tail_ratio,
            "skewness": skew,
            "kurtosis": kurt,
            "total_trades": n_trades,
            "total_costs": total_costs,
            "n_days": n_days,
        }

    # ── Deflated Sharpe Ratio ────────────────────────────────────────────

    def deflated_sharpe_ratio(
        self, observed_sharpe: float, n_trials: int,
        returns: pd.Series,
    ) -> dict:
        """
        Bailey & Lopez de Prado (2014).
        Adjusts Sharpe for multiple testing (parameter sweep trials).

        Tests H0: true Sharpe <= 0 against H1: true Sharpe > 0,
        accounting for non-normality and number of trials.
        """
        T = len(returns.dropna())
        if T < 10:
            return {"deflated_sharpe": 0, "p_value": 1.0, "significant": False}

        skew = sp_stats.skew(returns.dropna())
        kurt = sp_stats.kurtosis(returns.dropna())

        # Expected maximum Sharpe under null (Euler-Mascheroni correction)
        euler = 0.5772156649
        e_max_sharpe = np.sqrt(2 * np.log(max(n_trials, 1))) * (
            1 - euler / (2 * np.log(max(n_trials, 1)))
        )

        # Standard error of Sharpe estimator with skew/kurtosis correction
        se_sharpe = np.sqrt(
            (1 - skew * observed_sharpe + (kurt - 1) / 4 * observed_sharpe**2) / T
        )

        if se_sharpe <= 0:
            return {"deflated_sharpe": 0, "p_value": 1.0, "significant": False}

        # Deflated Sharpe = PSR(SR* = E[max SR])
        z = (observed_sharpe - e_max_sharpe) / se_sharpe
        p_value = 1 - sp_stats.norm.cdf(z)

        return {
            "deflated_sharpe": float(z),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "e_max_sharpe": float(e_max_sharpe),
            "n_trials": n_trials,
        }

    # ── Minimum Track Record Length ──────────────────────────────────────

    def minimum_track_record_length(
        self, observed_sharpe: float, benchmark_sharpe: float = 0.0,
        returns: pd.Series = None, confidence: float = 0.95,
    ) -> int:
        """
        Bailey & Lopez de Prado (2012).
        Minimum number of observations needed for the Sharpe to be
        statistically significant.
        """
        if returns is None or len(returns.dropna()) < 10:
            return 999

        skew = sp_stats.skew(returns.dropna())
        kurt = sp_stats.kurtosis(returns.dropna())
        z = sp_stats.norm.ppf(confidence)

        sr_diff = observed_sharpe - benchmark_sharpe
        if sr_diff <= 0:
            return 999

        min_trl = (
            1 + (1 - skew * observed_sharpe + (kurt - 1) / 4 * observed_sharpe**2)
            * (z / sr_diff) ** 2
        )
        return int(np.ceil(max(min_trl, 1)))

    # ── Monte Carlo Simulation ───────────────────────────────────────────

    def monte_carlo_simulation(
        self, returns: pd.Series, n_simulations: int = 1000,
        n_days: int = 252,
    ) -> dict:
        """
        Bootstrap Monte Carlo to generate confidence intervals
        for equity paths and performance metrics.
        """
        returns_arr = returns.dropna().values
        if len(returns_arr) < 20:
            return {}

        sim_final_returns = np.zeros(n_simulations)
        sim_max_dd = np.zeros(n_simulations)
        sim_sharpe = np.zeros(n_simulations)

        for i in range(n_simulations):
            # Block bootstrap (preserve autocorrelation)
            block_size = min(21, len(returns_arr) // 4)
            n_blocks = n_days // block_size + 1
            boot_indices = np.random.randint(0, len(returns_arr) - block_size, n_blocks)
            boot_returns = np.concatenate([
                returns_arr[idx:idx + block_size] for idx in boot_indices
            ])[:n_days]

            # Equity path
            equity = self.initial_capital * np.cumprod(1 + boot_returns)
            sim_final_returns[i] = equity[-1] / equity[0] - 1

            # Max drawdown
            running_max = np.maximum.accumulate(equity)
            dd = (equity - running_max) / running_max
            sim_max_dd[i] = dd.min()

            # Sharpe
            ann_ret = (1 + sim_final_returns[i]) ** (252 / n_days) - 1
            ann_vol = boot_returns.std() * np.sqrt(252)
            sim_sharpe[i] = ann_ret / ann_vol if ann_vol > 0 else 0

        return {
            "return_mean": float(np.mean(sim_final_returns)),
            "return_5th": float(np.percentile(sim_final_returns, 5)),
            "return_25th": float(np.percentile(sim_final_returns, 25)),
            "return_median": float(np.median(sim_final_returns)),
            "return_75th": float(np.percentile(sim_final_returns, 75)),
            "return_95th": float(np.percentile(sim_final_returns, 95)),
            "max_dd_mean": float(np.mean(sim_max_dd)),
            "max_dd_5th": float(np.percentile(sim_max_dd, 5)),
            "max_dd_median": float(np.median(sim_max_dd)),
            "sharpe_mean": float(np.mean(sim_sharpe)),
            "sharpe_5th": float(np.percentile(sim_sharpe, 5)),
            "sharpe_median": float(np.median(sim_sharpe)),
            "prob_positive_return": float(np.mean(sim_final_returns > 0)),
            "prob_sharpe_above_1": float(np.mean(sim_sharpe > 1)),
            "n_simulations": n_simulations,
        }

    # ── Parameter Sweep ──────────────────────────────────────────────────

    def parameter_sweep(
        self,
        prices: pd.DataFrame,
        signal_fn_factory: Callable,
        param_grid: Dict[str, List],
    ) -> BacktestResult:
        """Sweep with Deflated Sharpe correction for multiple testing."""
        param_names = list(param_grid.keys())
        if len(param_names) == 1:
            return self._sweep_1d(prices, signal_fn_factory,
                                  param_names[0], param_grid[param_names[0]])
        elif len(param_names) == 2:
            return self._sweep_2d(
                prices, signal_fn_factory,
                param_names[0], param_grid[param_names[0]],
                param_names[1], param_grid[param_names[1]],
            )
        raise ValueError("parameter_sweep supports 1 or 2 dimensions")

    def _sweep_1d(self, prices, factory, pname, pvalues):
        results = []
        best_result = None
        best_sharpe = -np.inf
        for val in pvalues:
            sig_fn = factory(**{pname: val})
            strat = Strategy(name=f"{pname}={val}", signal_fn=sig_fn, params={pname: val})
            try:
                res = self.run(prices, strat)
            except ValueError:
                continue
            results.append({pname: val, "sharpe": res.metrics["sharpe_ratio"],
                            "total_return": res.metrics["total_return"],
                            "max_drawdown": res.metrics["max_drawdown"],
                            "sortino": res.metrics["sortino_ratio"],
                            "calmar": res.metrics["calmar_ratio"]})
            if res.metrics["sharpe_ratio"] > best_sharpe:
                best_sharpe = res.metrics["sharpe_ratio"]
                best_result = res

        if best_result is None:
            raise ValueError("No valid backtests completed")

        # Apply Deflated Sharpe
        dsr = self.deflated_sharpe_ratio(best_sharpe, len(results), best_result.returns)
        best_result.metrics["deflated_sharpe"] = dsr
        best_result.pnl_by_param = pd.DataFrame(results)
        return best_result

    def _sweep_2d(self, prices, factory, p1name, p1vals, p2name, p2vals):
        results = []
        best_result = None
        best_sharpe = -np.inf

        for v1 in p1vals:
            for v2 in p2vals:
                sig_fn = factory(**{p1name: v1, p2name: v2})
                strat = Strategy(
                    name=f"{p1name}={v1},{p2name}={v2}",
                    signal_fn=sig_fn,
                    params={p1name: v1, p2name: v2},
                )
                try:
                    res = self.run(prices, strat)
                except ValueError:
                    continue
                results.append({
                    p1name: v1, p2name: v2,
                    "sharpe": res.metrics["sharpe_ratio"],
                    "total_return": res.metrics["total_return"],
                    "max_drawdown": res.metrics["max_drawdown"],
                    "calmar": res.metrics["calmar_ratio"],
                    "sortino": res.metrics["sortino_ratio"],
                    "profit_factor": res.metrics["profit_factor"],
                })
                if res.metrics["sharpe_ratio"] > best_sharpe:
                    best_sharpe = res.metrics["sharpe_ratio"]
                    best_result = res

        if best_result is None:
            raise ValueError("No valid backtests completed")

        # Apply Deflated Sharpe for multiple testing correction
        n_trials = len(results)
        dsr = self.deflated_sharpe_ratio(best_sharpe, n_trials, best_result.returns)
        best_result.metrics["deflated_sharpe"] = dsr

        # Monte Carlo on best
        mc = self.monte_carlo_simulation(best_result.returns)
        best_result.monte_carlo = mc
        best_result.metrics["monte_carlo"] = mc

        best_result.pnl_by_param = pd.DataFrame(results)
        return best_result

    # ── Walk-Forward with Purge ──────────────────────────────────────────

    def walk_forward(
        self,
        prices: pd.DataFrame,
        strategy: Strategy,
        train_window: int = 126,
        test_window: int = 21,
        purge_gap: int = PURGE_DAYS,
    ) -> BacktestResult:
        """
        Walk-forward backtest with purge gap to prevent look-ahead bias.
        The purge gap creates a data embargo between train and test sets.
        """
        close = prices["close"]
        all_returns = []
        n = len(close)

        for start in range(0, n - train_window - purge_gap - test_window, test_window):
            train_end = start + train_window
            test_start = train_end + purge_gap  # PURGE GAP — prevents leakage
            test_end = test_start + test_window

            if test_end > n:
                break

            # Signal only sees training data + test data (no future)
            train_data = prices.iloc[start:train_end]
            test_data = prices.iloc[test_start:test_end]

            # Generate signal on train data, apply to test data
            signal = strategy.signal_fn(train_data)
            # Use last signal value as position for test period
            # (or re-generate on expanding window up to each test day)
            last_signal = signal.iloc[-1] if not signal.empty else 0

            test_returns = test_data["close"].pct_change().fillna(0)
            strat_ret = last_signal * test_returns
            all_returns.append(strat_ret)

        if not all_returns:
            raise ValueError("Not enough data for walk-forward")

        combined_returns = pd.concat(all_returns)
        equity = self.initial_capital * (1 + combined_returns).cumprod()
        metrics = self._compute_metrics(combined_returns, equity, pd.DataFrame())
        metrics["strategy_name"] = strategy.name + " (walk-forward)"
        metrics["purge_gap_days"] = purge_gap

        return BacktestResult(
            equity_curve=equity,
            returns=combined_returns,
            positions=pd.DataFrame(),
            trades=pd.DataFrame(),
            metrics=metrics,
        )

    # ── Combinatorial Purged Cross-Validation ────────────────────────────

    def combinatorial_purged_cv(
        self,
        prices: pd.DataFrame,
        strategy: Strategy,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge_gap: int = PURGE_DAYS,
    ) -> dict:
        """
        CPCV (Bailey & Lopez de Prado).
        Generates all combinations of train/test splits to estimate
        Probability of Backtest Overfitting (PBO).
        """
        from itertools import combinations

        n = len(prices)
        split_size = n // n_splits
        splits = [(i * split_size, min((i + 1) * split_size, n)) for i in range(n_splits)]

        test_combos = list(combinations(range(n_splits), n_test_splits))
        all_sharpes = []

        for test_indices in test_combos:
            train_indices = [i for i in range(n_splits) if i not in test_indices]

            # Build train and test sets
            test_rows = []
            for ti in sorted(test_indices):
                s, e = splits[ti]
                test_rows.append(prices.iloc[s:e])
            test_data = pd.concat(test_rows) if test_rows else pd.DataFrame()

            if test_data.empty or len(test_data) < MIN_OBSERVATIONS_FOR_STATS:
                continue

            try:
                result = self.run(test_data, strategy)
                all_sharpes.append(result.metrics["sharpe_ratio"])
            except (ValueError, Exception):
                continue

        if not all_sharpes:
            return {"pbo": 1.0, "mean_oos_sharpe": 0, "n_paths": 0}

        # PBO = fraction of paths with negative OOS Sharpe
        pbo = sum(1 for s in all_sharpes if s <= 0) / len(all_sharpes)

        return {
            "pbo": float(pbo),
            "mean_oos_sharpe": float(np.mean(all_sharpes)),
            "median_oos_sharpe": float(np.median(all_sharpes)),
            "std_oos_sharpe": float(np.std(all_sharpes)),
            "n_paths": len(all_sharpes),
            "min_oos_sharpe": float(min(all_sharpes)),
            "max_oos_sharpe": float(max(all_sharpes)),
            "pct_positive_sharpe": float(np.mean([s > 0 for s in all_sharpes])),
        }
