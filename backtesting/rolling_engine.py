"""
Institutional Rolling Walk-Forward Backtester.

Features:
- Expanding/rolling window re-fitting at each rebalance
- Strict no-lookahead enforcement (signal at t uses only data up to t-1)
- Purge gap between train and test
- Newey-West HAC t-statistic for serial-correlation-robust inference
- Bootstrap t-statistic distribution
- Parameter optimization on in-sample, evaluation on out-of-sample only
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
)

logger = logging.getLogger(__name__)


@dataclass
class RollingBacktestResult:
    """Result from rolling walk-forward backtest."""
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    forecasts: pd.DataFrame
    metrics: dict
    in_sample_periods: List[Tuple[pd.Timestamp, pd.Timestamp]]
    out_of_sample_periods: List[Tuple[pd.Timestamp, pd.Timestamp]]
    param_history: List[dict] = field(default_factory=list)


# ── Newey-West HAC t-statistic ──────────────────────────────────────────

def newey_west_tstat(returns: pd.Series, lag: Optional[int] = None) -> dict:
    """
    Newey-West HAC (heteroscedasticity + autocorrelation consistent)
    t-statistic. Robust to serial correlation in returns.

    Andrews (1991) optimal bandwidth: lag = floor(4 * (N/100)^(2/9))
    """
    returns = returns.dropna()
    n = len(returns)
    if n < 10:
        return {"t_stat": 0.0, "p_value": 1.0, "sharpe": 0.0, "n": n}

    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
        lag = max(1, min(lag, n // 4))

    mean = returns.mean()
    resid = returns - mean

    # Long-run variance via Newey-West kernel
    gamma0 = (resid ** 2).mean()
    lrv = gamma0
    for k in range(1, lag + 1):
        weight = 1 - k / (lag + 1)
        gamma_k = (resid.iloc[k:].values * resid.iloc[:-k].values).mean()
        lrv += 2 * weight * gamma_k

    # Guard against negative LRV (shouldn't happen with Bartlett kernel, but...)
    lrv = max(lrv, gamma0 / 10)

    se_mean = np.sqrt(lrv / n)
    t_stat = mean / se_mean if se_mean > 0 else 0.0
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(t_stat)))

    # Annualized Sharpe
    sharpe_daily = mean / returns.std() if returns.std() > 0 else 0
    sharpe = sharpe_daily * np.sqrt(252)

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "sharpe": float(sharpe),
        "mean_daily_return": float(mean),
        "hac_se": float(se_mean),
        "nw_lag": int(lag),
        "n": int(n),
        "significant_at_1pct": bool(abs(t_stat) > 2.576),
        "significant_at_5sigma": bool(abs(t_stat) > 5.0),
    }


def bootstrap_tstat_distribution(
    returns: pd.Series, n_bootstrap: int = 1000, block_size: int = 21,
) -> dict:
    """
    Stationary block bootstrap of t-statistic distribution.
    Returns confidence intervals that don't assume normality.
    """
    returns = returns.dropna().values
    n = len(returns)
    if n < 30:
        return {}

    t_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # Block bootstrap
        n_blocks = n // block_size + 1
        starts = np.random.randint(0, n - block_size, n_blocks)
        sample = np.concatenate([returns[s:s + block_size] for s in starts])[:n]
        t_stats[i] = (sample.mean() / sample.std()) * np.sqrt(n) if sample.std() > 0 else 0

    return {
        "t_stat_5th": float(np.percentile(t_stats, 5)),
        "t_stat_50th": float(np.percentile(t_stats, 50)),
        "t_stat_95th": float(np.percentile(t_stats, 95)),
        "prob_t_gt_2": float(np.mean(t_stats > 2)),
        "prob_t_gt_5": float(np.mean(t_stats > 5)),
    }


# ── Rolling Walk-Forward Backtester ─────────────────────────────────────

class RollingBacktester:
    """
    Institutional rolling walk-forward backtester.

    Workflow per iteration:
    1. Define train/test windows with purge gap
    2. Optimize strategy parameters on in-sample
    3. Generate signals on out-of-sample using fixed optimal params
    4. Track param drift across windows
    """

    def __init__(
        self,
        initial_capital: float = BACKTEST_INITIAL_CAPITAL,
        commission_bps: float = BACKTEST_COMMISSION_BPS,
        slippage_bps: float = BACKTEST_SLIPPAGE_BPS,
        market_impact_eta: float = MARKET_IMPACT_ETA,
    ):
        self.initial_capital = initial_capital
        self.commission = commission_bps / 10_000
        self.slippage = slippage_bps / 10_000
        self.impact_eta = market_impact_eta

    def run(
        self,
        prices: pd.DataFrame,
        signal_fn_factory: Callable,
        param_grid: Dict[str, List],
        train_window: int = 252,
        test_window: int = 21,
        purge_gap: int = 5,
        optimization_metric: str = "sharpe",
        max_position: float = 1.0,
    ) -> RollingBacktestResult:
        """
        Run rolling walk-forward backtest with in-sample parameter optimization.
        """
        if prices.empty or "close" not in prices.columns:
            raise ValueError("prices must have a 'close' column")

        close = prices["close"]
        returns = close.pct_change().fillna(0)
        n = len(prices)

        # Storage
        oos_signals = pd.Series(0.0, index=prices.index)
        oos_mask = pd.Series(False, index=prices.index)
        in_sample_periods = []
        oos_periods = []
        param_history = []

        # Iterate rolling windows
        start = 0
        while start + train_window + purge_gap + test_window <= n:
            train_end = start + train_window
            test_start = train_end + purge_gap
            test_end = test_start + test_window

            train_slice = prices.iloc[start:train_end]
            # Include a bit of history before test start for signal lookback
            signal_slice = prices.iloc[start:test_end]

            # Optimize parameters on train
            best_params, best_metric = self._optimize_params(
                train_slice, signal_fn_factory, param_grid, optimization_metric,
            )

            if best_params is None:
                start += test_window
                continue

            param_history.append({
                "train_start": prices.index[start],
                "train_end": prices.index[train_end - 1],
                "test_start": prices.index[test_start],
                "test_end": prices.index[test_end - 1],
                "params": best_params,
                "is_metric": best_metric,
            })

            # Apply optimized strategy to test period
            sig_fn = signal_fn_factory(**best_params)
            full_signal = sig_fn(signal_slice).reindex(signal_slice.index).fillna(0)
            # Extract only test-period signals
            test_signal = full_signal.iloc[-test_window:]
            test_signal = test_signal.clip(-max_position, max_position)

            oos_signals.iloc[test_start:test_end] = test_signal.values
            oos_mask.iloc[test_start:test_end] = True

            in_sample_periods.append((prices.index[start], prices.index[train_end - 1]))
            oos_periods.append((prices.index[test_start], prices.index[test_end - 1]))

            start += test_window

        # Compute OOS returns with costs
        # Signal shifted by 1 to prevent look-ahead
        position_delta = oos_signals.diff().fillna(oos_signals).abs()
        rolling_vol = returns.rolling(21).std().fillna(returns.std())
        market_impact = self.impact_eta * rolling_vol * np.sqrt(position_delta)
        trade_cost = position_delta * (self.commission + self.slippage) + market_impact

        strategy_returns = oos_signals.shift(1).fillna(0) * returns - trade_cost
        # Only count returns in OOS periods
        strategy_returns = strategy_returns.where(oos_mask, 0)

        # Equity
        equity = self.initial_capital * (1 + strategy_returns).cumprod()

        # Extract only OOS returns for significance testing
        oos_returns = strategy_returns[oos_mask & (strategy_returns != 0)]

        # Compute metrics
        metrics = self._compute_rolling_metrics(strategy_returns, equity, oos_returns)

        return RollingBacktestResult(
            equity_curve=equity,
            returns=strategy_returns,
            positions=oos_signals,
            forecasts=pd.DataFrame(),
            metrics=metrics,
            in_sample_periods=in_sample_periods,
            out_of_sample_periods=oos_periods,
            param_history=param_history,
        )

    # ── In-sample Parameter Optimization ────────────────────────────────

    def _optimize_params(
        self,
        prices: pd.DataFrame,
        signal_fn_factory: Callable,
        param_grid: Dict[str, List],
        metric: str = "sharpe",
    ) -> Tuple[Optional[dict], float]:
        """Grid-search parameters on in-sample data."""
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = [param_grid[k] for k in param_names]

        best_params = None
        best_score = -np.inf
        close = prices["close"]
        returns = close.pct_change().fillna(0)

        for combo in product(*param_values):
            params = dict(zip(param_names, combo))
            try:
                sig_fn = signal_fn_factory(**params)
                signal = sig_fn(prices).reindex(prices.index).fillna(0).clip(-1, 1)
                strat_ret = signal.shift(1).fillna(0) * returns
                # Subtract estimated costs
                pos_delta = signal.diff().fillna(0).abs()
                cost = pos_delta * (self.commission + self.slippage)
                strat_ret = strat_ret - cost

                if strat_ret.std() == 0:
                    continue

                if metric == "sharpe":
                    score = strat_ret.mean() / strat_ret.std() * np.sqrt(252)
                elif metric == "tstat":
                    nw = newey_west_tstat(strat_ret)
                    score = nw["t_stat"]
                elif metric == "sortino":
                    downside = strat_ret[strat_ret < 0].std()
                    score = strat_ret.mean() / downside * np.sqrt(252) if downside > 0 else 0
                else:
                    score = strat_ret.sum()

                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_params = params
            except Exception:
                continue

        return best_params, float(best_score) if best_score > -np.inf else 0.0

    # ── Metrics ─────────────────────────────────────────────────────────

    def _compute_rolling_metrics(
        self, all_returns: pd.Series, equity: pd.Series, oos_returns: pd.Series,
    ) -> dict:
        """Compute institutional metrics focusing on OOS results."""
        if len(oos_returns) < 10:
            return {"error": "insufficient OOS data"}

        total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
        ann_factor = 252
        n_days = len(oos_returns)

        ann_return = oos_returns.mean() * ann_factor
        ann_vol = oos_returns.std() * np.sqrt(ann_factor)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Drawdown
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_dd = float(drawdown.min())

        # Newey-West HAC
        nw = newey_west_tstat(oos_returns)

        # Bootstrap distribution
        bs = bootstrap_tstat_distribution(oos_returns, n_bootstrap=500)

        # Win rate
        winning = (oos_returns > 0).sum()
        losing = (oos_returns < 0).sum()
        win_rate = winning / (winning + losing) if (winning + losing) > 0 else 0

        return {
            "total_return": float(total_ret),
            "oos_annualized_return": float(ann_return),
            "oos_annualized_volatility": float(ann_vol),
            "oos_sharpe": float(sharpe),
            "max_drawdown": max_dd,
            "calmar": float(ann_return / abs(max_dd)) if max_dd != 0 else 0,
            "win_rate": float(win_rate),
            "n_oos_days": n_days,
            "hac_t_stat": nw["t_stat"],
            "hac_p_value": nw["p_value"],
            "hac_significant_1pct": nw["significant_at_1pct"],
            "hac_significant_5sigma": nw["significant_at_5sigma"],
            "hac_lag": nw["nw_lag"],
            "bootstrap": bs,
        }
