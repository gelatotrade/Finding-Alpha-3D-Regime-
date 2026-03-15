"""
Backtesting Engine — vectorized backtester for multi-asset strategies.
Supports long/short signals, transaction costs, slippage, and
generates full PnL time-series for 3D surface visualization.
"""
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from config.settings import (
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_COMMISSION_PCT,
    BACKTEST_SLIPPAGE_PCT,
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
    pnl_by_param: Optional[pd.DataFrame] = None  # for 3D surface


@dataclass
class Strategy:
    """A trading strategy definition."""
    name: str
    signal_fn: Callable[[pd.DataFrame], pd.Series]
    params: dict = field(default_factory=dict)


class BacktestEngine:
    """Vectorized backtesting engine with full PnL tracking."""

    def __init__(
        self,
        initial_capital: float = BACKTEST_INITIAL_CAPITAL,
        commission: float = BACKTEST_COMMISSION_PCT,
        slippage: float = BACKTEST_SLIPPAGE_PCT,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(
        self, prices: pd.DataFrame, strategy: Strategy
    ) -> BacktestResult:
        """
        Run a backtest on a single asset.

        Parameters
        ----------
        prices : DataFrame with 'close' column and DatetimeIndex
        strategy : Strategy with signal_fn returning Series of positions [-1, 0, 1]
        """
        if prices.empty or "close" not in prices.columns:
            raise ValueError("prices must have a 'close' column")

        close = prices["close"].copy()
        signal = strategy.signal_fn(prices).reindex(close.index).fillna(0)
        signal = signal.clip(-1, 1)

        # Detect trades (position changes)
        trades_mask = signal.diff().fillna(0) != 0
        trade_costs = trades_mask.astype(float) * (self.commission + self.slippage)

        # Daily returns
        daily_returns = close.pct_change().fillna(0)

        # Strategy returns = position * market return - costs
        strategy_returns = signal.shift(1).fillna(0) * daily_returns - trade_costs

        # Equity curve
        equity = self.initial_capital * (1 + strategy_returns).cumprod()

        # Build trades log
        trade_indices = trades_mask[trades_mask].index
        trade_records = []
        for i, idx in enumerate(trade_indices):
            trade_records.append({
                "date": idx,
                "position": signal.loc[idx],
                "price": close.loc[idx],
                "trade_cost": trade_costs.loc[idx] * close.loc[idx],
            })
        trades_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()

        # Positions
        positions_df = pd.DataFrame({"position": signal, "close": close}, index=close.index)

        # Compute metrics
        metrics = self._compute_metrics(strategy_returns, equity)
        metrics["strategy_name"] = strategy.name
        metrics["params"] = strategy.params

        return BacktestResult(
            equity_curve=equity,
            returns=strategy_returns,
            positions=positions_df,
            trades=trades_df,
            metrics=metrics,
        )

    def parameter_sweep(
        self,
        prices: pd.DataFrame,
        signal_fn_factory: Callable,
        param_grid: Dict[str, List],
    ) -> BacktestResult:
        """
        Sweep over a parameter grid and collect PnL surfaces.

        Parameters
        ----------
        signal_fn_factory : callable(param1, param2, ...) -> signal_fn
        param_grid : {"param_name": [values], ...} — supports 1 or 2 param dims
        """
        param_names = list(param_grid.keys())
        if len(param_names) == 1:
            return self._sweep_1d(prices, signal_fn_factory, param_names[0], param_grid[param_names[0]])
        elif len(param_names) == 2:
            return self._sweep_2d(
                prices, signal_fn_factory,
                param_names[0], param_grid[param_names[0]],
                param_names[1], param_grid[param_names[1]],
            )
        else:
            raise ValueError("parameter_sweep supports 1 or 2 dimensions")

    def _sweep_1d(self, prices, factory, pname, pvalues):
        results = []
        best_result = None
        best_sharpe = -np.inf
        for val in pvalues:
            sig_fn = factory(**{pname: val})
            strat = Strategy(name=f"{pname}={val}", signal_fn=sig_fn, params={pname: val})
            res = self.run(prices, strat)
            results.append({pname: val, "sharpe": res.metrics["sharpe_ratio"],
                            "total_return": res.metrics["total_return"],
                            "max_drawdown": res.metrics["max_drawdown"]})
            if res.metrics["sharpe_ratio"] > best_sharpe:
                best_sharpe = res.metrics["sharpe_ratio"]
                best_result = res

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
                res = self.run(prices, strat)
                results.append({
                    p1name: v1, p2name: v2,
                    "sharpe": res.metrics["sharpe_ratio"],
                    "total_return": res.metrics["total_return"],
                    "max_drawdown": res.metrics["max_drawdown"],
                    "calmar": res.metrics["calmar_ratio"],
                })
                if res.metrics["sharpe_ratio"] > best_sharpe:
                    best_sharpe = res.metrics["sharpe_ratio"]
                    best_result = res

        best_result.pnl_by_param = pd.DataFrame(results)
        return best_result

    def _compute_metrics(self, returns: pd.Series, equity: pd.Series) -> dict:
        """Compute performance metrics."""
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        n_days = len(returns)
        ann_factor = 252

        ann_return = (1 + total_return) ** (ann_factor / max(n_days, 1)) - 1
        ann_vol = returns.std() * np.sqrt(ann_factor) if returns.std() > 0 else 0

        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Drawdown
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        winning = (returns > 0).sum()
        total_trades = (returns != 0).sum()
        win_rate = winning / total_trades if total_trades > 0 else 0

        # Sortino
        downside = returns[returns < 0].std() * np.sqrt(ann_factor)
        sortino = ann_return / downside if downside > 0 else 0

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "total_trades": int(total_trades),
            "n_days": n_days,
        }

    def walk_forward(
        self,
        prices: pd.DataFrame,
        strategy: Strategy,
        train_window: int = 126,
        test_window: int = 21,
    ) -> BacktestResult:
        """
        Walk-forward backtest: train on rolling window, test out-of-sample.
        """
        close = prices["close"]
        all_returns = []

        for start in range(0, len(close) - train_window - test_window, test_window):
            train_end = start + train_window
            test_end = train_end + test_window

            test_prices = prices.iloc[train_end:test_end]
            signal = strategy.signal_fn(prices.iloc[start:test_end])
            test_signal = signal.iloc[train_window:]

            daily_ret = test_prices["close"].pct_change().fillna(0)
            strat_ret = test_signal.shift(1).fillna(0) * daily_ret
            all_returns.append(strat_ret)

        if not all_returns:
            raise ValueError("Not enough data for walk-forward")

        combined_returns = pd.concat(all_returns)
        equity = self.initial_capital * (1 + combined_returns).cumprod()
        metrics = self._compute_metrics(combined_returns, equity)
        metrics["strategy_name"] = strategy.name + " (walk-forward)"

        return BacktestResult(
            equity_curve=equity,
            returns=combined_returns,
            positions=pd.DataFrame(),
            trades=pd.DataFrame(),
            metrics=metrics,
        )
