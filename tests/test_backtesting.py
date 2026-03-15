"""Tests for the backtesting engine."""
import numpy as np
import pandas as pd
import pytest
from backtesting.engine import BacktestEngine, Strategy
from screener.strategies import momentum_crossover, mean_reversion_bollinger


def _make_prices(n=252, seed=42):
    np.random.seed(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    returns = np.random.normal(0.0005, 0.015, n)
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "close": close,
        "high": close * (1 + np.abs(np.random.normal(0, 0.005, n))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.005, n))),
        "open": close * (1 + np.random.normal(0, 0.003, n)),
        "volume": np.random.randint(1e6, 1e7, n),
    }, index=dates)


class TestBacktestEngine:
    def test_basic_backtest(self):
        engine = BacktestEngine()
        prices = _make_prices()
        strat = Strategy("test", momentum_crossover(10, 30))
        result = engine.run(prices, strat)

        assert result.equity_curve is not None
        assert len(result.equity_curve) == len(prices)
        assert result.metrics["n_days"] == len(prices)
        assert -1 <= result.metrics["max_drawdown"] <= 0
        assert result.metrics["total_trades"] >= 0

    def test_no_lookahead_bias(self):
        """Signal at time t should use data up to t, not t+1."""
        engine = BacktestEngine()
        prices = _make_prices()

        # Constant signal = 1 → should match buy-and-hold minus costs
        strat = Strategy("always_long", lambda p: pd.Series(1.0, index=p.index))
        result = engine.run(prices, strat)

        # Returns should be shifted by 1 (signal.shift(1))
        # First day return should be 0 (no prior signal)
        assert result.returns.iloc[0] == 0.0

    def test_transaction_costs_reduce_returns(self):
        engine_free = BacktestEngine(commission_bps=0, slippage_bps=0, market_impact_eta=0)
        engine_costly = BacktestEngine(commission_bps=50, slippage_bps=20, market_impact_eta=0.5)
        prices = _make_prices()
        strat = Strategy("test", momentum_crossover(5, 15))

        result_free = engine_free.run(prices, strat)
        result_costly = engine_costly.run(prices, strat)

        assert result_free.metrics["total_return"] >= result_costly.metrics["total_return"]

    def test_parameter_sweep(self):
        engine = BacktestEngine()
        prices = _make_prices()

        from screener.strategies import momentum_crossover_factory
        result = engine.parameter_sweep(
            prices,
            signal_fn_factory=momentum_crossover_factory,
            param_grid={"fast": [5, 10], "slow": [20, 30]},
        )

        assert result.pnl_by_param is not None
        assert len(result.pnl_by_param) == 4  # 2x2 grid
        assert "deflated_sharpe" in result.metrics

    def test_deflated_sharpe(self):
        engine = BacktestEngine()
        prices = _make_prices()
        strat = Strategy("test", momentum_crossover(10, 30))
        result = engine.run(prices, strat)

        dsr = engine.deflated_sharpe_ratio(
            result.metrics["sharpe_ratio"], 10, result.returns
        )
        assert "deflated_sharpe" in dsr
        assert "p_value" in dsr
        assert 0 <= dsr["p_value"] <= 1

    def test_monte_carlo(self):
        engine = BacktestEngine()
        prices = _make_prices()
        strat = Strategy("test", momentum_crossover(10, 30))
        result = engine.run(prices, strat)

        mc = engine.monte_carlo_simulation(result.returns, n_simulations=100)
        assert mc["n_simulations"] == 100
        assert 0 <= mc["prob_positive_return"] <= 1
        assert mc["return_5th"] <= mc["return_median"] <= mc["return_95th"]

    def test_walk_forward(self):
        engine = BacktestEngine()
        prices = _make_prices(n=300)
        strat = Strategy("test", momentum_crossover(10, 30))

        result = engine.walk_forward(prices, strat, train_window=100,
                                      test_window=20, purge_gap=5)
        assert result.metrics["purge_gap_days"] == 5
        assert len(result.returns) > 0

    def test_cpcv(self):
        engine = BacktestEngine()
        prices = _make_prices(n=300)
        strat = Strategy("test", momentum_crossover(10, 30))

        pbo = engine.combinatorial_purged_cv(prices, strat, n_splits=4, n_test_splits=1)
        assert 0 <= pbo["pbo"] <= 1
        assert pbo["n_paths"] > 0

    def test_minimum_track_record_length(self):
        engine = BacktestEngine()
        prices = _make_prices()
        strat = Strategy("test", momentum_crossover(10, 30))
        result = engine.run(prices, strat)

        mtrl = engine.minimum_track_record_length(
            result.metrics["sharpe_ratio"], returns=result.returns
        )
        assert mtrl > 0

    def test_rejects_short_data(self):
        engine = BacktestEngine()
        prices = _make_prices(n=10)
        strat = Strategy("test", momentum_crossover(5, 10))

        with pytest.raises(ValueError, match="observations"):
            engine.run(prices, strat)


class TestMetrics:
    def test_metrics_completeness(self):
        engine = BacktestEngine()
        prices = _make_prices()
        strat = Strategy("test", momentum_crossover(10, 30))
        result = engine.run(prices, strat)
        m = result.metrics

        required = [
            "total_return", "annualized_return", "annualized_volatility",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown",
            "max_dd_duration_days", "win_rate", "profit_factor",
            "tail_ratio", "skewness", "kurtosis", "total_trades",
            "total_costs", "n_days",
        ]
        for key in required:
            assert key in m, f"Missing metric: {key}"
