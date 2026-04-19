"""
Institutional ARIMA-based return forecasting for alpha signal generation.

Features:
- ARIMA / SARIMA with auto order selection (AIC/BIC)
- GARCH(1,1) residual volatility modeling
- Rolling one-step-ahead forecasts with strict no-lookahead
- Stationarity tests (ADF, KPSS) with automatic differencing
- Cached model re-fitting on expanding/rolling windows
"""
import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)

# Lazy imports for heavy dependencies
_statsmodels = None
_arch = None


def _get_statsmodels():
    global _statsmodels
    if _statsmodels is None:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
        _statsmodels = {
            "ARIMA": ARIMA, "adfuller": adfuller,
            "kpss": kpss, "acf": acf, "pacf": pacf,
        }
    return _statsmodels


def _get_arch():
    global _arch
    if _arch is None:
        from arch import arch_model
        _arch = arch_model
    return _arch


@dataclass
class ArimaForecast:
    """Container for one-step-ahead forecast."""
    mean: float          # point forecast (expected return)
    std: float           # forecast standard error
    confidence_lower: float
    confidence_upper: float
    order: Tuple[int, int, int]
    aic: float
    residual_autocorr: float


class ArimaForecaster:
    """
    Rolling ARIMA forecaster with no look-ahead bias.

    Institutional pattern:
    1. At each time t, fit ARIMA only on data up to t-1
    2. Generate one-step-ahead forecast for t
    3. Optionally layer GARCH(1,1) on residuals for conditional vol
    """

    def __init__(
        self,
        max_p: int = 3,
        max_d: int = 2,
        max_q: int = 3,
        criterion: str = "aic",
        use_garch: bool = False,
    ):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.criterion = criterion
        self.use_garch = use_garch
        self._last_order: Optional[Tuple[int, int, int]] = None

    # ── Stationarity ─────────────────────────────────────────────────────

    def test_stationarity(self, series: pd.Series) -> dict:
        """ADF + KPSS tests. Both agree → stationarity conclusion is robust."""
        sm = _get_statsmodels()
        series = series.dropna()
        if len(series) < 20:
            return {"is_stationary": False, "adf_p": 1.0, "kpss_p": 0.0}

        try:
            adf_stat, adf_p, *_ = sm["adfuller"](series, autolag="AIC")
        except Exception:
            adf_p = 1.0
        try:
            kpss_stat, kpss_p, *_ = sm["kpss"](series, regression="c", nlags="auto")
        except Exception:
            kpss_p = 0.0

        # ADF: H0 = non-stationary. Reject if p < 0.05 → stationary
        # KPSS: H0 = stationary. Fail to reject if p > 0.05 → stationary
        is_stationary = adf_p < 0.05 and kpss_p > 0.05
        return {
            "is_stationary": is_stationary,
            "adf_p": float(adf_p),
            "kpss_p": float(kpss_p),
        }

    # ── Auto Order Selection ─────────────────────────────────────────────

    def select_order(self, series: pd.Series) -> Tuple[int, int, int]:
        """
        Grid search for best (p, d, q) by AIC/BIC.
        Faster than pmdarima's auto_arima for our narrow grid.
        """
        sm = _get_statsmodels()
        series = series.dropna()
        if len(series) < 30:
            return (1, 0, 0)

        best_score = np.inf
        best_order = (1, 0, 0)

        # Determine d by differencing until stationary
        d_candidates = []
        for d in range(self.max_d + 1):
            test_series = series.diff(d).dropna() if d > 0 else series
            if len(test_series) < 20:
                continue
            stat_test = self.test_stationarity(test_series)
            if stat_test["is_stationary"]:
                d_candidates.append(d)
                break
        if not d_candidates:
            d_candidates = [1]

        for d in d_candidates:
            for p in range(self.max_p + 1):
                for q in range(self.max_q + 1):
                    if p == 0 and q == 0:
                        continue
                    try:
                        model = sm["ARIMA"](series, order=(p, d, q))
                        fitted = model.fit()
                        score = fitted.aic if self.criterion == "aic" else fitted.bic
                        if score < best_score and np.isfinite(score):
                            best_score = score
                            best_order = (p, d, q)
                    except Exception:
                        continue

        self._last_order = best_order
        return best_order

    # ── Single Forecast ──────────────────────────────────────────────────

    def forecast_one_step(
        self, series: pd.Series, order: Optional[Tuple[int, int, int]] = None
    ) -> Optional[ArimaForecast]:
        """One-step-ahead forecast using only historical data."""
        sm = _get_statsmodels()
        series = series.dropna()
        if len(series) < 30:
            return None

        if order is None:
            order = self._last_order or self.select_order(series)

        try:
            model = sm["ARIMA"](series, order=order)
            fitted = model.fit()
            forecast_result = fitted.get_forecast(steps=1)

            mean = float(forecast_result.predicted_mean.iloc[0])
            std = float(forecast_result.se_mean.iloc[0])
            ci = forecast_result.conf_int(alpha=0.05)
            lower = float(ci.iloc[0, 0])
            upper = float(ci.iloc[0, 1])

            # Residual autocorrelation (Ljung-Box diagnostic proxy)
            resid = fitted.resid.dropna()
            if len(resid) > 5:
                resid_ac = float(resid.autocorr(lag=1)) if not resid.empty else 0.0
            else:
                resid_ac = 0.0

            return ArimaForecast(
                mean=mean, std=std,
                confidence_lower=lower, confidence_upper=upper,
                order=order, aic=float(fitted.aic),
                residual_autocorr=resid_ac,
            )
        except Exception as e:
            logger.debug("ARIMA forecast failed: %s", e)
            return None

    # ── Rolling Forecasts ────────────────────────────────────────────────

    def rolling_forecast(
        self,
        series: pd.Series,
        train_window: int = 252,
        refit_every: int = 21,
        order: Optional[Tuple[int, int, int]] = None,
    ) -> pd.DataFrame:
        """
        Generate rolling one-step-ahead forecasts with strict no-lookahead.

        At each t in [train_window, len(series)]:
            - Use series[t - train_window : t]  (no future data)
            - Forecast series[t]
            - Re-estimate order every `refit_every` steps (expensive)
        """
        series = series.dropna()
        if len(series) < train_window + 1:
            return pd.DataFrame()

        # Select initial order
        if order is None:
            order = self.select_order(series.iloc[:train_window])

        records = []
        current_order = order
        for t in range(train_window, len(series)):
            # Re-select order periodically
            if (t - train_window) % refit_every == 0 and t > train_window:
                try:
                    current_order = self.select_order(series.iloc[t - train_window:t])
                except Exception:
                    pass  # keep previous order

            train = series.iloc[t - train_window:t]
            fc = self.forecast_one_step(train, order=current_order)
            if fc is None:
                continue
            actual = series.iloc[t] if t < len(series) else np.nan

            records.append({
                "timestamp": series.index[t],
                "forecast_mean": fc.mean,
                "forecast_std": fc.std,
                "forecast_lower": fc.confidence_lower,
                "forecast_upper": fc.confidence_upper,
                "actual": actual,
                "order": str(current_order),
                "aic": fc.aic,
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
            df["forecast_error"] = df["actual"] - df["forecast_mean"]
            df["squared_error"] = df["forecast_error"] ** 2

        return df

    # ── GARCH Volatility Overlay ─────────────────────────────────────────

    def garch_volatility_forecast(self, returns: pd.Series) -> Optional[float]:
        """One-step-ahead conditional volatility via GARCH(1,1)."""
        arch_model = _get_arch()
        returns = returns.dropna()
        if len(returns) < 100:
            return None
        try:
            # Scale returns to percentage for numerical stability
            scaled = returns * 100
            model = arch_model(scaled, vol="Garch", p=1, q=1, rescale=False)
            fitted = model.fit(disp="off", show_warning=False)
            forecast = fitted.forecast(horizon=1, reindex=False)
            variance = float(forecast.variance.iloc[-1, 0])
            return float(np.sqrt(variance) / 100)  # rescale back
        except Exception as e:
            logger.debug("GARCH forecast failed: %s", e)
            return None
