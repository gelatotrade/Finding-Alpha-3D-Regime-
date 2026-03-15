"""
Institutional Tail Risk Early Warning Screener.

Features:
- VaR/CVaR: historical, parametric, Cornish-Fisher, EVT (GPD)
- EWMA volatility (exponentially weighted, GARCH-like)
- Drawdown monitoring with speed and recovery tracking
- Tail distribution: kurtosis, skewness, Hill estimator
- Cross-asset contagion (dynamic correlation + tail dependence)
- Liquidity stress proxy
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from config.settings import (
    TAIL_RISK_VAR_CONFIDENCE,
    TAIL_RISK_CVAR_CONFIDENCE,
    MAX_DRAWDOWN_ALERT_PCT,
    VOLATILITY_SPIKE_THRESHOLD,
    MIN_OBSERVATIONS_FOR_STATS,
)

logger = logging.getLogger(__name__)


@dataclass
class TailRiskAlert:
    alert_type: str
    severity: str       # "watch", "warning", "danger", "extreme"
    metric: str
    value: float
    threshold: float
    message: str
    timestamp: Optional[pd.Timestamp] = None


class TailRiskScreener:
    """Institutional-grade tail risk early warning system."""

    def __init__(self):
        self.alerts: List[TailRiskAlert] = []

    # ── Value at Risk ────────────────────────────────────────────────────

    def compute_var(
        self, returns: pd.Series, confidence: float = TAIL_RISK_VAR_CONFIDENCE,
        method: str = "historical",
    ) -> float:
        returns = returns.dropna()
        if len(returns) < MIN_OBSERVATIONS_FOR_STATS:
            return 0.0

        alpha = 1 - confidence

        if method == "historical":
            return float(np.percentile(returns, alpha * 100))

        elif method == "parametric":
            mu = returns.mean()
            sigma = returns.std()
            return float(mu + sp_stats.norm.ppf(alpha) * sigma)

        elif method == "cornish_fisher":
            mu = returns.mean()
            sigma = returns.std()
            s = sp_stats.skew(returns)
            k = sp_stats.kurtosis(returns)
            z = sp_stats.norm.ppf(alpha)
            z_cf = (z + (z**2 - 1) * s / 6
                    + (z**3 - 3*z) * (k) / 24
                    - (2*z**3 - 5*z) * s**2 / 36)
            result = float(mu + z_cf * sigma)
            # Sanity: CF can fail for extreme non-normality
            hist_var = float(np.percentile(returns, alpha * 100))
            if result > 0:  # VaR should be negative for losses
                return hist_var
            return result

        elif method == "evt":
            return self._evt_var(returns, confidence)

        return 0.0

    def _evt_var(self, returns: pd.Series, confidence: float) -> float:
        """
        Extreme Value Theory VaR using Generalized Pareto Distribution.
        Fits GPD to tail losses exceeding a threshold (10th percentile).
        """
        losses = -returns.dropna()
        if len(losses) < 50:
            return float(-np.percentile(returns.dropna(), (1 - confidence) * 100))

        # Threshold: 90th percentile of losses
        threshold = np.percentile(losses, 90)
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < 10:
            return float(-np.percentile(returns.dropna(), (1 - confidence) * 100))

        try:
            # Fit GPD to exceedances
            shape, loc, scale = sp_stats.genpareto.fit(exceedances, floc=0)

            # GPD VaR
            n = len(losses)
            n_u = len(exceedances)
            alpha = 1 - confidence

            var_gpd = threshold + (scale / shape) * (
                (n / n_u * alpha) ** (-shape) - 1
            )
            return float(-var_gpd)
        except (RuntimeError, ValueError):
            return float(-np.percentile(returns.dropna(), (1 - confidence) * 100))

    def compute_cvar(
        self, returns: pd.Series, confidence: float = TAIL_RISK_CVAR_CONFIDENCE,
    ) -> float:
        """Expected Shortfall (CVaR)."""
        returns = returns.dropna()
        if len(returns) < MIN_OBSERVATIONS_FOR_STATS:
            return 0.0
        var = self.compute_var(returns, confidence)
        tail = returns[returns <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    # ── EWMA Volatility (GARCH-like) ────────────────────────────────────

    def ewma_volatility(
        self, returns: pd.Series, decay: float = 0.94, annualize: bool = True,
    ) -> pd.Series:
        """
        Exponentially Weighted Moving Average volatility.
        RiskMetrics standard: decay=0.94 for daily data.
        """
        returns = returns.dropna()
        if len(returns) < 5:
            return pd.Series(dtype=float)

        var = returns.iloc[0] ** 2
        ewma_vars = [var]

        for r in returns.iloc[1:]:
            var = decay * var + (1 - decay) * r**2
            ewma_vars.append(var)

        vol = pd.Series(np.sqrt(ewma_vars), index=returns.index)
        if annualize:
            vol = vol * np.sqrt(252)
        return vol

    # ── Volatility Monitoring ────────────────────────────────────────────

    def detect_vol_spike(
        self,
        returns: pd.Series,
        short_window: int = 5,
        long_window: int = 63,
        threshold: float = VOLATILITY_SPIKE_THRESHOLD,
    ) -> List[TailRiskAlert]:
        alerts = []
        if len(returns) < long_window + 10:
            return alerts

        # Use EWMA for short-term, rolling for long-term
        ewma_vol = self.ewma_volatility(returns, annualize=False)
        long_vol = returns.rolling(long_window).std()
        long_vol_mean = long_vol.rolling(long_window).mean()
        long_vol_std = long_vol.rolling(long_window).std()

        latest_ewma = ewma_vol.iloc[-1] if not ewma_vol.empty else 0
        latest_long = long_vol.iloc[-1] if not long_vol.empty else 0
        latest_long_std = long_vol_std.iloc[-1] if not long_vol_std.empty else 0

        if latest_long_std > 0:
            z_score = (latest_ewma - long_vol_mean.iloc[-1]) / latest_long_std
        else:
            z_score = 0

        if not np.isnan(z_score) and z_score > threshold:
            if z_score > threshold * 2:
                severity = "extreme"
            elif z_score > threshold * 1.5:
                severity = "danger"
            else:
                severity = "warning"

            alerts.append(TailRiskAlert(
                alert_type="vol_spike",
                severity=severity,
                metric="vol_z_score",
                value=float(z_score),
                threshold=threshold,
                message=f"Vol spike: z={z_score:.2f} "
                        f"(EWMA={latest_ewma*np.sqrt(252):.1%}, "
                        f"long={latest_long*np.sqrt(252):.1%})",
                timestamp=pd.Timestamp.now(),
            ))

        self.alerts.extend(alerts)
        return alerts

    # ── Drawdown Monitoring ──────────────────────────────────────────────

    def monitor_drawdown(
        self,
        equity: pd.Series,
        alert_threshold: float = MAX_DRAWDOWN_ALERT_PCT / 100,
    ) -> List[TailRiskAlert]:
        alerts = []
        if len(equity) < 5:
            return alerts

        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        current_dd = float(drawdown.iloc[-1])

        if current_dd < -alert_threshold:
            if current_dd < -alert_threshold * 2:
                severity = "extreme"
            elif current_dd < -alert_threshold * 1.5:
                severity = "danger"
            else:
                severity = "warning"

            # Drawdown speed
            dd_speed = float(drawdown.diff().iloc[-5:].mean())

            # Drawdown duration
            in_dd = drawdown < 0
            if in_dd.iloc[-1]:
                dd_start = in_dd[::-1].idxmin()
                dd_duration = len(drawdown.loc[dd_start:])
            else:
                dd_duration = 0

            alerts.append(TailRiskAlert(
                alert_type="drawdown",
                severity=severity,
                metric="current_drawdown",
                value=current_dd,
                threshold=-alert_threshold,
                message=f"Drawdown: {current_dd:.1%} (threshold={-alert_threshold:.1%}, "
                        f"speed={dd_speed:.3%}/day, duration={dd_duration}d)",
                timestamp=pd.Timestamp.now(),
            ))

        self.alerts.extend(alerts)
        return alerts

    # ── Tail Distribution (Hill Estimator) ───────────────────────────────

    def hill_tail_index(self, returns: pd.Series, k: Optional[int] = None) -> float:
        """
        Hill estimator for tail index α.
        α < 2 → infinite variance (extremely fat tails)
        α < 4 → infinite kurtosis
        Lower α = fatter tails = more tail risk.
        """
        losses = (-returns.dropna()).sort_values(ascending=False)
        n = len(losses)
        if n < 50:
            return float('inf')

        if k is None:
            k = int(np.sqrt(n))  # standard choice

        k = min(k, n - 1)
        top_k = losses.iloc[:k].values
        threshold = losses.iloc[k]

        if threshold <= 0:
            return float('inf')

        log_ratios = np.log(top_k / threshold)
        hill_estimate = 1.0 / np.mean(log_ratios) if np.mean(log_ratios) > 0 else float('inf')
        return float(hill_estimate)

    def check_tail_risk_indicators(
        self, returns: pd.Series, window: int = 63
    ) -> List[TailRiskAlert]:
        alerts = []
        if len(returns) < window:
            return alerts

        recent = returns.iloc[-window:].dropna()
        if len(recent) < MIN_OBSERVATIONS_FOR_STATS:
            return alerts

        kurt = sp_stats.kurtosis(recent)
        skew = sp_stats.skew(recent)
        hill = self.hill_tail_index(recent)

        if kurt > 5:
            alerts.append(TailRiskAlert(
                alert_type="fat_tails",
                severity="danger" if kurt > 8 else "warning",
                metric="kurtosis",
                value=float(kurt),
                threshold=5.0,
                message=f"Fat tails: kurtosis={kurt:.1f} (normal=0). "
                        f"Hill tail index α={hill:.1f}",
                timestamp=pd.Timestamp.now(),
            ))

        if skew < -1.0:
            alerts.append(TailRiskAlert(
                alert_type="negative_skew",
                severity="danger" if skew < -2 else "warning",
                metric="skewness",
                value=float(skew),
                threshold=-1.0,
                message=f"Negative skew: {skew:.2f}. "
                        f"Downside risk is asymmetrically elevated.",
                timestamp=pd.Timestamp.now(),
            ))

        if hill < 3:
            alerts.append(TailRiskAlert(
                alert_type="tail_index",
                severity="danger" if hill < 2 else "warning",
                metric="hill_tail_index",
                value=float(hill),
                threshold=3.0,
                message=f"Extreme tail risk: Hill α={hill:.2f} "
                        f"({'infinite variance' if hill < 2 else 'fat tails'})",
                timestamp=pd.Timestamp.now(),
            ))

        self.alerts.extend(alerts)
        return alerts

    # ── Cross-Asset Contagion ────────────────────────────────────────────

    def detect_contagion(
        self,
        returns_dict: Dict[str, pd.Series],
        window: int = 21,
        corr_spike_threshold: float = 0.8,
    ) -> List[TailRiskAlert]:
        """Detect correlation spikes and tail dependence."""
        alerts = []
        if len(returns_dict) < 2:
            return alerts

        df = pd.DataFrame(returns_dict).dropna()
        if len(df) < window * 2:
            return alerts

        recent_corr = df.iloc[-window:].corr()
        hist_corr = df.iloc[:-window].corr()

        upper = np.triu_indices_from(recent_corr.values, k=1)
        if len(upper[0]) == 0:
            return alerts

        recent_upper = recent_corr.values[upper]
        hist_upper = hist_corr.values[upper]

        mean_recent = float(np.nanmean(recent_upper))
        mean_hist = float(np.nanmean(hist_upper))

        if mean_recent > corr_spike_threshold and mean_recent > mean_hist + 0.15:
            alerts.append(TailRiskAlert(
                alert_type="contagion",
                severity="danger" if mean_recent > 0.9 else "warning",
                metric="mean_correlation",
                value=mean_recent,
                threshold=corr_spike_threshold,
                message=f"Contagion: mean corr spiked to {mean_recent:.2f} "
                        f"(historical: {mean_hist:.2f}). "
                        f"Diversification breakdown.",
                timestamp=pd.Timestamp.now(),
            ))

        # Tail dependence: % of days both assets are in their 5th percentile
        n_assets = len(df.columns)
        if n_assets >= 2 and len(df) >= 50:
            recent_df = df.iloc[-window:]
            thresholds = df.quantile(0.05)
            tail_events = (recent_df < thresholds).sum(axis=1)
            joint_tail_pct = float((tail_events >= 2).mean())

            # Under independence: P(joint) = P(A) * P(B) = 0.05^2 = 0.0025
            expected_joint = 0.05 ** 2 * n_assets * (n_assets - 1) / 2
            if joint_tail_pct > 0.05:  # much higher than expected
                alerts.append(TailRiskAlert(
                    alert_type="tail_dependence",
                    severity="danger",
                    metric="joint_tail_pct",
                    value=joint_tail_pct,
                    threshold=0.05,
                    message=f"Tail dependence: {joint_tail_pct:.1%} of days have "
                            f"joint tail events (expected: {expected_joint:.1%})",
                    timestamp=pd.Timestamp.now(),
                ))

        self.alerts.extend(alerts)
        return alerts

    # ── Full Scan ────────────────────────────────────────────────────────

    def full_tail_risk_scan(
        self,
        returns: pd.Series,
        equity: Optional[pd.Series] = None,
        returns_dict: Optional[Dict[str, pd.Series]] = None,
    ) -> List[TailRiskAlert]:
        all_alerts = []

        # VaR breach check
        var_99 = self.compute_var(returns, TAIL_RISK_VAR_CONFIDENCE)
        var_99_evt = self.compute_var(returns, TAIL_RISK_VAR_CONFIDENCE, "evt")
        cvar_975 = self.compute_cvar(returns, TAIL_RISK_CVAR_CONFIDENCE)

        latest_return = float(returns.iloc[-1]) if len(returns) > 0 else 0
        if latest_return < var_99:
            all_alerts.append(TailRiskAlert(
                alert_type="var_breach",
                severity="danger",
                metric="daily_return_vs_var99",
                value=latest_return,
                threshold=var_99,
                message=f"VaR breach: return {latest_return:.2%} "
                        f"< VaR(99%) hist={var_99:.2%}, EVT={var_99_evt:.2%}",
                timestamp=pd.Timestamp.now(),
            ))

        all_alerts.extend(self.detect_vol_spike(returns))
        if equity is not None:
            all_alerts.extend(self.monitor_drawdown(equity))
        all_alerts.extend(self.check_tail_risk_indicators(returns))
        if returns_dict:
            all_alerts.extend(self.detect_contagion(returns_dict))

        severity_order = {"extreme": 0, "danger": 1, "warning": 2, "watch": 3}
        all_alerts.sort(key=lambda a: severity_order.get(a.severity, 4))

        logger.info(
            "Tail risk scan: VaR(99%%)=%.2f%%, EVT-VaR=%.2f%%, CVaR(97.5%%)=%.2f%%, %d alerts",
            var_99 * 100, var_99_evt * 100, cvar_975 * 100, len(all_alerts),
        )
        return all_alerts

    def get_risk_dashboard_data(self, returns: pd.Series) -> dict:
        returns = returns.dropna()
        if len(returns) < MIN_OBSERVATIONS_FOR_STATS:
            return {}

        ewma_vol = self.ewma_volatility(returns)

        return {
            "var_99_hist": self.compute_var(returns, 0.99, "historical"),
            "var_99_param": self.compute_var(returns, 0.99, "parametric"),
            "var_99_cf": self.compute_var(returns, 0.99, "cornish_fisher"),
            "var_99_evt": self.compute_var(returns, 0.99, "evt"),
            "cvar_975": self.compute_cvar(returns, 0.975),
            "ewma_vol": float(ewma_vol.iloc[-1]) if not ewma_vol.empty else 0,
            "rolling_vol_21d": float(returns.iloc[-21:].std() * np.sqrt(252)),
            "rolling_vol_63d": float(returns.std() * np.sqrt(252)),
            "kurtosis": float(sp_stats.kurtosis(returns)),
            "skewness": float(sp_stats.skew(returns)),
            "hill_tail_index": self.hill_tail_index(returns),
            "worst_day": float(returns.min()),
            "best_day": float(returns.max()),
            "pct_negative_days": float((returns < 0).mean()),
        }
