"""
Tail Risk Early Warning Screener.
Monitors for extreme market conditions and provides
early warning signals before tail events materialize.
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
)

logger = logging.getLogger(__name__)


@dataclass
class TailRiskAlert:
    """A tail risk warning."""
    alert_type: str
    severity: str       # "watch", "warning", "danger", "extreme"
    metric: str
    value: float
    threshold: float
    message: str
    timestamp: Optional[pd.Timestamp] = None


class TailRiskScreener:
    """
    Multi-dimensional tail risk early warning system.

    Monitors:
    1. VaR / CVaR breaches
    2. Volatility spikes (GARCH-like detection without heavy dependencies)
    3. Drawdown monitoring
    4. Tail index / kurtosis alerts
    5. Cross-asset contagion (correlation spikes)
    6. Liquidity stress signals
    """

    def __init__(self):
        self.alerts: List[TailRiskAlert] = []

    # ── Value at Risk ────────────────────────────────────────────────────

    def compute_var(
        self, returns: pd.Series, confidence: float = TAIL_RISK_VAR_CONFIDENCE,
        method: str = "historical",
    ) -> float:
        """Compute Value at Risk."""
        returns = returns.dropna()
        if len(returns) < 20:
            return 0.0

        if method == "historical":
            return float(np.percentile(returns, (1 - confidence) * 100))
        elif method == "parametric":
            mu = returns.mean()
            sigma = returns.std()
            z = sp_stats.norm.ppf(1 - confidence)
            return float(mu + z * sigma)
        elif method == "cornish_fisher":
            mu = returns.mean()
            sigma = returns.std()
            s = sp_stats.skew(returns)
            k = sp_stats.kurtosis(returns)
            z = sp_stats.norm.ppf(1 - confidence)
            # Cornish-Fisher expansion
            z_cf = (z + (z**2 - 1) * s / 6
                    + (z**3 - 3*z) * (k - 3) / 24
                    - (2*z**3 - 5*z) * s**2 / 36)
            return float(mu + z_cf * sigma)
        return 0.0

    def compute_cvar(
        self, returns: pd.Series, confidence: float = TAIL_RISK_CVAR_CONFIDENCE,
    ) -> float:
        """Compute Conditional VaR (Expected Shortfall)."""
        returns = returns.dropna()
        if len(returns) < 20:
            return 0.0
        var = self.compute_var(returns, confidence)
        tail_returns = returns[returns <= var]
        return float(tail_returns.mean()) if len(tail_returns) > 0 else var

    # ── Volatility Monitoring ────────────────────────────────────────────

    def detect_vol_spike(
        self,
        returns: pd.Series,
        short_window: int = 5,
        long_window: int = 63,
        threshold: float = VOLATILITY_SPIKE_THRESHOLD,
    ) -> List[TailRiskAlert]:
        """Detect volatility spikes using z-score of short vs long vol."""
        alerts = []
        if len(returns) < long_window + 10:
            return alerts

        short_vol = returns.rolling(short_window).std() * np.sqrt(252)
        long_vol = returns.rolling(long_window).std() * np.sqrt(252)
        long_vol_std = long_vol.rolling(long_window).std()

        z_score = (short_vol - long_vol) / long_vol_std.replace(0, np.nan)
        latest_z = z_score.iloc[-1]

        if not np.isnan(latest_z) and latest_z > threshold:
            if latest_z > threshold * 2:
                severity = "extreme"
            elif latest_z > threshold * 1.5:
                severity = "danger"
            else:
                severity = "warning"

            alert = TailRiskAlert(
                alert_type="vol_spike",
                severity=severity,
                metric="vol_z_score",
                value=latest_z,
                threshold=threshold,
                message=f"Volatility spike detected: z-score={latest_z:.2f} "
                        f"(short={short_vol.iloc[-1]:.1%}, long={long_vol.iloc[-1]:.1%})",
                timestamp=pd.Timestamp.now(),
            )
            alerts.append(alert)

        self.alerts.extend(alerts)
        return alerts

    # ── Drawdown Monitoring ──────────────────────────────────────────────

    def monitor_drawdown(
        self,
        equity: pd.Series,
        alert_threshold: float = MAX_DRAWDOWN_ALERT_PCT / 100,
    ) -> List[TailRiskAlert]:
        """Monitor current drawdown and speed of drawdown."""
        alerts = []
        if len(equity) < 5:
            return alerts

        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        current_dd = drawdown.iloc[-1]

        if current_dd < -alert_threshold:
            if current_dd < -alert_threshold * 2:
                severity = "extreme"
            elif current_dd < -alert_threshold * 1.5:
                severity = "danger"
            else:
                severity = "warning"

            # Drawdown speed: how fast are we drawing down?
            dd_speed = drawdown.diff().iloc[-5:].mean()

            alert = TailRiskAlert(
                alert_type="drawdown",
                severity=severity,
                metric="current_drawdown",
                value=current_dd,
                threshold=-alert_threshold,
                message=f"Drawdown alert: {current_dd:.1%} "
                        f"(threshold={-alert_threshold:.1%}, "
                        f"daily speed={dd_speed:.2%}/day)",
                timestamp=pd.Timestamp.now(),
            )
            alerts.append(alert)

        self.alerts.extend(alerts)
        return alerts

    # ── Tail Distribution Analysis ───────────────────────────────────────

    def check_tail_risk_indicators(
        self, returns: pd.Series, window: int = 63
    ) -> List[TailRiskAlert]:
        """Check kurtosis and skewness for tail risk buildup."""
        alerts = []
        if len(returns) < window:
            return alerts

        recent = returns.iloc[-window:].dropna()
        kurt = sp_stats.kurtosis(recent)
        skew = sp_stats.skew(recent)

        # Excess kurtosis > 3 signals fat tails
        if kurt > 5:
            alert = TailRiskAlert(
                alert_type="fat_tails",
                severity="danger" if kurt > 8 else "warning",
                metric="kurtosis",
                value=kurt,
                threshold=5.0,
                message=f"Fat tail warning: kurtosis={kurt:.1f} (normal=3). "
                        f"Extreme moves more likely than normal distribution implies.",
                timestamp=pd.Timestamp.now(),
            )
            alerts.append(alert)

        # Significant negative skew
        if skew < -1.0:
            alert = TailRiskAlert(
                alert_type="negative_skew",
                severity="danger" if skew < -2 else "warning",
                metric="skewness",
                value=skew,
                threshold=-1.0,
                message=f"Negative skew warning: skew={skew:.2f}. "
                        f"Returns are asymmetrically distributed to the downside.",
                timestamp=pd.Timestamp.now(),
            )
            alerts.append(alert)

        self.alerts.extend(alerts)
        return alerts

    # ── Cross-Asset Contagion ────────────────────────────────────────────

    def detect_contagion(
        self,
        returns_dict: Dict[str, pd.Series],
        window: int = 21,
        corr_spike_threshold: float = 0.8,
    ) -> List[TailRiskAlert]:
        """
        Detect correlation spikes across assets — a sign of contagion/panic.
        When correlations go to 1, diversification breaks down.
        """
        alerts = []
        if len(returns_dict) < 2:
            return alerts

        # Build returns matrix
        df = pd.DataFrame(returns_dict)
        df = df.dropna()

        if len(df) < window * 2:
            return alerts

        recent_corr = df.iloc[-window:].corr()
        historical_corr = df.iloc[:-window].corr()

        # Check for correlation spike
        recent_upper = recent_corr.values[np.triu_indices_from(recent_corr.values, k=1)]
        hist_upper = historical_corr.values[np.triu_indices_from(historical_corr.values, k=1)]

        mean_recent = np.nanmean(recent_upper)
        mean_hist = np.nanmean(hist_upper)

        if mean_recent > corr_spike_threshold and mean_recent > mean_hist + 0.2:
            alert = TailRiskAlert(
                alert_type="contagion",
                severity="danger",
                metric="mean_correlation",
                value=mean_recent,
                threshold=corr_spike_threshold,
                message=f"Cross-asset contagion detected: mean correlation "
                        f"spiked to {mean_recent:.2f} (historical: {mean_hist:.2f}). "
                        f"Diversification may be breaking down.",
                timestamp=pd.Timestamp.now(),
            )
            alerts.append(alert)

        self.alerts.extend(alerts)
        return alerts

    # ── Full Scan ────────────────────────────────────────────────────────

    def full_tail_risk_scan(
        self,
        returns: pd.Series,
        equity: Optional[pd.Series] = None,
        returns_dict: Optional[Dict[str, pd.Series]] = None,
    ) -> List[TailRiskAlert]:
        """Run all tail risk checks and return combined alerts."""
        all_alerts = []

        # VaR / CVaR
        var_99 = self.compute_var(returns, TAIL_RISK_VAR_CONFIDENCE)
        cvar_975 = self.compute_cvar(returns, TAIL_RISK_CVAR_CONFIDENCE)
        latest_return = returns.iloc[-1] if len(returns) > 0 else 0

        if latest_return < var_99:
            all_alerts.append(TailRiskAlert(
                alert_type="var_breach",
                severity="danger",
                metric="daily_return_vs_var99",
                value=latest_return,
                threshold=var_99,
                message=f"VaR breach: today's return {latest_return:.2%} "
                        f"< 99% VaR {var_99:.2%}",
                timestamp=pd.Timestamp.now(),
            ))

        # Vol spike
        all_alerts.extend(self.detect_vol_spike(returns))

        # Drawdown
        if equity is not None:
            all_alerts.extend(self.monitor_drawdown(equity))

        # Tail indicators
        all_alerts.extend(self.check_tail_risk_indicators(returns))

        # Contagion
        if returns_dict:
            all_alerts.extend(self.detect_contagion(returns_dict))

        # Sort by severity
        severity_order = {"extreme": 0, "danger": 1, "warning": 2, "watch": 3}
        all_alerts.sort(key=lambda a: severity_order.get(a.severity, 4))

        # Summary stats
        logger.info(
            "Tail Risk Scan: VaR(99%%)=%.2f%%, CVaR(97.5%%)=%.2f%%, "
            "%d alerts generated",
            var_99 * 100, cvar_975 * 100, len(all_alerts),
        )
        return all_alerts

    def get_risk_dashboard_data(self, returns: pd.Series) -> dict:
        """Generate data for risk dashboard display."""
        returns = returns.dropna()
        if len(returns) < 30:
            return {}
        return {
            "var_99_hist": self.compute_var(returns, 0.99, "historical"),
            "var_99_cf": self.compute_var(returns, 0.99, "cornish_fisher"),
            "cvar_975": self.compute_cvar(returns, 0.975),
            "current_vol": float(returns.iloc[-21:].std() * np.sqrt(252)),
            "long_vol": float(returns.std() * np.sqrt(252)),
            "kurtosis": float(sp_stats.kurtosis(returns)),
            "skewness": float(sp_stats.skew(returns)),
            "worst_day": float(returns.min()),
            "best_day": float(returns.max()),
            "pct_negative_days": float((returns < 0).mean()),
        }
