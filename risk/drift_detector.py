"""
Drift Detection Constraint System.
Monitors portfolio and strategy for distributional drift, regime changes,
and parameter instability to protect against silent strategy degradation.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from config.settings import MAX_PORTFOLIO_DRIFT_PCT

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """A single drift alert."""
    alert_type: str        # "allocation_drift", "return_drift", "regime_change", "param_instability"
    severity: str          # "low", "medium", "high", "critical"
    metric: str
    current_value: float
    threshold: float
    message: str
    timestamp: Optional[pd.Timestamp] = None


class DriftDetector:
    """
    Detect strategy and portfolio drift using statistical tests.

    Methods:
    - Allocation drift: current vs target weights
    - Return distribution drift: KS test on rolling windows
    - Regime change detection: HMM-free approach using rolling stats
    - Parameter instability: CUSUM on rolling Sharpe
    """

    def __init__(self, max_drift_pct: float = MAX_PORTFOLIO_DRIFT_PCT):
        self.max_drift_pct = max_drift_pct
        self.alerts: List[DriftAlert] = []

    def check_allocation_drift(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> List[DriftAlert]:
        """Check if portfolio allocation has drifted beyond threshold."""
        alerts = []
        for asset, target in target_weights.items():
            current = current_weights.get(asset, 0)
            drift = abs(current - target) * 100

            if drift > self.max_drift_pct:
                severity = "critical" if drift > self.max_drift_pct * 2 else "high"
                alert = DriftAlert(
                    alert_type="allocation_drift",
                    severity=severity,
                    metric=f"{asset}_weight",
                    current_value=current,
                    threshold=target,
                    message=f"{asset}: weight drifted {drift:.1f}% from target "
                            f"(current={current:.1%}, target={target:.1%})",
                    timestamp=pd.Timestamp.now(),
                )
                alerts.append(alert)
                logger.warning("DRIFT ALERT: %s", alert.message)
        self.alerts.extend(alerts)
        return alerts

    def check_return_distribution_drift(
        self,
        returns: pd.Series,
        reference_window: int = 63,
        test_window: int = 21,
        significance: float = 0.05,
    ) -> List[DriftAlert]:
        """
        Use Kolmogorov-Smirnov test to detect if recent returns come from
        a different distribution than the reference period.
        """
        alerts = []
        if len(returns) < reference_window + test_window:
            return alerts

        ref = returns.iloc[-(reference_window + test_window):-test_window].dropna()
        test = returns.iloc[-test_window:].dropna()

        if len(ref) < 10 or len(test) < 5:
            return alerts

        ks_stat, p_value = stats.ks_2samp(ref, test)

        if p_value < significance:
            severity = "critical" if p_value < 0.01 else "high" if p_value < 0.025 else "medium"
            alert = DriftAlert(
                alert_type="return_drift",
                severity=severity,
                metric="ks_statistic",
                current_value=ks_stat,
                threshold=significance,
                message=f"Return distribution shift detected (KS={ks_stat:.3f}, "
                        f"p={p_value:.4f}). Recent returns differ significantly "
                        f"from reference period.",
                timestamp=pd.Timestamp.now(),
            )
            alerts.append(alert)
            logger.warning("DRIFT ALERT: %s", alert.message)

        self.alerts.extend(alerts)
        return alerts

    def detect_regime_change(
        self,
        returns: pd.Series,
        vol_window: int = 21,
        vol_threshold: float = 2.0,
        corr_data: Optional[pd.DataFrame] = None,
    ) -> List[DriftAlert]:
        """
        Detect regime changes using volatility regime shifts and
        correlation breakdowns.
        """
        alerts = []
        if len(returns) < vol_window * 3:
            return alerts

        # Volatility regime detection
        rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
        long_vol = returns.rolling(vol_window * 3).std() * np.sqrt(252)

        vol_ratio = rolling_vol / long_vol
        latest_ratio = vol_ratio.iloc[-1]

        if latest_ratio > vol_threshold:
            alert = DriftAlert(
                alert_type="regime_change",
                severity="high" if latest_ratio > vol_threshold * 1.5 else "medium",
                metric="vol_regime_ratio",
                current_value=latest_ratio,
                threshold=vol_threshold,
                message=f"Volatility regime shift: short-term vol is {latest_ratio:.1f}x "
                        f"long-term average (threshold={vol_threshold:.1f}x)",
                timestamp=pd.Timestamp.now(),
            )
            alerts.append(alert)

        # Correlation breakdown detection
        if corr_data is not None and len(corr_data) > vol_window * 2:
            recent_corr = corr_data.iloc[-vol_window:].corr()
            hist_corr = corr_data.iloc[:-vol_window].corr()

            corr_diff = (recent_corr - hist_corr).abs()
            max_corr_shift = corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)].max()

            if max_corr_shift > 0.4:
                alert = DriftAlert(
                    alert_type="regime_change",
                    severity="high",
                    metric="correlation_breakdown",
                    current_value=max_corr_shift,
                    threshold=0.4,
                    message=f"Correlation structure breakdown: max correlation shift "
                            f"= {max_corr_shift:.2f}",
                    timestamp=pd.Timestamp.now(),
                )
                alerts.append(alert)

        self.alerts.extend(alerts)
        return alerts

    def check_parameter_stability(
        self,
        returns: pd.Series,
        window: int = 42,
        cusum_threshold: float = 3.0,
    ) -> List[DriftAlert]:
        """
        CUSUM test on rolling Sharpe ratio to detect strategy degradation.
        """
        alerts = []
        if len(returns) < window * 2:
            return alerts

        rolling_sharpe = (
            returns.rolling(window).mean() / returns.rolling(window).std()
        ) * np.sqrt(252)

        rolling_sharpe = rolling_sharpe.dropna()
        if len(rolling_sharpe) < 10:
            return alerts

        mean_sharpe = rolling_sharpe.mean()
        std_sharpe = rolling_sharpe.std()

        if std_sharpe == 0:
            return alerts

        cusum = ((rolling_sharpe - mean_sharpe) / std_sharpe).cumsum()
        max_cusum = cusum.abs().max()

        if max_cusum > cusum_threshold:
            alert = DriftAlert(
                alert_type="param_instability",
                severity="high" if max_cusum > cusum_threshold * 1.5 else "medium",
                metric="sharpe_cusum",
                current_value=max_cusum,
                threshold=cusum_threshold,
                message=f"Strategy parameter instability detected: CUSUM={max_cusum:.2f} "
                        f"(threshold={cusum_threshold:.1f}). Rolling Sharpe is unstable.",
                timestamp=pd.Timestamp.now(),
            )
            alerts.append(alert)

        self.alerts.extend(alerts)
        return alerts

    def full_drift_scan(
        self,
        returns: pd.Series,
        current_weights: Optional[Dict[str, float]] = None,
        target_weights: Optional[Dict[str, float]] = None,
        corr_data: Optional[pd.DataFrame] = None,
    ) -> List[DriftAlert]:
        """Run all drift checks and return combined alerts."""
        all_alerts = []

        if current_weights and target_weights:
            all_alerts.extend(
                self.check_allocation_drift(current_weights, target_weights)
            )

        all_alerts.extend(self.check_return_distribution_drift(returns))
        all_alerts.extend(self.detect_regime_change(returns, corr_data=corr_data))
        all_alerts.extend(self.check_parameter_stability(returns))

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_alerts.sort(key=lambda a: severity_order.get(a.severity, 4))
        return all_alerts

    def get_drift_summary(self) -> pd.DataFrame:
        """Return all historical alerts as a DataFrame."""
        if not self.alerts:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "timestamp": a.timestamp,
                "type": a.alert_type,
                "severity": a.severity,
                "metric": a.metric,
                "value": a.current_value,
                "threshold": a.threshold,
                "message": a.message,
            }
            for a in self.alerts
        ])
