"""
Institutional Drift Detection Constraint System.

Features:
- Allocation drift with weight validation
- Return distribution drift (KS test with effective sample size)
- Regime change (Mahalanobis distance, vol regime, correlation breakdown)
- Parameter stability (CUSUM, structural break detection)
- Autocorrelation-adjusted statistical tests
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from config.settings import MAX_PORTFOLIO_DRIFT_PCT, MIN_OBSERVATIONS_FOR_STATS

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """A single drift alert."""
    alert_type: str
    severity: str        # "low", "medium", "high", "critical"
    metric: str
    current_value: float
    threshold: float
    message: str
    timestamp: Optional[pd.Timestamp] = None


class DriftDetector:
    """
    Institutional drift detection with autocorrelation-corrected
    statistical tests and Mahalanobis-based regime detection.
    """

    def __init__(self, max_drift_pct: float = MAX_PORTFOLIO_DRIFT_PCT):
        self.max_drift_pct = max_drift_pct
        self.alerts: List[DriftAlert] = []

    # ── Allocation Drift ─────────────────────────────────────────────────

    def check_allocation_drift(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> List[DriftAlert]:
        """Check if portfolio weights have drifted beyond threshold."""
        alerts = []

        # Validate weights
        cw_sum = sum(current_weights.values())
        tw_sum = sum(target_weights.values())
        if abs(cw_sum - 1.0) > 0.05:
            alerts.append(DriftAlert(
                alert_type="weight_validation",
                severity="high",
                metric="current_weight_sum",
                current_value=cw_sum,
                threshold=1.0,
                message=f"Current weights sum to {cw_sum:.2f}, expected ~1.0",
                timestamp=pd.Timestamp.now(),
            ))

        for asset, target in target_weights.items():
            current = current_weights.get(asset, 0)
            drift_pct = abs(current - target) * 100

            if drift_pct > self.max_drift_pct:
                severity = "critical" if drift_pct > self.max_drift_pct * 2 else "high"
                alert = DriftAlert(
                    alert_type="allocation_drift",
                    severity=severity,
                    metric=f"{asset}_weight",
                    current_value=current,
                    threshold=target,
                    message=f"{asset}: drifted {drift_pct:.1f}% "
                            f"(current={current:.1%}, target={target:.1%})",
                    timestamp=pd.Timestamp.now(),
                )
                alerts.append(alert)

        self.alerts.extend(alerts)
        return alerts

    # ── Return Distribution Drift (autocorrelation-corrected) ────────────

    def _effective_sample_size(self, series: pd.Series) -> float:
        """
        Adjust sample size for autocorrelation.
        n_eff = n / (1 + 2 * sum(autocorrelations))
        """
        n = len(series)
        if n < 10:
            return float(n)

        # Compute autocorrelations up to lag 10
        max_lag = min(10, n // 4)
        autocorrs = []
        for lag in range(1, max_lag + 1):
            ac = series.autocorr(lag=lag)
            if np.isnan(ac) or abs(ac) < 1.96 / np.sqrt(n):
                break  # insignificant
            autocorrs.append(ac)

        if not autocorrs:
            return float(n)

        correction = 1 + 2 * sum(autocorrs)
        return max(float(n / correction), 3)

    def check_return_distribution_drift(
        self,
        returns: pd.Series,
        reference_window: int = 63,
        test_window: int = 21,
        significance: float = 0.05,
    ) -> List[DriftAlert]:
        """KS test with autocorrelation-adjusted effective sample size."""
        alerts = []
        if len(returns) < reference_window + test_window:
            return alerts

        ref = returns.iloc[-(reference_window + test_window):-test_window].dropna()
        test = returns.iloc[-test_window:].dropna()

        if len(ref) < 10 or len(test) < 5:
            return alerts

        ks_stat, p_value = stats.ks_2samp(ref, test)

        # Adjust p-value for autocorrelation
        n_eff_ref = self._effective_sample_size(ref)
        n_eff_test = self._effective_sample_size(test)

        # Adjusted critical value (larger effective sample = more power)
        adj_factor = min(n_eff_ref / len(ref), n_eff_test / len(test))
        adjusted_significance = significance / adj_factor if adj_factor > 0 else significance

        if p_value < adjusted_significance:
            severity = "critical" if p_value < 0.01 else "high" if p_value < 0.025 else "medium"
            alert = DriftAlert(
                alert_type="return_drift",
                severity=severity,
                metric="ks_statistic",
                current_value=ks_stat,
                threshold=adjusted_significance,
                message=f"Return distribution shift: KS={ks_stat:.3f}, "
                        f"p={p_value:.4f} (adj. sig={adjusted_significance:.4f}, "
                        f"n_eff_ref={n_eff_ref:.0f}, n_eff_test={n_eff_test:.0f})",
                timestamp=pd.Timestamp.now(),
            )
            alerts.append(alert)

        self.alerts.extend(alerts)
        return alerts

    # ── Regime Change (Mahalanobis) ──────────────────────────────────────

    def detect_regime_change(
        self,
        returns: pd.Series,
        vol_window: int = 21,
        vol_threshold: float = 2.0,
        corr_data: Optional[pd.DataFrame] = None,
    ) -> List[DriftAlert]:
        """
        Regime detection using:
        1. Volatility regime ratio
        2. Mahalanobis distance (if multi-asset data available)
        3. Correlation structure breakdown
        """
        alerts = []
        if len(returns) < vol_window * 3:
            return alerts

        # ── Volatility regime ────────────────────────────────────────
        short_vol = returns.rolling(vol_window).std()
        long_vol = returns.rolling(vol_window * 3).std()

        # Use ratio (not z-score) for robustness
        vol_ratio = (short_vol / long_vol.replace(0, np.nan)).dropna()
        if len(vol_ratio) == 0:
            return alerts

        latest_ratio = float(vol_ratio.iloc[-1])
        if latest_ratio > vol_threshold:
            severity = "high" if latest_ratio > vol_threshold * 1.5 else "medium"
            alerts.append(DriftAlert(
                alert_type="regime_change",
                severity=severity,
                metric="vol_regime_ratio",
                current_value=latest_ratio,
                threshold=vol_threshold,
                message=f"Vol regime shift: short/long vol ratio={latest_ratio:.2f}x "
                        f"(threshold={vol_threshold:.1f}x)",
                timestamp=pd.Timestamp.now(),
            ))

        # ── Mahalanobis distance (multi-asset) ──────────────────────
        if corr_data is not None and len(corr_data) > vol_window * 2:
            recent = corr_data.iloc[-vol_window:].dropna()
            historical = corr_data.iloc[:-vol_window].dropna()

            if len(recent) >= 5 and len(historical) >= 20:
                try:
                    hist_mean = historical.mean().values
                    hist_cov = historical.cov().values
                    recent_mean = recent.mean().values

                    # Regularize covariance for inversion
                    cov_reg = hist_cov + np.eye(len(hist_cov)) * 1e-6
                    cov_inv = np.linalg.inv(cov_reg)

                    diff = recent_mean - hist_mean
                    mahal_dist = float(np.sqrt(diff @ cov_inv @ diff))

                    # Chi-squared threshold for p=0.01
                    chi2_thresh = stats.chi2.ppf(0.99, df=len(hist_mean))

                    if mahal_dist > np.sqrt(chi2_thresh):
                        alerts.append(DriftAlert(
                            alert_type="regime_change",
                            severity="high",
                            metric="mahalanobis_distance",
                            current_value=mahal_dist,
                            threshold=np.sqrt(chi2_thresh),
                            message=f"Mahalanobis regime shift: d={mahal_dist:.2f} "
                                    f"(threshold={np.sqrt(chi2_thresh):.2f})",
                            timestamp=pd.Timestamp.now(),
                        ))
                except (np.linalg.LinAlgError, ValueError) as e:
                    logger.debug("Mahalanobis calculation failed: %s", e)

            # ── Correlation breakdown ────────────────────────────────
            if len(recent) >= 5 and len(historical) >= 20:
                recent_corr = recent.corr()
                hist_corr = historical.corr()
                corr_diff = (recent_corr - hist_corr).abs()

                upper_idx = np.triu_indices_from(corr_diff.values, k=1)
                if len(upper_idx[0]) > 0:
                    max_shift = float(corr_diff.values[upper_idx].max())
                    mean_shift = float(corr_diff.values[upper_idx].mean())

                    if max_shift > 0.4:
                        alerts.append(DriftAlert(
                            alert_type="regime_change",
                            severity="high" if max_shift > 0.6 else "medium",
                            metric="correlation_breakdown",
                            current_value=max_shift,
                            threshold=0.4,
                            message=f"Correlation breakdown: max shift={max_shift:.2f}, "
                                    f"mean shift={mean_shift:.2f}",
                            timestamp=pd.Timestamp.now(),
                        ))

        self.alerts.extend(alerts)
        return alerts

    # ── Parameter Stability (CUSUM + Structural Break) ───────────────────

    def check_parameter_stability(
        self,
        returns: pd.Series,
        window: int = 42,
        cusum_threshold: float = 3.0,
    ) -> List[DriftAlert]:
        """CUSUM test on rolling Sharpe + Chow-like structural break."""
        alerts = []
        if len(returns) < window * 2:
            return alerts

        # ── CUSUM on rolling Sharpe ──────────────────────────────────
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(252)).dropna()

        if len(rolling_sharpe) < 10:
            return alerts

        mean_sharpe = rolling_sharpe.mean()
        std_sharpe = rolling_sharpe.std()

        if std_sharpe > 0:
            cusum = ((rolling_sharpe - mean_sharpe) / std_sharpe).cumsum()
            max_cusum = float(cusum.abs().max())

            if max_cusum > cusum_threshold:
                alerts.append(DriftAlert(
                    alert_type="param_instability",
                    severity="high" if max_cusum > cusum_threshold * 1.5 else "medium",
                    metric="sharpe_cusum",
                    current_value=max_cusum,
                    threshold=cusum_threshold,
                    message=f"Parameter instability: CUSUM={max_cusum:.2f} "
                            f"(threshold={cusum_threshold:.1f}). "
                            f"Rolling Sharpe drifted from {rolling_sharpe.iloc[:window].mean():.2f} "
                            f"to {rolling_sharpe.iloc[-window:].mean():.2f}",
                    timestamp=pd.Timestamp.now(),
                ))

        # ── Structural break (Chow-like) ────────────────────────────
        mid = len(returns) // 2
        first_half = returns.iloc[:mid].dropna()
        second_half = returns.iloc[mid:].dropna()

        if len(first_half) >= 20 and len(second_half) >= 20:
            t_stat, t_pval = stats.ttest_ind(first_half, second_half)
            f_stat, f_pval = stats.levene(first_half, second_half)

            if t_pval < 0.01:
                alerts.append(DriftAlert(
                    alert_type="structural_break",
                    severity="high",
                    metric="mean_break_t_stat",
                    current_value=abs(t_stat),
                    threshold=2.576,  # z for 0.01
                    message=f"Structural break in mean returns: "
                            f"t={t_stat:.2f}, p={t_pval:.4f} "
                            f"(first_half={first_half.mean():.4f}, "
                            f"second_half={second_half.mean():.4f})",
                    timestamp=pd.Timestamp.now(),
                ))

            if f_pval < 0.01:
                alerts.append(DriftAlert(
                    alert_type="structural_break",
                    severity="medium",
                    metric="variance_break_f_stat",
                    current_value=abs(f_stat),
                    threshold=0.01,
                    message=f"Structural break in variance: "
                            f"F={f_stat:.2f}, p={f_pval:.4f}",
                    timestamp=pd.Timestamp.now(),
                ))

        self.alerts.extend(alerts)
        return alerts

    # ── Full Scan ────────────────────────────────────────────────────────

    def full_drift_scan(
        self,
        returns: pd.Series,
        current_weights: Optional[Dict[str, float]] = None,
        target_weights: Optional[Dict[str, float]] = None,
        corr_data: Optional[pd.DataFrame] = None,
    ) -> List[DriftAlert]:
        all_alerts = []
        if current_weights and target_weights:
            all_alerts.extend(self.check_allocation_drift(current_weights, target_weights))
        all_alerts.extend(self.check_return_distribution_drift(returns))
        all_alerts.extend(self.detect_regime_change(returns, corr_data=corr_data))
        all_alerts.extend(self.check_parameter_stability(returns))

        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_alerts.sort(key=lambda a: severity_order.get(a.severity, 4))
        return all_alerts

    def get_drift_summary(self) -> pd.DataFrame:
        if not self.alerts:
            return pd.DataFrame()
        return pd.DataFrame([{
            "timestamp": a.timestamp, "type": a.alert_type,
            "severity": a.severity, "metric": a.metric,
            "value": a.current_value, "threshold": a.threshold,
            "message": a.message,
        } for a in self.alerts])
