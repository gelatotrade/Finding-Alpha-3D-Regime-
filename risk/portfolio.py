"""
Institutional Portfolio Construction.

Features:
- Kelly Criterion (full and fractional)
- Risk Parity
- Mean-Variance with robust covariance (Ledoit-Wolf shrinkage)
- Black-Litterman
- Maximum Diversification
"""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class PortfolioConstructor:
    """Multi-method portfolio optimizer."""

    # ── Kelly Criterion ──────────────────────────────────────────────────

    def kelly_weights(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        fraction: float = 0.5,
    ) -> np.ndarray:
        """
        Full Kelly: w* = Σ^(-1) * μ
        Fractional Kelly (default 0.5): w* = f * Σ^(-1) * μ

        Using half-Kelly is standard institutional practice for robustness.
        """
        n = len(expected_returns)
        try:
            # Ledoit-Wolf shrinkage for robustness
            cov_shrunk = self._ledoit_wolf_shrinkage(cov_matrix)
            cov_inv = np.linalg.inv(cov_shrunk)
            weights = fraction * cov_inv @ expected_returns
            # Normalize to sum to 1 (long-only)
            weights = np.clip(weights, 0, None)
            total = weights.sum()
            if total > 0:
                weights = weights / total
            else:
                weights = np.ones(n) / n
            return weights
        except np.linalg.LinAlgError:
            logger.warning("Kelly: singular covariance, returning equal weight")
            return np.ones(n) / n

    # ── Risk Parity ──────────────────────────────────────────────────────

    def risk_parity_weights(
        self,
        cov_matrix: np.ndarray,
        risk_budget: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Risk parity: each asset contributes equally to total portfolio risk.
        Solves: min Σ (RC_i - target_RC)^2
        where RC_i = w_i * (Σw)_i / σ_p
        """
        n = cov_matrix.shape[0]
        if risk_budget is None:
            risk_budget = np.ones(n) / n

        def objective(w):
            w = np.abs(w)
            port_vol = np.sqrt(w @ cov_matrix @ w)
            if port_vol == 0:
                return 1e10
            marginal_risk = cov_matrix @ w
            risk_contrib = w * marginal_risk / port_vol
            target_risk = risk_budget * port_vol
            return np.sum((risk_contrib - target_risk) ** 2)

        x0 = np.ones(n) / n
        bounds = [(0.01, 0.5) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        result = minimize(objective, x0, bounds=bounds, constraints=constraints,
                         method="SLSQP", options={"maxiter": 500})

        if result.success:
            return np.abs(result.x) / np.sum(np.abs(result.x))
        logger.warning("Risk parity optimization failed, returning equal weight")
        return np.ones(n) / n

    # ── Mean-Variance (Markowitz with shrinkage) ─────────────────────────

    def mean_variance_weights(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        target_return: Optional[float] = None,
        risk_aversion: float = 2.0,
        max_weight: float = 0.30,
    ) -> np.ndarray:
        """
        Mean-variance with Ledoit-Wolf shrinkage and position limits.
        """
        n = len(expected_returns)
        cov_shrunk = self._ledoit_wolf_shrinkage(cov_matrix)

        if target_return is not None:
            # Minimize risk subject to target return
            def objective(w):
                return w @ cov_shrunk @ w

            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w: w @ expected_returns - target_return},
            ]
        else:
            # Maximize utility: μ'w - (λ/2) w'Σw
            def objective(w):
                return -(w @ expected_returns - risk_aversion / 2 * w @ cov_shrunk @ w)

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        bounds = [(0, max_weight) for _ in range(n)]
        x0 = np.ones(n) / n

        result = minimize(objective, x0, bounds=bounds, constraints=constraints,
                         method="SLSQP", options={"maxiter": 500})

        if result.success:
            return result.x
        return np.ones(n) / n

    # ── Black-Litterman ──────────────────────────────────────────────────

    def black_litterman_weights(
        self,
        market_weights: np.ndarray,
        cov_matrix: np.ndarray,
        views: np.ndarray,          # P matrix (K x N): view selection
        view_returns: np.ndarray,    # Q vector (K,): expected returns from views
        view_confidence: np.ndarray, # Omega diagonal (K,): uncertainty
        risk_aversion: float = 2.5,
        tau: float = 0.05,
    ) -> np.ndarray:
        """
        Black-Litterman model combining market equilibrium with investor views.

        Parameters:
        - market_weights: capitalization-weighted market portfolio
        - cov_matrix: asset covariance matrix
        - views: P matrix (each row is a view linking assets)
        - view_returns: Q vector (expected returns per view)
        - view_confidence: diagonal of Omega (lower = more confident)
        """
        n = len(market_weights)
        cov_shrunk = self._ledoit_wolf_shrinkage(cov_matrix)

        # Equilibrium returns: π = λΣw_mkt
        pi = risk_aversion * cov_shrunk @ market_weights

        # Uncertainty in prior
        tau_sigma = tau * cov_shrunk

        # Omega: uncertainty in views
        omega = np.diag(view_confidence)

        try:
            # BL posterior: μ_BL = [(τΣ)^-1 + P'Ω^-1 P]^-1 [(τΣ)^-1 π + P'Ω^-1 Q]
            tau_sigma_inv = np.linalg.inv(tau_sigma)
            omega_inv = np.linalg.inv(omega)

            posterior_cov_inv = tau_sigma_inv + views.T @ omega_inv @ views
            posterior_cov = np.linalg.inv(posterior_cov_inv)

            posterior_returns = posterior_cov @ (
                tau_sigma_inv @ pi + views.T @ omega_inv @ view_returns
            )

            # Optimize with posterior
            return self.mean_variance_weights(posterior_returns, cov_shrunk)

        except np.linalg.LinAlgError:
            logger.warning("Black-Litterman: singular matrix, returning market weights")
            return market_weights

    # ── Maximum Diversification ──────────────────────────────────────────

    def max_diversification_weights(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Maximize diversification ratio: DR = w'σ / sqrt(w'Σw)
        """
        n = cov_matrix.shape[0]
        vols = np.sqrt(np.diag(cov_matrix))

        def neg_div_ratio(w):
            port_vol = np.sqrt(w @ cov_matrix @ w)
            if port_vol == 0:
                return 1e10
            return -(w @ vols) / port_vol

        bounds = [(0.01, 0.5) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        x0 = np.ones(n) / n

        result = minimize(neg_div_ratio, x0, bounds=bounds,
                         constraints=constraints, method="SLSQP")
        if result.success:
            return result.x
        return np.ones(n) / n

    # ── Ledoit-Wolf Shrinkage ────────────────────────────────────────────

    def _ledoit_wolf_shrinkage(self, S: np.ndarray) -> np.ndarray:
        """
        Ledoit & Wolf (2004) shrinkage estimator for covariance matrix.
        Shrinks sample covariance toward scaled identity matrix.
        """
        n = S.shape[0]
        if n <= 1:
            return S

        trace_S = np.trace(S)
        mu = trace_S / n
        delta = S - mu * np.eye(n)
        delta_sq_sum = np.sum(delta ** 2)

        # Simplified shrinkage intensity
        # In practice, use sklearn's LedoitWolf, but this avoids the dependency
        shrinkage_intensity = min(1.0, max(0.0, delta_sq_sum / (trace_S ** 2 + delta_sq_sum)))

        return (1 - shrinkage_intensity) * S + shrinkage_intensity * mu * np.eye(n)

    # ── Convenience ──────────────────────────────────────────────────────

    def optimize(
        self,
        returns_df: pd.DataFrame,
        method: str = "risk_parity",
        **kwargs,
    ) -> Dict[str, float]:
        """
        High-level API: compute optimal weights from a returns DataFrame.
        """
        if returns_df.empty or len(returns_df) < 30:
            # Equal weight fallback
            n = len(returns_df.columns)
            return {col: 1.0 / n for col in returns_df.columns}

        mu = returns_df.mean().values * 252  # annualize
        cov = returns_df.cov().values * 252

        if method == "risk_parity":
            weights = self.risk_parity_weights(cov)
        elif method == "kelly":
            weights = self.kelly_weights(mu, cov, fraction=kwargs.get("fraction", 0.5))
        elif method == "mean_variance":
            weights = self.mean_variance_weights(mu, cov, **kwargs)
        elif method == "max_diversification":
            weights = self.max_diversification_weights(cov)
        else:
            n = len(mu)
            weights = np.ones(n) / n

        return {col: float(w) for col, w in zip(returns_df.columns, weights)}
