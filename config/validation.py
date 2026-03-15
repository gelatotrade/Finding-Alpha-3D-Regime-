"""
Data validation utilities for institutional-grade data quality.
"""
import logging
from typing import List, Optional, Set
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataQualityReport:
    """Report from a data quality check."""

    def __init__(self, source: str):
        self.source = source
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.rows_before = 0
        self.rows_after = 0
        self.nulls_filled = 0
        self.outliers_removed = 0

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return (f"DataQuality[{self.source}]: {status}, "
                f"{len(self.issues)} issues, {len(self.warnings)} warnings, "
                f"rows {self.rows_before}→{self.rows_after}")


def validate_ohlcv(
    df: pd.DataFrame,
    source: str = "unknown",
    min_rows: int = 20,
    max_gap_days: int = 5,
    max_return_pct: float = 50.0,
) -> tuple:
    """
    Validate and clean OHLCV data.

    Returns (cleaned_df, DataQualityReport).
    """
    report = DataQualityReport(source)
    report.rows_before = len(df)

    if df.empty:
        report.issues.append("Empty DataFrame")
        report.rows_after = 0
        return df, report

    # Required columns
    required = {"close"}
    missing = required - set(df.columns)
    if missing:
        report.issues.append(f"Missing columns: {missing}")
        report.rows_after = len(df)
        return df, report

    df = df.copy()

    # Sort by index
    if isinstance(df.index, pd.DatetimeIndex):
        df.sort_index(inplace=True)

    # Drop rows where close is NaN or zero
    invalid_close = df["close"].isna() | (df["close"] <= 0)
    if invalid_close.any():
        n = invalid_close.sum()
        report.warnings.append(f"Removed {n} rows with invalid close prices")
        df = df[~invalid_close]

    # OHLC consistency: high >= low, high >= close, low <= close
    for col in ["high", "low", "open"]:
        if col in df.columns:
            df[col] = df[col].fillna(df["close"])
            report.nulls_filled += df[col].isna().sum()

    if "high" in df.columns and "low" in df.columns:
        inconsistent = df["high"] < df["low"]
        if inconsistent.any():
            n = inconsistent.sum()
            report.warnings.append(f"Fixed {n} rows where high < low (swapped)")
            df.loc[inconsistent, ["high", "low"]] = df.loc[inconsistent, ["low", "high"]].values

    # Outlier detection: returns > max_return_pct
    returns = df["close"].pct_change()
    extreme = returns.abs() > (max_return_pct / 100)
    if extreme.any():
        n = extreme.sum()
        report.warnings.append(f"Flagged {n} extreme returns (>{max_return_pct}%)")
        report.outliers_removed = n
        # Don't remove — flag only. Institutional systems keep the data but flag it.
        df.loc[extreme, "_outlier"] = True

    # Gap detection
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        gaps = df.index.to_series().diff().dt.days
        large_gaps = gaps[gaps > max_gap_days]
        if not large_gaps.empty:
            report.warnings.append(
                f"{len(large_gaps)} gaps > {max_gap_days} days detected"
            )

    # Minimum data
    if len(df) < min_rows:
        report.issues.append(f"Only {len(df)} rows, need >= {min_rows}")

    report.rows_after = len(df)
    return df, report


def validate_prediction_market(df: pd.DataFrame, source: str = "unknown") -> tuple:
    """Validate prediction market data."""
    report = DataQualityReport(source)
    report.rows_before = len(df)

    if df.empty:
        report.issues.append("Empty DataFrame")
        report.rows_after = 0
        return df, report

    df = df.copy()

    # Price bounds: probabilities must be in [0, 1]
    for col in ["outcome_yes", "outcome_no", "yes_price", "no_price"]:
        if col in df.columns:
            invalid = (df[col] < 0) | (df[col] > 1)
            if invalid.any():
                n = invalid.sum()
                report.warnings.append(f"Clamped {n} values in {col} to [0,1]")
                df[col] = df[col].clip(0, 1)

    # Arbitrage check: yes + no should ≈ 1 (with spread)
    if "outcome_yes" in df.columns and "outcome_no" in df.columns:
        total = df["outcome_yes"].fillna(0) + df["outcome_no"].fillna(0)
        arb = (total < 0.9) | (total > 1.15)
        if arb.any():
            report.warnings.append(f"{arb.sum()} markets with potential arbitrage (yes+no outside [0.9, 1.15])")

    report.rows_after = len(df)
    return df, report


def compute_data_quality_score(df: pd.DataFrame) -> float:
    """Compute a 0-1 quality score for a time series DataFrame."""
    if df.empty:
        return 0.0

    scores = []

    # Completeness: % of non-null values
    completeness = 1 - df.isnull().sum().sum() / (len(df) * len(df.columns))
    scores.append(completeness)

    # Freshness: how recent is the latest data
    if isinstance(df.index, pd.DatetimeIndex):
        staleness_days = (pd.Timestamp.now() - df.index.max()).days
        freshness = max(0, 1 - staleness_days / 30)
        scores.append(freshness)

    # Consistency: check for duplicate indices
    dup_ratio = 1 - df.index.duplicated().sum() / max(len(df), 1)
    scores.append(dup_ratio)

    return float(np.mean(scores))
