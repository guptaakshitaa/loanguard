"""PSI-based feature drift detection for LoanGuard.

Computes Population Stability Index (PSI) for each feature
by comparing the training distribution against recent predictions.

PSI thresholds:
    < 0.10  → Stable      (no action needed)
    < 0.20  → Monitor     (watch closely)
    >= 0.20 → Alert       (investigate / consider retraining)

Usage:
    from monitoring.drift_detector import DriftDetector
    detector = DriftDetector(reference_df)
    report = detector.compute(current_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PSI_MONITOR = 0.10
PSI_ALERT   = 0.20


def _psi_numeric(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Compute PSI for a numeric feature."""
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    exp_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    act_pct = np.histogram(actual,   bins=bins)[0] / len(actual)

    exp_pct = np.where(exp_pct == 0, epsilon, exp_pct)
    act_pct = np.where(act_pct == 0, epsilon, act_pct)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _psi_categorical(
    expected: np.ndarray,
    actual: np.ndarray,
    epsilon: float = 1e-4,
) -> float:
    """Compute PSI for a categorical feature."""
    cats = np.union1d(
        np.unique(expected[pd.notna(expected)]),
        np.unique(actual[pd.notna(actual)]),
    )
    psi = 0.0
    for cat in cats:
        e = np.mean(expected == cat) or epsilon
        a = np.mean(actual == cat) or epsilon
        psi += (a - e) * np.log(a / e)
    return float(psi)


@dataclass
class FeatureDriftResult:
    feature: str
    psi: float
    status: str   # stable / monitor / alert
    n_reference: int
    n_current: int


@dataclass
class DriftReport:
    overall_status: str
    n_features_checked: int
    n_alerts: int
    features: list[FeatureDriftResult] = field(default_factory=list)
    reference_period: str = ""
    current_period: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "n_features_checked": self.n_features_checked,
            "n_alerts": self.n_alerts,
            "features": [
                {
                    "feature": f.feature,
                    "psi": round(f.psi, 4),
                    "status": f.status,
                }
                for f in self.features
            ],
            "reference_period": self.reference_period,
            "current_period": self.current_period,
            "timestamp": self.timestamp.isoformat(),
        }


class DriftDetector:
    """Computes PSI drift for all features against a reference dataset.

    Args:
        reference_df: Training / reference DataFrame.
        numeric_cols: List of numeric feature names.
        categorical_cols: List of categorical feature names.
    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        numeric_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
    ) -> None:
        self._ref = reference_df.copy()

        if numeric_cols is not None:
            self._num_cols = numeric_cols
        else:
            self._num_cols = reference_df.select_dtypes(include=np.number).columns.tolist()

        if categorical_cols is not None:
            self._cat_cols = categorical_cols
        else:
            self._cat_cols = reference_df.select_dtypes(exclude=np.number).columns.tolist()

        # Remove ID columns
        for col in ["application_id"]:
            self._num_cols = [c for c in self._num_cols if c != col]
            self._cat_cols = [c for c in self._cat_cols if c != col]

        logger.info(
            "DriftDetector initialised: %d numeric + %d categorical features",
            len(self._num_cols), len(self._cat_cols),
        )

    def compute(
        self,
        current_df: pd.DataFrame,
        reference_period: str = "training",
        current_period: str = "recent",
    ) -> DriftReport:
        """Compute PSI for every feature and produce a drift report.

        Args:
            current_df: Recent production data (from prediction log).
            reference_period: Label for the reference window.
            current_period: Label for the current window.

        Returns:
            DriftReport dataclass with per-feature PSI scores.
        """
        if len(current_df) < 30:
            logger.warning(
                "Only %d samples in current window — PSI may be unreliable", len(current_df)
            )

        results: list[FeatureDriftResult] = []

        for col in self._num_cols:
            if col not in current_df.columns:
                continue
            psi = _psi_numeric(
                self._ref[col].values.astype(float),
                current_df[col].values.astype(float),
            )
            status = (
                "alert"   if psi >= PSI_ALERT   else
                "monitor" if psi >= PSI_MONITOR  else
                "stable"
            )
            results.append(FeatureDriftResult(
                feature=col, psi=psi, status=status,
                n_reference=len(self._ref), n_current=len(current_df),
            ))

        for col in self._cat_cols:
            if col not in current_df.columns:
                continue
            psi = _psi_categorical(
                self._ref[col].values.astype(str),
                current_df[col].values.astype(str),
            )
            status = (
                "alert"   if psi >= PSI_ALERT   else
                "monitor" if psi >= PSI_MONITOR  else
                "stable"
            )
            results.append(FeatureDriftResult(
                feature=col, psi=psi, status=status,
                n_reference=len(self._ref), n_current=len(current_df),
            ))

        results.sort(key=lambda r: r.psi, reverse=True)
        n_alerts = sum(1 for r in results if r.status == "alert")
        n_monitor = sum(1 for r in results if r.status == "monitor")

        overall = (
            "alert"   if n_alerts > 0  else
            "monitor" if n_monitor > 0 else
            "stable"
        )

        logger.info(
            "Drift check: %d features | %d alerts | %d monitor | overall=%s",
            len(results), n_alerts, n_monitor, overall,
        )

        return DriftReport(
            overall_status=overall,
            n_features_checked=len(results),
            n_alerts=n_alerts,
            features=results,
            reference_period=reference_period,
            current_period=current_period,
        )
