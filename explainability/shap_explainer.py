"""SHAP-based explainability for LoanGuard predictions.

Wraps the calibrated sklearn pipeline to extract the inner XGBoost
model and run TreeExplainer for fast, accurate SHAP values.

Usage:
    from explainability.shap_explainer import SHAPExplainer
    explainer = SHAPExplainer(pipeline)
    result = explainer.explain(X_row)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_xgb_pipeline(calibrated_pipeline: Any) -> Any:
    """Unwrap CalibratedClassifierCV → inner sklearn Pipeline.

    Args:
        calibrated_pipeline: Fitted CalibratedClassifierCV.

    Returns:
        The base sklearn Pipeline (steps: derive→select→impute→woe→scale→clf).

    Raises:
        AttributeError: If structure doesn't match expected wrapping.
    """
    try:
        return calibrated_pipeline.calibrated_classifiers_[0].estimator
    except (AttributeError, IndexError) as exc:
        raise AttributeError(
            "Could not unwrap calibrated pipeline. "
            "Expected CalibratedClassifierCV wrapping a Pipeline."
        ) from exc


def _transform_through_pipeline(pipeline: Any, X: pd.DataFrame) -> np.ndarray:
    """Run X through all preprocessing steps (everything except the classifier).

    Args:
        pipeline: Base sklearn Pipeline.
        X: Raw feature DataFrame.

    Returns:
        Numpy array after all transformations.
    """
    X_t = X.copy()
    steps = pipeline.steps[:-1]  # all but last (classifier)
    for _, transformer in steps:
        X_t = transformer.transform(X_t)
    if isinstance(X_t, pd.DataFrame):
        return X_t.values
    return np.array(X_t)


class SHAPExplainer:
    """SHAP TreeExplainer wrapper for the LoanGuard pipeline.

    Extracts the XGBoost classifier from inside the calibrated pipeline
    and uses SHAP's TreeExplainer for fast, exact Shapley values.

    Args:
        calibrated_pipeline: Fitted CalibratedClassifierCV from train.py.
    """

    def __init__(self, calibrated_pipeline: Any) -> None:
        try:
            import shap
            self._shap = shap
        except ImportError as exc:
            raise ImportError("shap is required: pip install shap") from exc

        self._calibrated = calibrated_pipeline
        self._base_pipeline = _extract_xgb_pipeline(calibrated_pipeline)
        self._classifier = self._base_pipeline.steps[-1][1]  # XGBClassifier

        # Fit TreeExplainer on the classifier
        self._explainer = shap.TreeExplainer(self._classifier)
        self._feature_names: list[str] = self._get_feature_names()
        logger.info(
            "SHAPExplainer ready: %d features, base_value=%.4f",
            len(self._feature_names),
            float(np.mean(self._explainer.expected_value))
            if hasattr(self._explainer.expected_value, "__len__")
            else self._explainer.expected_value,
        )

    def _get_feature_names(self) -> list[str]:
        """Extract feature names after WoE transformation."""
        try:
            woe = self._base_pipeline.named_steps["woe"]
            return woe.get_feature_names_out()
        except (KeyError, AttributeError):
            return []

    def explain(
        self,
        X: pd.DataFrame,
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Compute SHAP values for a single prediction.

        Args:
            X: Single-row feature DataFrame (raw, pre-preprocessing).
            top_n: Number of top features to include in summary.

        Returns:
            Dict with keys:
                - shap_values: feature → SHAP value
                - base_value: model expected value
                - top_factors: list of top_n {feature, shap, direction}
        """
        # Transform through preprocessing only
        X_transformed = _transform_through_pipeline(self._base_pipeline, X)

        # Compute SHAP values
        shap_vals = self._explainer.shap_values(X_transformed)

        # For binary classification, shap_vals may be shape (1, n_features)
        # or a list [class0_vals, class1_vals]
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]  # class 1 (default) for first row
        else:
            vals = shap_vals[0] if shap_vals.ndim == 2 else shap_vals

        # Expected value
        ev = self._explainer.expected_value
        if hasattr(ev, "__len__"):
            base_value = float(ev[1])
        else:
            base_value = float(ev)

        # Build feature → shap map
        names = self._feature_names or [f"f{i}" for i in range(len(vals))]
        shap_map = {name: float(v) for name, v in zip(names, vals)}

        # Top N by absolute value
        sorted_items = sorted(shap_map.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top_factors = [
            {
                "feature": feat,
                "shap_impact": round(sv, 4),
                "direction": "increases_risk" if sv > 0 else "decreases_risk",
                "rank": i + 1,
            }
            for i, (feat, sv) in enumerate(sorted_items[:top_n])
        ]

        return {
            "shap_values": {k: round(v, 4) for k, v in shap_map.items()},
            "base_value": round(base_value, 4),
            "top_factors": top_factors,
        }
