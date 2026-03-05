"""LIME-based local explanation for individual predictions.

LIME perturbs the input around a single point and fits a local
linear model to approximate the complex model's behaviour there.
Complements SHAP for regulatory explainability.

Usage:
    from explainability.lime_explainer import LIMEExplainer
    explainer = LIMEExplainer(pipeline, X_train)
    weights = explainer.explain(X_row)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """LIME wrapper for the LoanGuard calibrated pipeline.

    Args:
        pipeline: Fitted CalibratedClassifierCV.
        X_train: Training DataFrame used to fit the LIME explainer
                 (provides feature statistics for perturbation).
        n_samples: Number of perturbed samples per explanation.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        pipeline: Any,
        X_train: pd.DataFrame,
        n_samples: int = 500,
        random_state: int = 42,
    ) -> None:
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError as exc:
            raise ImportError("lime is required: pip install lime") from exc

        # Drop ID column if present
        X_ref = X_train.drop(columns=["application_id"], errors="ignore")

        self._pipeline = pipeline
        self._feature_names = X_ref.columns.tolist()
        self._categorical_features = [
            i for i, col in enumerate(self._feature_names)
            if X_ref[col].dtype == object or X_ref[col].nunique() < 10
        ]
        self._n_samples = n_samples

        # Fit LIME on training data
        from lime.lime_tabular import LimeTabularExplainer
        self._explainer = LimeTabularExplainer(
            training_data=X_ref.values,
            feature_names=self._feature_names,
            categorical_features=self._categorical_features,
            class_names=["non_default", "default"],
            mode="classification",
            discretize_continuous=True,
            random_state=random_state,
        )
        logger.info(
            "LIMEExplainer ready: %d features, %d categorical",
            len(self._feature_names), len(self._categorical_features),
        )

    def _predict_fn(self, X_array: np.ndarray) -> np.ndarray:
        """Prediction function for LIME — takes numpy array, returns probas."""
        df = pd.DataFrame(X_array, columns=self._feature_names)
        return self._pipeline.predict_proba(df)

    def explain(self, X_row: pd.DataFrame, top_n: int = 8) -> dict[str, float]:
        """Generate LIME explanation for a single row.

        Args:
            X_row: Single-row DataFrame (raw features).
            top_n: Number of features to include.

        Returns:
            Dict of feature_description → local weight.
            Positive weight = increases default probability.
        """
        X_ref = X_row.drop(columns=["application_id"], errors="ignore")
        row_array = X_ref.values[0]

        exp = self._explainer.explain_instance(
            data_row=row_array,
            predict_fn=self._predict_fn,
            num_features=top_n,
            num_samples=self._n_samples,
            labels=(1,),  # explain class 1 (default)
        )

        # Extract feature → weight for default class
        weights = exp.as_list(label=1)
        result = {feat: round(float(weight), 4) for feat, weight in weights}
        return result
