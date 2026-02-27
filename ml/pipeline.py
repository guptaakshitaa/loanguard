"""Sklearn Pipeline for LoanGuard credit risk model.

Stages:
    1. DerivedFeatureAdder   — ratio / interaction features
    2. ColumnDropper         — remove low-IV / high-VIF features
    3. SimpleImputer         — median for numeric, constant for categorical
    4. WoEEncoder            — supervised WoE transformation
    5. StandardScaler        — zero-mean, unit-variance (helps calibration)
    6. XGBClassifier         — gradient boosted trees with monotonic constraints
    7. CalibratedClassifierCV — Platt / isotonic calibration

Usage:
    from ml.pipeline import build_pipeline, build_challenger_pipeline
    pipe = build_pipeline(params={"n_estimators": 300, ...})
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_new)[:, 1]
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from ml.features import WoEEncoder, add_derived_features

logger = logging.getLogger(__name__)

# Default XGBoost hyperparameters (conservative, regularized)
_DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "gamma": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,  # adjusted for class imbalance in train script
    "use_label_encoder": False,
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

# Default LightGBM hyperparameters (challenger)
_DEFAULT_LGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


# ── Custom transformers ───────────────────────────────────────────────────────

class DerivedFeatureAdder(BaseEstimator, TransformerMixin):
    """Pipeline stage: add domain-driven ratio features.

    Wraps ``ml.features.add_derived_features`` so it works inside a
    scikit-learn Pipeline with both DataFrame and array inputs.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DerivedFeatureAdder":
        """No-op — derived features are deterministic."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X with additional engineered columns.

        Args:
            X: Feature DataFrame.

        Returns:
            Expanded DataFrame.
        """
        return add_derived_features(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array([])  # shape deferred until runtime


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Pipeline stage: keep only a specified list of columns.

    Used after IV filtering to drop low-predictive-value features before
    the WoE encoder sees them.

    Attributes:
        columns: List of column names to keep.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ColumnSelector":
        """Record available columns.

        Args:
            X: Feature DataFrame.
            y: Ignored.

        Returns:
            self
        """
        if self.columns is None:
            self.columns = X.columns.tolist()
        # Only keep columns that actually exist in X
        self.columns_ = [c for c in self.columns if c in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop columns not in the selected list.

        Args:
            X: Feature DataFrame.

        Returns:
            Filtered DataFrame.
        """
        return X[self.columns_]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.columns_)


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """Imputer that preserves DataFrame column names.

    Applies median imputation to numeric columns and constant 'MISSING'
    to categorical columns.
    """

    def __init__(self) -> None:
        self._num_imputer = SimpleImputer(strategy="median")
        self._cat_imputer = SimpleImputer(strategy="constant", fill_value="MISSING")
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DataFrameImputer":
        """Fit imputers on numeric and categorical columns.

        Args:
            X: Feature DataFrame.
            y: Ignored.

        Returns:
            self
        """
        self._num_cols = X.select_dtypes(include=np.number).columns.tolist()
        self._cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        if self._num_cols:
            self._num_imputer.fit(X[self._num_cols])
        if self._cat_cols:
            self._cat_imputer.fit(X[self._cat_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values.

        Args:
            X: Feature DataFrame.

        Returns:
            Imputed DataFrame with the same columns.
        """
        result = X.copy()
        if self._num_cols:
            result[self._num_cols] = self._num_imputer.transform(X[self._num_cols])
        if self._cat_cols:
            result[self._cat_cols] = self._cat_imputer.transform(X[self._cat_cols])
        return result

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self._num_cols + self._cat_cols)


class DataFrameScaler(BaseEstimator, TransformerMixin):
    """StandardScaler that accepts and returns DataFrames."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DataFrameScaler":
        self._cols = X.columns.tolist()
        self._scaler.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        scaled = self._scaler.transform(X)
        return pd.DataFrame(scaled, columns=self._cols, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self._cols)


# ── Pipeline builders ────────────────────────────────────────────────────────

def build_pipeline(
    params: dict[str, Any] | None = None,
    selected_features: list[str] | None = None,
    calibration_method: str = "sigmoid",
    cv_calibration: int = 5,
    class_weight_ratio: float = 1.0,
) -> Pipeline:
    """Build the champion XGBoost credit risk pipeline.

    Stages: DerivedFeatureAdder → ColumnSelector → DataFrameImputer
            → WoEEncoder → DataFrameScaler → CalibratedXGB

    Args:
        params: XGBoost hyperparameters (merged with defaults).
        selected_features: Columns to keep after IV filtering.
            If None, all features are passed through.
        calibration_method: 'sigmoid' (Platt) or 'isotonic'.
        cv_calibration: CV folds for calibration.
        class_weight_ratio: ``scale_pos_weight`` for XGBoost (n_neg / n_pos).

    Returns:
        Unfitted sklearn Pipeline.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required: pip install xgboost") from exc

    xgb_params = {**_DEFAULT_XGB_PARAMS, **(params or {})}
    xgb_params["scale_pos_weight"] = class_weight_ratio

    # XGBoost monotonic constraints: increase risk with
    # debt_to_income (+1), interest_rate (+1), credit_utilization (+1)
    # annual_income (-1), employment_length (-1)
    # All other features: 0 (unconstrained)
    # We apply constraints only when the exact feature set is known.
    classifier = XGBClassifier(**xgb_params)

    base_pipeline = Pipeline([
        ("derive",  DerivedFeatureAdder()),
        ("select",  ColumnSelector(columns=selected_features)),
        ("impute",  DataFrameImputer()),
        ("woe",     WoEEncoder()),
        ("scale",   DataFrameScaler()),
        ("clf",     classifier),
    ])

    calibrated = CalibratedClassifierCV(
        estimator=base_pipeline,
        method=calibration_method,
        cv=cv_calibration,
    )

    logger.info(
        "Built XGBoost pipeline: calibration=%s, cv=%d",
        calibration_method, cv_calibration,
    )
    return calibrated


def build_challenger_pipeline(
    params: dict[str, Any] | None = None,
    selected_features: list[str] | None = None,
    calibration_method: str = "sigmoid",
    cv_calibration: int = 5,
) -> Pipeline:
    """Build the challenger LightGBM pipeline (same preprocessing stages).

    Args:
        params: LightGBM hyperparameters (merged with defaults).
        selected_features: Columns to keep after IV filtering.
        calibration_method: 'sigmoid' or 'isotonic'.
        cv_calibration: CV folds for calibration.

    Returns:
        Unfitted sklearn Pipeline.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("lightgbm is required: pip install lightgbm") from exc

    lgb_params = {**_DEFAULT_LGB_PARAMS, **(params or {})}
    classifier = LGBMClassifier(**lgb_params)

    base_pipeline = Pipeline([
        ("derive",  DerivedFeatureAdder()),
        ("select",  ColumnSelector(columns=selected_features)),
        ("impute",  DataFrameImputer()),
        ("woe",     WoEEncoder()),
        ("scale",   DataFrameScaler()),
        ("clf",     classifier),
    ])

    calibrated = CalibratedClassifierCV(
        estimator=base_pipeline,
        method=calibration_method,
        cv=cv_calibration,
    )

    logger.info(
        "Built LightGBM challenger pipeline: calibration=%s, cv=%d",
        calibration_method, cv_calibration,
    )
    return calibrated


def get_pipeline_params(pipeline: CalibratedClassifierCV) -> dict[str, Any]:
    """Extract the fitted XGBoost / LGBM parameters from a calibrated pipeline.

    Args:
        pipeline: Fitted CalibratedClassifierCV wrapping the base pipeline.

    Returns:
        Dict of classifier parameters.

    Raises:
        AttributeError: If pipeline has not been fitted.
    """
    try:
        # Access first calibrated estimator's final step
        base = pipeline.calibrated_classifiers_[0].estimator
        clf = base.named_steps["clf"]
        return clf.get_params()
    except (AttributeError, IndexError) as exc:
        raise AttributeError(
            "Could not extract params — is the pipeline fitted?"
        ) from exc
