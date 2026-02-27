"""Feature engineering for credit risk modelling.

Implements:
- Weight of Evidence (WoE) encoding with supervised binning
- Information Value (IV) calculation and filtering
- VIF-based multicollinearity check
- Domain-driven ratio / interaction feature creation

Usage:
    from ml.features import WoEEncoder, compute_iv_table, select_features_by_iv
    enc = WoEEncoder(min_bin_pct=0.05)
    enc.fit(X_train, y_train)
    X_woe = enc.transform(X_train)
    iv_table = compute_iv_table(enc)
    selected = select_features_by_iv(iv_table, threshold=0.02)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer

logger = logging.getLogger(__name__)

_IV_LABELS = [
    (0.0, 0.02, "Useless"),
    (0.02, 0.1, "Weak"),
    (0.1, 0.3, "Medium"),
    (0.3, 0.5, "Strong"),
    (0.5, float("inf"), "Very Strong (check for leakage)"),
]


def _iv_label(iv: float) -> str:
    for lo, hi, label in _IV_LABELS:
        if lo <= iv < hi:
            return label
    return "Unknown"


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Weight of Evidence encoder for supervised credit risk feature encoding.

    Replaces raw feature values with their WoE scores so that any downstream
    model (linear or tree-based) sees a monotone, calibrated risk signal.

    Attributes:
        min_bin_pct: Minimum fraction of samples per WoE bin.
        max_bins: Maximum number of bins for numeric features.
        smoothing: Laplace smoothing applied to counts.
        woe_maps_: Mapping of feature → {bin_label: woe_value}.
        iv_: Mapping of feature → Information Value.
        bin_edges_: Mapping of numeric feature → bin edge array.
    """

    def __init__(
        self,
        min_bin_pct: float = 0.05,
        max_bins: int = 10,
        smoothing: float = 0.5,
    ) -> None:
        self.min_bin_pct = min_bin_pct
        self.max_bins = max_bins
        self.smoothing = smoothing

        # Set during fit
        self.woe_maps_: dict[str, dict[str, float]] = {}
        self.iv_: dict[str, float] = {}
        self.bin_edges_: dict[str, np.ndarray] = {}
        self._numeric_cols: list[str] = []
        self._cat_cols: list[str] = []

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        """Compute WoE maps from training data.

        Args:
            X: Feature DataFrame.
            y: Binary target Series (1 = event/default).

        Returns:
            Fitted self.

        Raises:
            ValueError: If y contains only one class.
        """
        y_arr = np.array(y, dtype=int)
        n_ev = y_arr.sum()
        n_nev = len(y_arr) - n_ev
        if n_ev == 0 or n_nev == 0:
            raise ValueError("y must contain both classes (0 and 1).")

        self._numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        self._cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        for col in self._numeric_cols:
            self._fit_numeric(X[col], y_arr, col, n_ev, n_nev)
        for col in self._cat_cols:
            self._fit_categorical(X[col], y_arr, col, n_ev, n_nev)

        logger.info(
            "WoEEncoder fitted: %d numeric + %d categorical features",
            len(self._numeric_cols), len(self._cat_cols),
        )
        return self

    def _fit_numeric(
        self,
        series: pd.Series,
        y: np.ndarray,
        col: str,
        n_ev: int,
        n_nev: int,
    ) -> None:
        valid_mask = series.notna().values
        x_valid = series.values[valid_mask].reshape(-1, 1)
        y_valid = y[valid_mask]

        n_bins = max(2, min(self.max_bins, int(1.0 / max(self.min_bin_pct, 0.01))))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
                kbd.fit(x_valid)
            except Exception:
                kbd = KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="uniform")
                kbd.fit(x_valid)

        edges = kbd.bin_edges_[0]
        self.bin_edges_[col] = edges
        bins = kbd.transform(x_valid).flatten().astype(int)

        woe_map: dict[str, float] = {}
        iv_total = 0.0
        actual_bins = np.unique(bins)
        n_actual_bins = len(actual_bins)

        for b in actual_bins:
            mask_b = bins == b
            lo = edges[b]
            hi = edges[b + 1] if b + 1 < len(edges) else edges[-1]
            label = f"({lo:.4g}, {hi:.4g}]"

            ev = float(y_valid[mask_b].sum()) + self.smoothing
            nev = float(mask_b.sum() - y_valid[mask_b].sum()) + self.smoothing
            dist_ev = ev / (n_ev + self.smoothing * n_actual_bins)
            dist_nev = nev / (n_nev + self.smoothing * n_actual_bins)
            woe = float(np.log(dist_ev / dist_nev)) if dist_nev > 0 else 0.0
            woe_map[label] = woe
            iv_total += (dist_ev - dist_nev) * woe

        # Missing-value bin
        miss_mask = ~valid_mask
        if miss_mask.sum() > 0:
            ev = float(y[miss_mask].sum()) + self.smoothing
            nev = float(miss_mask.sum() - y[miss_mask].sum()) + self.smoothing
            dist_ev = ev / (n_ev + self.smoothing)
            dist_nev = nev / (n_nev + self.smoothing)
            woe_map["MISSING"] = float(np.log(dist_ev / dist_nev)) if dist_nev > 0 else 0.0
        else:
            woe_map["MISSING"] = 0.0

        self.woe_maps_[col] = woe_map
        self.iv_[col] = iv_total

    def _fit_categorical(
        self,
        series: pd.Series,
        y: np.ndarray,
        col: str,
        n_ev: int,
        n_nev: int,
    ) -> None:
        categories = [c for c in series.dropna().unique()]
        n_cats = len(categories) + 1
        woe_map: dict[str, float] = {}
        iv_total = 0.0

        for cat in categories:
            mask_c = (series == cat).values
            ev = float(y[mask_c].sum()) + self.smoothing
            nev = float(mask_c.sum() - y[mask_c].sum()) + self.smoothing
            dist_ev = ev / (n_ev + self.smoothing * n_cats)
            dist_nev = nev / (n_nev + self.smoothing * n_cats)
            woe = float(np.log(dist_ev / dist_nev)) if dist_nev > 0 else 0.0
            iv_total += (dist_ev - dist_nev) * woe
            woe_map[str(cat)] = woe

        miss_mask = series.isna().values
        if miss_mask.sum() > 0:
            ev = float(y[miss_mask].sum()) + self.smoothing
            nev = float(miss_mask.sum() - y[miss_mask].sum()) + self.smoothing
            dist_ev = ev / (n_ev + self.smoothing)
            dist_nev = nev / (n_nev + self.smoothing)
            woe_map["MISSING"] = float(np.log(dist_ev / dist_nev)) if dist_nev > 0 else 0.0
        else:
            woe_map["MISSING"] = 0.0

        self.woe_maps_[col] = woe_map
        self.iv_[col] = iv_total

    # ── transform ────────────────────────────────────────────────────────────

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace feature values with WoE scores.

        Args:
            X: Feature DataFrame with same columns as fitted on.

        Returns:
            All-numeric DataFrame containing WoE scores.

        Raises:
            RuntimeError: If encoder has not been fitted.
        """
        if not self.woe_maps_:
            raise RuntimeError("Call fit() before transform().")

        result = pd.DataFrame(index=X.index)

        for col in self._numeric_cols:
            if col not in X.columns:
                result[col] = 0.0
                continue
            edges = self.bin_edges_[col]
            woe_map = self.woe_maps_[col]
            series = X[col]
            woe_vals = np.zeros(len(series), dtype=float)

            for i, val in enumerate(series):
                if pd.isna(val):
                    woe_vals[i] = woe_map.get("MISSING", 0.0)
                else:
                    bin_idx = int(np.searchsorted(edges[1:-1], float(val), side="right"))
                    lo = edges[bin_idx]
                    hi = edges[bin_idx + 1] if bin_idx + 1 < len(edges) else edges[-1]
                    label = f"({lo:.4g}, {hi:.4g}]"
                    woe_vals[i] = woe_map.get(label, 0.0)

            result[col] = woe_vals

        for col in self._cat_cols:
            if col not in X.columns:
                result[col] = 0.0
                continue
            woe_map = self.woe_maps_[col]
            result[col] = X[col].apply(
                lambda v: woe_map.get(str(v), woe_map.get("MISSING", 0.0))
                if pd.notna(v) else woe_map.get("MISSING", 0.0)
            )

        return result

    def get_feature_names_out(self, input_features: object = None) -> list[str]:
        """Return output feature names."""
        return self._numeric_cols + self._cat_cols


# ── IV utilities ─────────────────────────────────────────────────────────────

def compute_iv_table(encoder: WoEEncoder) -> pd.DataFrame:
    """Build a summary IV table from a fitted WoEEncoder.

    Args:
        encoder: Fitted WoEEncoder.

    Returns:
        DataFrame with columns: feature, iv, strength, n_bins.
    """
    rows = [
        {
            "feature": feat,
            "iv": round(iv, 4),
            "strength": _iv_label(iv),
            "n_bins": len(encoder.woe_maps_.get(feat, {})),
        }
        for feat, iv in encoder.iv_.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )


def select_features_by_iv(
    iv_table: pd.DataFrame,
    threshold: float = 0.02,
) -> list[str]:
    """Return feature names with IV ≥ threshold.

    Args:
        iv_table: Output of ``compute_iv_table``.
        threshold: Minimum IV to keep.

    Returns:
        List of selected feature names.
    """
    selected = iv_table.loc[iv_table["iv"] >= threshold, "feature"].tolist()
    dropped = iv_table.loc[iv_table["iv"] < threshold, "feature"].tolist()
    logger.info(
        "IV filter (≥ %.3f): kept=%d, dropped=%d %s",
        threshold, len(selected), len(dropped), dropped,
    )
    return selected


# ── VIF check ────────────────────────────────────────────────────────────────

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each numeric feature.

    Args:
        X: Numeric DataFrame (rows with NaN are dropped internally).

    Returns:
        DataFrame with columns: feature, vif — sorted descending.
    """
    X_num = X.select_dtypes(include=np.number).dropna()
    X_np = X_num.values
    n = len(X_np)
    vifs = []

    for i, col in enumerate(X_num.columns):
        y_col = X_np[:, i]
        X_other = np.delete(X_np, i, axis=1)
        A = np.column_stack([np.ones(n), X_other])
        try:
            coeffs = np.linalg.lstsq(A, y_col, rcond=None)[0]
            y_hat = A @ coeffs
            ss_res = float(np.sum((y_col - y_hat) ** 2))
            ss_tot = float(np.sum((y_col - y_col.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        except Exception:
            vif = float("inf")
        vifs.append({"feature": col, "vif": round(vif, 2)})

    return (
        pd.DataFrame(vifs)
        .sort_values("vif", ascending=False)
        .reset_index(drop=True)
    )


def drop_high_vif_features(
    X: pd.DataFrame,
    threshold: float = 10.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Iteratively drop the highest-VIF feature until all VIF ≤ threshold.

    Args:
        X: Numeric DataFrame.
        threshold: Maximum acceptable VIF.

    Returns:
        Tuple of (filtered DataFrame, list of dropped column names).
    """
    dropped: list[str] = []
    X_work = X.copy()
    while True:
        vif_df = compute_vif(X_work)
        if vif_df.empty or vif_df.iloc[0]["vif"] <= threshold:
            break
        col_drop = vif_df.iloc[0]["feature"]
        logger.info(
            "VIF drop: '%s' VIF=%.2f > threshold=%.1f",
            col_drop, vif_df.iloc[0]["vif"], threshold,
        )
        X_work = X_work.drop(columns=[col_drop])
        dropped.append(col_drop)
    return X_work, dropped


# ── Derived features ──────────────────────────────────────────────────────────

def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add domain-informed ratio and interaction features.

    Creates new columns without modifying *X*.

    Args:
        X: Raw feature DataFrame.

    Returns:
        Expanded DataFrame with additional engineered columns.
    """
    df = X.copy()

    if {"loan_amount", "annual_income"}.issubset(df.columns):
        df["loan_to_income"] = df["loan_amount"] / (df["annual_income"].clip(lower=1))

    if {"revolving_balance", "annual_income"}.issubset(df.columns):
        df["revolving_to_income"] = df["revolving_balance"] / (df["annual_income"].clip(lower=1))

    if {"num_derog_records", "total_accounts"}.issubset(df.columns):
        df["derog_rate"] = df["num_derog_records"] / (df["total_accounts"].clip(lower=1))

    if {"num_open_accounts", "total_accounts"}.issubset(df.columns):
        df["open_account_rate"] = df["num_open_accounts"] / (df["total_accounts"].clip(lower=1))

    if {"interest_rate", "debt_to_income"}.issubset(df.columns):
        df["rate_x_dti"] = df["interest_rate"] * df["debt_to_income"]

    logger.debug("Derived features added; shape now %s", df.shape)
    return df
