"""Synthetic credit-risk dataset generator.

Generates 50k realistic loan applications with correlated features
and a non-linear default label. Used when no real CSV/Parquet is
available (CI, demo, unit tests).

Usage:
    from data.generator import generate_credit_dataset
    df = generate_credit_dataset(n_samples=50_000, seed=42)
    df.to_parquet("data/raw/credit_risk.parquet", index=False)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature metadata — drives correlated sampling
# ---------------------------------------------------------------------------
_FEATURE_SPECS: dict[str, dict] = {
    "age": {"dist": "normal", "loc": 40, "scale": 12, "clip": (18, 80)},
    "annual_income": {"dist": "lognormal", "mean": 10.8, "sigma": 0.6, "clip": (15_000, 500_000)},
    "employment_length_years": {"dist": "exponential", "scale": 5, "clip": (0, 40)},
    "loan_amount": {"dist": "lognormal", "mean": 10.0, "sigma": 0.7, "clip": (1_000, 100_000)},
    "interest_rate": {"dist": "normal", "loc": 13.5, "scale": 4.5, "clip": (5.0, 30.0)},
    "loan_term_months": {"dist": "choice", "choices": [36, 60], "probs": [0.55, 0.45]},
    "num_open_accounts": {"dist": "poisson", "lam": 10, "clip": (1, 40)},
    "num_derog_records": {"dist": "poisson", "lam": 0.3, "clip": (0, 10)},
    "num_credit_inquiries": {"dist": "poisson", "lam": 1.5, "clip": (0, 20)},
    "credit_utilization_ratio": {"dist": "beta", "a": 2, "b": 5, "clip": (0.0, 1.0)},
    "debt_to_income": {"dist": "normal", "loc": 0.18, "scale": 0.09, "clip": (0.0, 0.80)},
    "months_since_last_delinq": {"dist": "exponential", "scale": 30, "clip": (0, 120)},
    "revolving_balance": {"dist": "lognormal", "mean": 9.0, "sigma": 1.0, "clip": (0, 200_000)},
    "total_accounts": {"dist": "poisson", "lam": 22, "clip": (2, 60)},
    "home_ownership": {
        "dist": "choice",
        "choices": ["RENT", "OWN", "MORTGAGE", "OTHER"],
        "probs": [0.35, 0.15, 0.45, 0.05],
    },
    "loan_purpose": {
        "dist": "choice",
        "choices": [
            "debt_consolidation", "credit_card", "home_improvement",
            "other", "major_purchase", "medical",
        ],
        "probs": [0.40, 0.25, 0.12, 0.10, 0.08, 0.05],
    },
    "grade": {
        "dist": "choice",
        "choices": ["A", "B", "C", "D", "E", "F"],
        "probs": [0.20, 0.28, 0.25, 0.15, 0.08, 0.04],
    },
    "verification_status": {
        "dist": "choice",
        "choices": ["Not Verified", "Source Verified", "Verified"],
        "probs": [0.40, 0.30, 0.30],
    },
}


def _sample_feature(spec: dict, rng: Generator, n: int) -> np.ndarray:
    """Sample *n* values for a single feature according to *spec*."""
    dist = spec["dist"]
    if dist == "normal":
        vals = rng.normal(spec["loc"], spec["scale"], n)
    elif dist == "lognormal":
        vals = rng.lognormal(spec["mean"], spec["sigma"], n)
    elif dist == "exponential":
        vals = rng.exponential(spec["scale"], n)
    elif dist == "poisson":
        vals = rng.poisson(spec["lam"], n).astype(float)
    elif dist == "beta":
        vals = rng.beta(spec["a"], spec["b"], n)
    elif dist == "choice":
        vals = rng.choice(spec["choices"], size=n, p=spec["probs"])
        return vals  # categorical — no clipping
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    lo, hi = spec.get("clip", (-np.inf, np.inf))
    return np.clip(vals, lo, hi)


def _build_default_probability(df: pd.DataFrame) -> np.ndarray:
    """Compute a non-linear default probability from raw features.

    Combines domain-realistic risk drivers with controlled noise so
    that a trained model can achieve ~0.78–0.82 AUC.

    Args:
        df: Raw feature DataFrame (numeric columns must exist).

    Returns:
        Array of default probabilities in [0, 1].
    """
    # Grade risk mapping (ordinal)
    grade_risk = {"A": -1.5, "B": -0.8, "C": 0.0, "D": 0.7, "E": 1.3, "F": 2.0}
    grade_score = df["grade"].map(grade_risk).fillna(0.0).values

    # Normalise continuous drivers to roughly zero-mean unit-variance
    dti = (df["debt_to_income"].values - 0.18) / 0.09
    util = (df["credit_utilization_ratio"].values - 0.40) / 0.25
    derog = np.log1p(df["num_derog_records"].values)
    inq = (df["num_credit_inquiries"].values - 1.5) / 1.5
    rate = (df["interest_rate"].values - 13.5) / 4.5
    income = -(np.log(df["annual_income"].values + 1) - 10.8) / 0.6
    emp = -(np.log1p(df["employment_length_years"].values) - 1.6) / 0.8
    delinq = -(np.log1p(df["months_since_last_delinq"].values) - 3.0) / 1.0

    # Non-linear interaction: high DTI + high utilisation is extra risky
    interaction = dti * util * 0.5

    logit = (
        -2.5          # baseline (controls overall default rate ~12%)
        + 0.9  * dti
        + 0.8  * util
        + 0.7  * grade_score
        + 1.2  * derog
        + 0.4  * inq
        + 0.5  * rate
        + 0.4  * income
        + 0.3  * emp
        + 0.3  * delinq
        + 0.6  * interaction
    )

    # Inject calibrated noise
    rng_noise = np.random.default_rng(seed=99)
    logit += rng_noise.normal(0, 0.6, len(df))

    return 1.0 / (1.0 + np.exp(-logit))


def generate_credit_dataset(
    n_samples: int = 50_000,
    seed: int = 42,
    default_rate_target: float = 0.12,
) -> pd.DataFrame:
    """Generate a synthetic credit-risk dataset.

    Args:
        n_samples: Number of loan applications to generate.
        seed: Random seed for reproducibility.
        default_rate_target: Approximate fraction of defaults.
            Used only as a sanity check; actual rate is driven
            by the logistic model in ``_build_default_probability``.

    Returns:
        DataFrame with features + ``default`` (binary target, 1 = defaulted).

    Example:
        >>> df = generate_credit_dataset(1_000, seed=0)
        >>> df.shape[0]
        1000
        >>> "default" in df.columns
        True
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating synthetic dataset: n=%d, seed=%d", n_samples, seed)

    data: dict[str, np.ndarray] = {}
    for feature, spec in _FEATURE_SPECS.items():
        data[feature] = _sample_feature(spec, rng, n_samples)

    df = pd.DataFrame(data)

    # Introduce 3–8% missing values on selected features (realistic)
    missing_features = [
        "months_since_last_delinq",
        "num_derog_records",
        "employment_length_years",
        "revolving_balance",
    ]
    for feat in missing_features:
        mask = rng.random(n_samples) < 0.06
        df.loc[mask, feat] = np.nan

    # Numeric casts
    int_cols = [
        "loan_term_months", "num_open_accounts", "num_derog_records",
        "num_credit_inquiries", "total_accounts",
    ]
    for col in int_cols:
        df[col] = df[col].astype("Int64")  # nullable integer

    # Compute default label
    # Convert nullable-int columns to float before building probabilities
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    prob = _build_default_probability(df)
    df["default_probability_true"] = prob          # kept for evaluation only
    df["default"] = (rng.random(n_samples) < prob).astype(int)

    # Unique application ID
    df.insert(0, "application_id", [f"APP{i:07d}" for i in range(n_samples)])

    actual_rate = df["default"].mean()
    logger.info(
        "Dataset ready. Default rate=%.3f (target=%.3f), shape=%s",
        actual_rate, default_rate_target, df.shape,
    )
    return df


def save_dataset(
    df: pd.DataFrame,
    output_dir: str | Path = "data/raw",
    filename: str = "credit_risk.parquet",
) -> Path:
    """Persist dataset to Parquet.

    Args:
        df: Dataset returned by ``generate_credit_dataset``.
        output_dir: Directory to write into (created if absent).
        filename: Output filename (must end in .parquet or .csv).

    Returns:
        Path to the saved file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename

    if filename.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif filename.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {filename}")

    logger.info("Dataset saved → %s (%d rows)", path, len(df))
    return path


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    dataset = generate_credit_dataset(n_samples=n)
    saved = save_dataset(dataset)
    print(f"Saved to {saved}")
    print(dataset.describe())
