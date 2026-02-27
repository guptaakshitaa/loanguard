"""Data ingestion layer.

Supports CSV, Parquet, and SQLAlchemy-compatible databases.
Always returns (X, y, meta) tuples with a consistent schema.

Usage:
    from data.loader import load_dataset, split_dataset
    X, y, meta = load_dataset("data/raw/credit_risk.parquet")
    splits = split_dataset(X, y, test_size=0.2, val_size=0.1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ── Schema constants ──────────────────────────────────────────────────────────
TARGET_COL = "default"
ID_COL = "application_id"
LEAKAGE_COLS = ["default_probability_true"]

NUMERIC_FEATURES = [
    "age", "annual_income", "employment_length_years", "loan_amount",
    "interest_rate", "loan_term_months", "num_open_accounts",
    "num_derog_records", "num_credit_inquiries", "credit_utilization_ratio",
    "debt_to_income", "months_since_last_delinq", "revolving_balance",
    "total_accounts",
]

CATEGORICAL_FEATURES = [
    "home_ownership", "loan_purpose", "grade", "verification_status",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class DatasetMeta:
    """Metadata attached to every loaded dataset."""
    source: str
    n_rows: int
    n_features: int
    default_rate: float
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    missing_summary: dict[str, float] = field(default_factory=dict)


@dataclass
class DataSplits:
    """Train / validation / test splits."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    ids_test: pd.Series


def _read_raw(source: str | Path) -> pd.DataFrame:
    """Read CSV or Parquet into a DataFrame.

    Args:
        source: File path ending in .parquet, .csv, or .tsv.

    Returns:
        Raw DataFrame.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If extension is unsupported.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")
    ext = path.suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in {".csv", ".tsv"}:
        df = pd.read_csv(path, sep="\t" if ext == ".tsv" else ",", low_memory=False)
    else:
        raise ValueError(f"Unsupported file format: {ext!r}")
    logger.info("Loaded %d rows × %d cols from %s", *df.shape, path.name)
    return df


def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leakage columns and coerce dtypes.

    Args:
        df: Raw DataFrame from disk.

    Returns:
        Cleaned DataFrame.
    """
    drop_cols = [c for c in LEAKAGE_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", np.nan)
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def _missing_summary(df: pd.DataFrame) -> dict[str, float]:
    pct = df.isnull().mean()
    return {col: round(float(v), 4) for col, v in pct.items() if v > 0}


def load_dataset(
    source: str | Path,
    drop_id: bool = False,
) -> tuple[pd.DataFrame, pd.Series, DatasetMeta]:
    """Load a credit-risk dataset from disk.

    Args:
        source: Path to .parquet or .csv file.
        drop_id: Remove application_id from X if True.

    Returns:
        Tuple of (X, y, DatasetMeta).

    Raises:
        FileNotFoundError: If file does not exist.
        KeyError: If target column is missing.
    """
    raw = _read_raw(source)
    df = _enforce_schema(raw)

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found.")

    y = df[TARGET_COL].copy()
    exclude = {TARGET_COL}
    if drop_id and ID_COL in df.columns:
        exclude.add(ID_COL)
    X = df.drop(columns=list(exclude))

    num_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    meta = DatasetMeta(
        source=str(source),
        n_rows=len(df),
        n_features=len(X.columns),
        default_rate=round(float(y.mean()), 4),
        numeric_features=num_cols,
        categorical_features=cat_cols,
        missing_summary=_missing_summary(X),
    )
    logger.info(
        "Dataset: %d rows | %d features | default_rate=%.3f",
        meta.n_rows, meta.n_features, meta.default_rate,
    )
    return X, y, meta


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = 42,
    stratify: bool = True,
) -> DataSplits:
    """Split into stratified train / validation / test sets.

    Validation is carved from training so the test set is never touched
    during hyperparameter search.

    Args:
        X: Feature DataFrame.
        y: Binary target Series.
        test_size: Fraction for final test evaluation.
        val_size: Fraction of *total* data for validation.
        seed: Random seed.
        stratify: Preserve class proportions across splits.

    Returns:
        DataSplits dataclass.
    """
    strat = y if stratify else None
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, random_state=seed,
        stratify=y_tv if stratify else None,
    )
    ids_test = (
        X_test[ID_COL].reset_index(drop=True)
        if ID_COL in X_test.columns
        else pd.Series(range(len(X_test)), name=ID_COL)
    )
    logger.info(
        "Splits → train=%d | val=%d | test=%d",
        len(X_train), len(X_val), len(X_test),
    )
    return DataSplits(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        ids_test=ids_test,
    )


def load_and_split(
    source: str | Path,
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = 42,
) -> tuple[DataSplits, DatasetMeta]:
    """One-call convenience: load + split.

    Args:
        source: Path to dataset file.
        test_size: Test fraction.
        val_size: Validation fraction.
        seed: Random seed.

    Returns:
        Tuple of (DataSplits, DatasetMeta).
    """
    X, y, meta = load_dataset(source, drop_id=False)
    splits = split_dataset(X, y, test_size=test_size, val_size=val_size, seed=seed)
    return splits, meta
