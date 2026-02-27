"""Application configuration via environment variables (12-factor style).

Usage:
    from config.settings import get_settings
    cfg = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """LoanGuard AI — Stage 1 configuration.

    Every value has a sensible default; override via env vars prefixed
    with ``LOANGUARD_`` (e.g. ``LOANGUARD_MLFLOW_URI=http://…``).
    """

    # ── Paths ─────────────────────────────────────────────────────────────────
    base_dir: Path = Field(default=Path(__file__).resolve().parents[1])
    data_dir: Path = Field(default=Path("data/raw"))
    model_dir: Path = Field(default=Path("models"))

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_filename: str = Field(default="credit_risk.parquet")
    synthetic_n_samples: int = Field(default=50_000, ge=1_000)
    synthetic_random_seed: int = Field(default=42)
    test_size: float = Field(default=0.20, gt=0.0, lt=1.0)
    val_size: float = Field(default=0.10, gt=0.0, lt=1.0)

    # ── Feature engineering ───────────────────────────────────────────────────
    woe_min_bin_pct: float = Field(default=0.05, description="Min % samples per WoE bin")
    iv_filter_threshold: float = Field(default=0.02, description="Drop features with IV < this")
    vif_threshold: float = Field(default=10.0, description="Drop features with VIF > this")

    # ── Model / training ──────────────────────────────────────────────────────
    cv_folds: int = Field(default=5, ge=2)
    optuna_n_trials: int = Field(default=50, ge=1)          # 100 in prod; 50 for speed
    optuna_timeout_seconds: int = Field(default=600)
    calibration_method: Literal["sigmoid", "isotonic"] = Field(default="sigmoid")
    champion_model_name: str = Field(default="loanguard_xgb")

    # ── Scorecard ─────────────────────────────────────────────────────────────
    score_min: int = Field(default=300)
    score_max: int = Field(default=850)
    score_base_score: float = Field(default=600.0)
    score_base_odds: float = Field(default=50.0)
    score_pdo: float = Field(default=20.0, description="Points to double the odds")

    # ── Decision thresholds ───────────────────────────────────────────────────
    threshold_approve: float = Field(default=0.30)
    threshold_decline: float = Field(default=0.70)

    # ── Cost matrix (FN costs 5× more than FP) ────────────────────────────────
    cost_fn: float = Field(default=5.0, description="Cost of missing a defaulter")
    cost_fp: float = Field(default=1.0, description="Cost of wrongly declining good borrower")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(default="mlruns")
    mlflow_experiment_name: str = Field(default="loanguard_credit_risk")

    # ── Misc ──────────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    model_config = {"env_prefix": "LOANGUARD_", "case_sensitive": False}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
