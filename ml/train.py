"""LoanGuard AI — Model Training Script.

Orchestrates the full training pipeline:
    1. Generate (or load) dataset
    2. Split into train / val / test
    3. Feature selection via IV filtering
    4. Optuna hyperparameter search (XGBoost champion)
    5. Train final calibrated model
    6. Train LightGBM challenger
    7. Evaluate both on held-out test set
    8. Log everything to MLflow
    9. Save versioned model artifacts

Usage:
    python -m loanguard.ml.train
    python -m loanguard.ml.train --n-trials 20 --no-challenger
    LOANGUARD_OPTUNA_N_TRIALS=10 python -m loanguard.ml.train
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

# ── Local imports ─────────────────────────────────────────────────────────────
# Support running as both `python train.py` and `python -m loanguard.ml.train`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import get_settings
from data.generator import generate_credit_dataset, save_dataset
from data.loader import (
    ID_COL,
    load_and_split,
    load_dataset,
    split_dataset,
)
from ml.evaluate import EvaluationResults, evaluate_model, probability_to_score
from ml.features import (
    WoEEncoder,
    add_derived_features,
    compute_iv_table,
    select_features_by_iv,
)
from ml.pipeline import build_challenger_pipeline, build_pipeline

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("loanguard.train")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Artifact helpers ──────────────────────────────────────────────────────────

def _save_artifact(obj: Any, path: Path, meta: dict | None = None) -> None:
    """Pickle an object and write optional metadata JSON alongside it.

    Args:
        obj: Python object to serialise.
        path: Destination .pkl file.
        meta: Optional metadata dict written to <path>.meta.json.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved artifact → %s (%d bytes)", path, path.stat().st_size)

    if meta:
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        logger.info("Saved metadata → %s", meta_path)


def _load_artifact(path: Path) -> Any:
    """Load a pickled artifact.

    Args:
        path: Path to .pkl file.

    Returns:
        Deserialised object.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ── Optuna objective ──────────────────────────────────────────────────────────

def _xgb_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    selected_features: list[str],
    class_weight_ratio: float,
    cv_folds: int,
    calibration_method: str,
) -> float:
    """Optuna objective: return validation AUC-ROC.

    Args:
        trial: Optuna trial object.
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        selected_features: IV-filtered feature list.
        class_weight_ratio: n_neg / n_pos.
        cv_folds: CV folds for calibration.
        calibration_method: 'sigmoid' or 'isotonic'.

    Returns:
        Validation AUC-ROC (higher = better).
    """
    from sklearn.metrics import roc_auc_score

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    pipeline = build_pipeline(
        params=params,
        selected_features=selected_features,
        calibration_method=calibration_method,
        cv_calibration=cv_folds,
        class_weight_ratio=class_weight_ratio,
    )

    # Combine train + val for CV (val is only used here to guide Optuna)
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)

    pipeline.fit(X_combined, y_combined)
    y_prob_val = pipeline.predict_proba(X_val)[:, 1]
    auc = float(roc_auc_score(y_val, y_prob_val))
    return auc


# ── Feature selection helper ──────────────────────────────────────────────────

def _run_iv_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    iv_threshold: float,
) -> tuple[list[str], pd.DataFrame]:
    """Fit a WoE encoder on training data, compute IV, select features.

    Args:
        X_train: Training feature DataFrame (raw, before pipeline).
        y_train: Binary target.
        iv_threshold: Minimum IV to keep a feature.

    Returns:
        Tuple of (selected_feature_names, iv_table DataFrame).
    """
    # Add derived features before IV selection so ratios can be ranked
    X_derived = add_derived_features(X_train.drop(columns=[ID_COL], errors="ignore"))

    woe = WoEEncoder()
    woe.fit(X_derived, y_train)
    iv_table = compute_iv_table(woe)
    selected = select_features_by_iv(iv_table, threshold=iv_threshold)

    logger.info("IV selection: %d / %d features kept", len(selected), len(X_derived.columns))
    return selected, iv_table


# ── Main training orchestrator ────────────────────────────────────────────────

def train(
    n_trials: int | None = None,
    train_challenger: bool = True,
    data_path: str | Path | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Full training run with Optuna + MLflow.

    Args:
        n_trials: Number of Optuna trials. Defaults to settings value.
        train_challenger: Whether to also train the LightGBM challenger.
        data_path: Path to dataset. Generates synthetic data if None.
        run_name: MLflow run name override.

    Returns:
        Dict with 'champion_path', 'challenger_path', 'champion_results',
        'challenger_results', 'iv_table'.
    """
    cfg = get_settings()
    n_trials = n_trials or cfg.optuna_n_trials

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    if data_path is None:
        raw_path = Path(cfg.data_dir) / cfg.dataset_filename
        if not raw_path.exists():
            logger.info("No dataset found — generating synthetic data...")
            df = generate_credit_dataset(
                n_samples=cfg.synthetic_n_samples,
                seed=cfg.synthetic_random_seed,
            )
            raw_path = save_dataset(df, output_dir=cfg.data_dir)
        data_path = raw_path

    splits, meta = load_and_split(
        data_path,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        seed=cfg.synthetic_random_seed,
    )
    logger.info("Data loaded: %s", meta)

    X_train, y_train = splits.X_train, splits.y_train
    X_val, y_val = splits.X_val, splits.y_val
    X_test, y_test = splits.X_test, splits.y_test

    # Class weight for imbalanced data
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    class_weight_ratio = n_neg / max(n_pos, 1)
    logger.info(
        "Class balance: pos=%d, neg=%d, ratio=%.2f",
        n_pos, n_neg, class_weight_ratio,
    )

    # ── 2. Feature selection ──────────────────────────────────────────────────
    selected_features, iv_table = _run_iv_selection(
        X_train, y_train, iv_threshold=cfg.iv_filter_threshold
    )
    logger.info("\n%s", iv_table.to_string(index=False))

    # ── 3. MLflow setup ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"loanguard_train_{timestamp}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: %s (id=%s)", run_name, run_id)

        # Log dataset metadata
        mlflow.log_params({
            "data_source": str(data_path),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "default_rate_train": float(y_train.mean()),
            "n_features_selected": len(selected_features),
            "iv_threshold": cfg.iv_filter_threshold,
            "calibration_method": cfg.calibration_method,
            "cv_folds": cfg.cv_folds,
            "n_optuna_trials": n_trials,
        })

        # Log IV table as artifact
        iv_path = Path(cfg.model_dir) / "iv_table.csv"
        iv_path.parent.mkdir(parents=True, exist_ok=True)
        iv_table.to_csv(iv_path, index=False)
        mlflow.log_artifact(str(iv_path), artifact_path="feature_selection")

        # ── 4. Optuna hyperparameter search ───────────────────────────────────
        logger.info("Starting Optuna search: %d trials...", n_trials)
        t0 = time.time()

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=cfg.synthetic_random_seed),
            study_name=f"xgb_{timestamp}",
        )
        study.optimize(
            lambda trial: _xgb_objective(
                trial,
                X_train, y_train,
                X_val, y_val,
                selected_features,
                class_weight_ratio,
                cfg.cv_folds,
                cfg.calibration_method,
            ),
            n_trials=n_trials,
            timeout=cfg.optuna_timeout_seconds,
            show_progress_bar=False,
        )

        tuning_time = time.time() - t0
        best_params = study.best_params
        best_val_auc = study.best_value

        logger.info(
            "Optuna done in %.1fs: best_val_AUC=%.4f, params=%s",
            tuning_time, best_val_auc, best_params,
        )
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_val_auc", best_val_auc)
        mlflow.log_metric("optuna_tuning_seconds", tuning_time)

        # ── 5. Train final champion on train+val ──────────────────────────────
        logger.info("Training final XGBoost champion...")
        X_trainval = pd.concat([X_train, X_val], ignore_index=True)
        y_trainval = pd.concat([y_train, y_val], ignore_index=True)

        champion = build_pipeline(
            params=best_params,
            selected_features=selected_features,
            calibration_method=cfg.calibration_method,
            cv_calibration=cfg.cv_folds,
            class_weight_ratio=class_weight_ratio,
        )
        t_fit = time.time()
        champion.fit(X_trainval, y_trainval)
        fit_time = time.time() - t_fit
        logger.info("Champion trained in %.1fs", fit_time)

        # Evaluate champion
        train_probs = champion.predict_proba(X_trainval)[:, 1]
        champ_results = evaluate_model(
            champion, X_test, y_test,
            train_probs=train_probs,
            cost_fn=cfg.cost_fn,
            cost_fp=cfg.cost_fp,
        )
        logger.info("\n%s", champ_results.summary())

        # Log champion metrics
        mlflow.log_metrics(
            {f"champion_{k}": v for k, v in champ_results.to_dict().items()
             if isinstance(v, (int, float))}
        )
        mlflow.log_metric("champion_fit_seconds", fit_time)

        # Save and register champion
        champ_meta = {
            "model_type": "XGBoost",
            "version": timestamp,
            "run_id": run_id,
            "n_features": len(selected_features),
            "selected_features": selected_features,
            "best_params": best_params,
            "auc_roc": champ_results.auc_roc,
            "ks_statistic": champ_results.ks_statistic,
            "gini": champ_results.gini,
            "optimal_threshold": champ_results.optimal_threshold,
            "training_timestamp": timestamp,
            "calibration_method": cfg.calibration_method,
        }
        champ_path = Path(cfg.model_dir) / f"champion_{timestamp}.pkl"
        _save_artifact(champion, champ_path, meta=champ_meta)

        # Also save as "latest" for easy loading
        latest_path = Path(cfg.model_dir) / "champion_latest.pkl"
        _save_artifact(champion, latest_path, meta=champ_meta)

        mlflow.sklearn.log_model(
            champion,
            artifact_path="champion_model",
            registered_model_name=cfg.champion_model_name,
        )
        mlflow.log_artifact(str(champ_path), artifact_path="artifacts")

        # ── 6. Train LightGBM challenger ──────────────────────────────────────
        challenger_results: EvaluationResults | None = None
        chall_path: Path | None = None

        if train_challenger:
            logger.info("Training LightGBM challenger...")
            challenger = build_challenger_pipeline(
                selected_features=selected_features,
                calibration_method=cfg.calibration_method,
                cv_calibration=cfg.cv_folds,
            )
            t_chall = time.time()
            challenger.fit(X_trainval, y_trainval)
            chall_fit_time = time.time() - t_chall

            challenger_results = evaluate_model(
                challenger, X_test, y_test,
                train_probs=train_probs,
                cost_fn=cfg.cost_fn,
                cost_fp=cfg.cost_fp,
            )
            logger.info(
                "Challenger — AUC=%.4f | KS=%.4f | Gini=%.4f",
                challenger_results.auc_roc,
                challenger_results.ks_statistic,
                challenger_results.gini,
            )

            mlflow.log_metrics(
                {f"challenger_{k}": v for k, v in challenger_results.to_dict().items()
                 if isinstance(v, (int, float))}
            )
            mlflow.log_metric("challenger_fit_seconds", chall_fit_time)

            chall_meta = {
                "model_type": "LightGBM",
                "version": timestamp,
                "run_id": run_id,
                "auc_roc": challenger_results.auc_roc,
                "training_timestamp": timestamp,
            }
            chall_path = Path(cfg.model_dir) / f"challenger_{timestamp}.pkl"
            _save_artifact(challenger, chall_path, meta=chall_meta)

            mlflow.sklearn.log_model(
                challenger,
                artifact_path="challenger_model",
                registered_model_name=cfg.challenger_model_name,
            )

            # Champion vs challenger comparison
            delta_auc = champ_results.auc_roc - challenger_results.auc_roc
            mlflow.log_metric("champion_vs_challenger_auc_delta", delta_auc)
            logger.info(
                "Champion vs Challenger ΔAuC=%.4f (%s wins)",
                abs(delta_auc),
                "Champion" if delta_auc >= 0 else "Challenger",
            )

        # ── 7. Summary log ────────────────────────────────────────────────────
        mlflow.set_tags({
            "model_version": timestamp,
            "data_version": meta.source,
            "stage": "training",
            "champion_auc_passes": str(champ_results.auc_roc >= 0.75),
        })

        logger.info(
            "Training complete. Champion AUC=%.4f | KS=%.4f | Gini=%.4f",
            champ_results.auc_roc,
            champ_results.ks_statistic,
            champ_results.gini,
        )

    return {
        "champion_path": champ_path,
        "challenger_path": chall_path,
        "champion_results": champ_results,
        "challenger_results": challenger_results,
        "iv_table": iv_table,
        "selected_features": selected_features,
        "run_id": run_id,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoanGuard AI — Train credit risk model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Optuna trials (overrides settings)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to dataset file (.parquet or .csv). Generates synthetic if omitted.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name",
    )
    parser.add_argument(
        "--no-challenger",
        action="store_true",
        default=False,
        help="Skip LightGBM challenger training",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = train(
        n_trials=args.n_trials,
        train_challenger=not args.no_challenger,
        data_path=args.data_path,
        run_name=args.run_name,
    )
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Champion model  : {results['champion_path']}")
    print(f"MLflow run ID   : {results['run_id']}")
    print(results["champion_results"].summary())
    if results["challenger_results"]:
        print("\nChallenger:")
        print(results["challenger_results"].summary())
