"""Model evaluation suite for LoanGuard credit risk.

Computes:
- AUC-ROC
- KS Statistic (max separation between cum. default and non-default distributions)
- Gini Coefficient (2 * AUC - 1)
- F1 / Precision / Recall at optimal cost-weighted threshold
- Calibration error (ECE)
- PSI on train vs test score distributions
- Cost-weighted confusion matrix (FN costs 5× FP)
- Risk scorecard conversion and validation

Usage:
    from ml.evaluate import evaluate_model, probability_to_score
    results = evaluate_model(pipeline, X_test, y_test)
    print(results.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ── Scorecard conversion ──────────────────────────────────────────────────────

def probability_to_score(
    prob: float | np.ndarray,
    base_score: float = 600.0,
    base_odds: float = 50.0,
    pdo: float = 20.0,
    score_min: int = 300,
    score_max: int = 850,
) -> int | np.ndarray:
    """Convert default probability to a 300–850 credit scorecard score.

    Uses the standard Points to Double the Odds (PDO) formula:
        factor = PDO / ln(2)
        offset = base_score - factor * ln(base_odds)
        score  = offset - factor * ln(p / (1 - p))

    Higher score → lower risk (mirrors FICO convention).

    Args:
        prob: Default probability in (0, 1).
        base_score: Score at which the odds equal base_odds.
        base_odds: Odds (non-events / events) at base_score.
        pdo: Points needed to double the odds.
        score_min: Floor (clamp low scores to this).
        score_max: Ceiling (clamp high scores to this).

    Returns:
        Integer score (or array of integers) in [score_min, score_max].
    """
    p = np.asarray(prob, dtype=float)
    p = np.clip(p, 1e-7, 1 - 1e-7)

    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    log_odds = np.log(p / (1.0 - p))
    raw_score = offset - factor * log_odds

    clipped = np.clip(np.round(raw_score), score_min, score_max).astype(int)
    return int(clipped) if clipped.ndim == 0 else clipped


def score_to_decision(
    score: int | np.ndarray,
    threshold_approve: int = 620,
    threshold_decline: int = 540,
) -> str | np.ndarray:
    """Map scorecard score to a lending decision.

    Args:
        score: Integer score(s) in [300, 850].
        threshold_approve: Scores at or above this → APPROVE.
        threshold_decline: Scores at or below this → DECLINE.

    Returns:
        Decision string(s): 'APPROVE', 'REVIEW', or 'DECLINE'.
    """
    scores = np.asarray(score)
    scalar = scores.ndim == 0
    scores = np.atleast_1d(scores)

    decisions = np.where(
        scores >= threshold_approve,
        "APPROVE",
        np.where(scores <= threshold_decline, "DECLINE", "REVIEW"),
    )
    return decisions[0] if scalar else decisions


# ── Metric functions ──────────────────────────────────────────────────────────

def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov separation statistic.

    Measures the maximum vertical distance between the cumulative
    distribution of event scores and non-event scores.

    Args:
        y_true: Binary labels (1 = default).
        y_prob: Predicted default probabilities.

    Returns:
        KS statistic in [0, 1]. Higher = better separation.
    """
    sorted_idx = np.argsort(y_prob)[::-1]
    y_sorted = y_true[sorted_idx]
    cum_events = np.cumsum(y_sorted) / y_true.sum()
    cum_non_events = np.cumsum(1 - y_sorted) / (1 - y_true).sum()
    return float(np.max(np.abs(cum_events - cum_non_events)))


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Compute Population Stability Index between two score distributions.

    PSI < 0.1  → no significant change
    PSI < 0.2  → moderate change (monitor)
    PSI >= 0.2 → significant change (investigate / retrain)

    Args:
        expected: Reference distribution (e.g., training scores).
        actual: Current distribution (e.g., production scores).
        n_bins: Number of equal-width bins.
        epsilon: Small constant to avoid log(0).

    Returns:
        PSI value.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    expected_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=bins)[0] / len(actual)

    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of equal-width probability bins.

    Returns:
        ECE value in [0, 1]. Closer to 0 = better calibrated.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(acc - conf)
    return float(ece)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
) -> float:
    """Find the probability threshold that minimises business cost.

    Business cost = FN × cost_fn + FP × cost_fp

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.
        cost_fn: Cost of a false negative (missed default).
        cost_fp: Cost of a false positive (wrongly declined).

    Returns:
        Optimal threshold in [0, 1].
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_cost = float("inf")
    best_thresh = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = fn * cost_fn + fp * cost_fp
        if cost < best_cost:
            best_cost = cost
            best_thresh = float(t)

    return best_thresh


def cost_weighted_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
) -> dict[str, Any]:
    """Compute confusion matrix with associated business cost.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        cost_fn: Relative cost of missing a defaulter.
        cost_fp: Relative cost of wrongly declining a good borrower.

    Returns:
        Dict with tn, fp, fn, tp, total_cost, cost_per_decision.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_cost = float(fn * cost_fn + fp * cost_fp)
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_cost": total_cost,
        "cost_per_decision": total_cost / len(y_true),
    }


# ── Results container ─────────────────────────────────────────────────────────

@dataclass
class EvaluationResults:
    """Container for all evaluation metrics."""

    auc_roc: float
    ks_statistic: float
    gini: float
    brier_score: float
    ece: float
    average_precision: float

    # At optimal threshold
    optimal_threshold: float
    f1: float
    precision: float
    recall: float

    # Confusion matrix
    confusion: dict[str, Any] = field(default_factory=dict)

    # Scorecard stats
    score_mean: float = 0.0
    score_std: float = 0.0
    score_min: int = 0
    score_max: int = 0

    # PSI (requires reference distribution)
    psi: float | None = None

    # Metadata
    n_samples: int = 0
    default_rate: float = 0.0

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "═" * 50,
            "  LoanGuard AI — Model Evaluation Report",
            "═" * 50,
            f"  Samples        : {self.n_samples:,}",
            f"  Default Rate   : {self.default_rate:.2%}",
            "─" * 50,
            f"  AUC-ROC        : {self.auc_roc:.4f}",
            f"  KS Statistic   : {self.ks_statistic:.4f}",
            f"  Gini Coeff     : {self.gini:.4f}",
            f"  Avg Precision  : {self.average_precision:.4f}",
            f"  Brier Score    : {self.brier_score:.4f}",
            f"  Calib ECE      : {self.ece:.4f}",
            "─" * 50,
            f"  Optimal Thresh : {self.optimal_threshold:.3f}",
            f"  F1 Score       : {self.f1:.4f}",
            f"  Precision      : {self.precision:.4f}",
            f"  Recall         : {self.recall:.4f}",
            "─" * 50,
            f"  Score Mean     : {self.score_mean:.1f}",
            f"  Score Std      : {self.score_std:.1f}",
            f"  Score Range    : [{self.score_min}, {self.score_max}]",
        ]
        if self.psi is not None:
            stability = (
                "Stable" if self.psi < 0.1
                else "Monitor" if self.psi < 0.2
                else "⚠ Unstable"
            )
            lines.append(f"  PSI            : {self.psi:.4f} ({stability})")
        if self.confusion:
            lines += [
                "─" * 50,
                f"  TN={self.confusion['tn']:,}  FP={self.confusion['fp']:,}",
                f"  FN={self.confusion['fn']:,}  TP={self.confusion['tp']:,}",
                f"  Business Cost  : {self.confusion['total_cost']:,.0f}",
            ]
        lines.append("═" * 50)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat dict (for MLflow logging)."""
        return {
            "auc_roc": self.auc_roc,
            "ks_statistic": self.ks_statistic,
            "gini": self.gini,
            "brier_score": self.brier_score,
            "ece": self.ece,
            "average_precision": self.average_precision,
            "optimal_threshold": self.optimal_threshold,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "psi": self.psi if self.psi is not None else -1.0,
            "n_samples": self.n_samples,
            "default_rate": self.default_rate,
            **{f"confusion_{k}": v for k, v in self.confusion.items()},
        }


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate_model(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    train_probs: np.ndarray | None = None,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    score_params: dict | None = None,
) -> EvaluationResults:
    """Run the full evaluation suite on a fitted pipeline.

    Args:
        pipeline: Fitted sklearn pipeline with predict_proba.
        X: Feature DataFrame.
        y: Binary target Series.
        train_probs: Training set probabilities (used to compute PSI).
            Pass None to skip PSI.
        cost_fn: Cost of a false negative.
        cost_fp: Cost of a false positive.
        score_params: Optional overrides for ``probability_to_score``.

    Returns:
        EvaluationResults with all computed metrics.
    """
    y_arr = np.array(y, dtype=int)
    y_prob = pipeline.predict_proba(X)[:, 1]

    auc = float(roc_auc_score(y_arr, y_prob))
    ks = compute_ks_statistic(y_arr, y_prob)
    gini = 2 * auc - 1
    brier = float(brier_score_loss(y_arr, y_prob))
    ece = compute_calibration_error(y_arr, y_prob)
    ap = float(average_precision_score(y_arr, y_prob))

    opt_thresh = find_optimal_threshold(y_arr, y_prob, cost_fn, cost_fp)
    y_pred = (y_prob >= opt_thresh).astype(int)
    f1 = float(f1_score(y_arr, y_pred, zero_division=0))
    prec = float(precision_score(y_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_arr, y_pred, zero_division=0))
    cm = cost_weighted_confusion_matrix(y_arr, y_pred, cost_fn, cost_fp)

    # Scorecard
    sp = score_params or {}
    scores = probability_to_score(y_prob, **sp)
    psi = compute_psi(train_probs, y_prob) if train_probs is not None else None

    results = EvaluationResults(
        auc_roc=auc,
        ks_statistic=ks,
        gini=gini,
        brier_score=brier,
        ece=ece,
        average_precision=ap,
        optimal_threshold=opt_thresh,
        f1=f1,
        precision=prec,
        recall=rec,
        confusion=cm,
        score_mean=float(np.mean(scores)),
        score_std=float(np.std(scores)),
        score_min=int(np.min(scores)),
        score_max=int(np.max(scores)),
        psi=psi,
        n_samples=len(y_arr),
        default_rate=float(y_arr.mean()),
    )

    logger.info(
        "Evaluation complete: AUC=%.4f | KS=%.4f | Gini=%.4f | F1=%.4f",
        auc, ks, gini, f1,
    )
    return results


def calibration_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute data for a reliability (calibration) diagram.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins.

    Returns:
        DataFrame with columns: bin_center, mean_predicted, fraction_positive, count.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_center": (lo + hi) / 2,
            "mean_predicted": float(y_prob[mask].mean()),
            "fraction_positive": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)
