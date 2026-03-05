"""Counterfactual explanation generator.

Answers: "What would need to change for this application to be APPROVED?"
Uses a gradient-based greedy search rather than DiCE (no heavy dependency)
while still being interpretable and actionable.

Usage:
    from explainability.counterfactual import CounterfactualGenerator
    gen = CounterfactualGenerator(pipeline, feature_bounds)
    changes = gen.generate(X_row, target_decision="APPROVE")
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Human-readable descriptions for each feature change direction
_FEATURE_DESCRIPTIONS = {
    "debt_to_income":          ("lower", "Reduce your monthly debt-to-income ratio"),
    "credit_utilization_ratio":("lower", "Pay down revolving credit balances"),
    "interest_rate":           ("lower", "Seek a lower interest rate loan product"),
    "num_derog_records":       ("lower", "Resolve derogatory records on credit report"),
    "num_credit_inquiries":    ("lower", "Avoid applying for new credit in the next 12 months"),
    "annual_income":           ("higher","Increase verifiable annual income"),
    "employment_length_years": ("higher","Maintain current employment for longer"),
    "months_since_last_delinq":("higher","Maintain clean payment history over time"),
    "loan_amount":             ("lower", "Request a smaller loan amount"),
    "revolving_balance":       ("lower", "Pay down revolving credit balance"),
}


class CounterfactualGenerator:
    """Greedy counterfactual search for actionable recourse.

    Iteratively nudges the most impactful mutable features toward
    lower default probability until the decision threshold is crossed.

    Args:
        pipeline: Fitted calibrated pipeline.
        threshold_approve: Probability below which decision is APPROVE.
        max_changes: Maximum number of feature changes to suggest.
    """

    def __init__(
        self,
        pipeline: Any,
        threshold_approve: float = 0.30,
        threshold_decline: float = 0.70,
        max_changes: int = 4,
    ) -> None:
        self._pipeline = pipeline
        self._threshold_approve = threshold_approve
        self._threshold_decline = threshold_decline
        self._max_changes = max_changes

    def _predict(self, X: pd.DataFrame) -> float:
        """Return default probability for a single row."""
        return float(self._pipeline.predict_proba(X)[:, 1][0])

    def _decision(self, prob: float) -> str:
        if prob <= self._threshold_approve:
            return "APPROVE"
        elif prob >= self._threshold_decline:
            return "DECLINE"
        return "REVIEW"

    def generate(
        self,
        X_row: pd.DataFrame,
        target_decision: str = "APPROVE",
    ) -> dict[str, Any]:
        """Generate counterfactual changes to reach target_decision.

        Args:
            X_row: Single-row DataFrame with raw features.
            target_decision: Desired outcome ('APPROVE' or 'REVIEW').

        Returns:
            Dict with keys:
                - changes: list of {feature, current_value, suggested_value, description}
                - counterfactual_decision: decision after applying changes
                - counterfactual_probability: probability after changes
                - achievable: whether target was reached within max_changes
        """
        X_cf = X_row.copy().drop(columns=["application_id"], errors="ignore")
        current_prob = self._predict(X_cf)
        current_decision = self._decision(current_prob)

        if current_decision == target_decision:
            return {
                "changes": [],
                "counterfactual_decision": current_decision,
                "counterfactual_probability": round(current_prob, 4),
                "achievable": True,
                "message": "Application already meets target decision.",
            }

        # Target probability: just below approve threshold
        target_prob = self._threshold_approve * 0.9

        changes = []
        mutable = [f for f in _FEATURE_DESCRIPTIONS if f in X_cf.columns]

        for _ in range(self._max_changes):
            best_feat = None
            best_delta = 0.0
            best_val = None

            for feat in mutable:
                if feat in [c["feature"] for c in changes]:
                    continue  # already changed

                direction, _ = _FEATURE_DESCRIPTIONS[feat]
                current_val = float(X_cf[feat].iloc[0]) if pd.notna(X_cf[feat].iloc[0]) else 0.0

                # Compute a candidate improvement
                if direction == "lower":
                    candidate_val = current_val * 0.75  # 25% reduction
                else:
                    candidate_val = current_val * 1.30  # 30% increase

                X_test = X_cf.copy()
                X_test[feat] = candidate_val

                try:
                    new_prob = self._predict(X_test)
                except Exception:
                    continue

                delta = current_prob - new_prob  # positive = improvement
                if delta > best_delta:
                    best_delta = delta
                    best_feat = feat
                    best_val = candidate_val

            if best_feat is None or best_delta < 0.001:
                break  # No more useful changes

            # Apply best change
            orig_val = float(X_cf[best_feat].iloc[0]) if pd.notna(X_cf[best_feat].iloc[0]) else 0.0
            X_cf[best_feat] = best_val
            current_prob = self._predict(X_cf)

            _, desc = _FEATURE_DESCRIPTIONS[best_feat]
            changes.append({
                "feature": best_feat,
                "current_value": round(orig_val, 4),
                "suggested_value": round(float(best_val), 4),
                "change_description": desc,
            })

            if self._decision(current_prob) == target_decision:
                break

        final_decision = self._decision(current_prob)
        return {
            "changes": changes,
            "counterfactual_decision": final_decision,
            "counterfactual_probability": round(current_prob, 4),
            "achievable": final_decision == target_decision,
            "message": (
                f"Applying {len(changes)} change(s) would move decision to {final_decision}."
                if changes else "No actionable changes found."
            ),
        }
