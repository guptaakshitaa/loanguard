"""POST /explain — SHAP + LIME + counterfactual explanations."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

_PKG = Path(__file__).resolve().parents[2]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from api.model_registry import ModelRegistry, ModelNotLoadedError, get_registry
from api.schemas import (
    CounterfactualChange,
    ExplanationResponse,
    LoanApplication,
    RiskFactor,
)
from ml.evaluate import probability_to_score

router = APIRouter()


@router.post(
    "/explain",
    response_model=ExplanationResponse,
    summary="Prediction explanation",
    description=(
        "Returns SHAP values, LIME weights, top risk factors, "
        "and counterfactual suggestions for a loan application."
    ),
)
async def explain(
    application: LoanApplication,
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    request: Request,
) -> ExplanationResponse:
    """Generate full explainability payload for a loan application.

    Combines:
    - SHAP TreeExplainer for global feature attribution
    - LIME for local linear approximation
    - Greedy counterfactual search for actionable recourse
    """
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    from config.settings import get_settings
    cfg = get_settings()

    app_id = application.application_id or "EXPLAIN-REQUEST"

    # Build feature DataFrame
    data = application.model_dump()
    X = pd.DataFrame([data])

    # ── Base prediction ───────────────────────────────────────────────────────
    try:
        prob = float(registry.predict_proba(X)[0])
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    score = int(probability_to_score(
        prob,
        base_score=cfg.score_base_score,
        base_odds=cfg.score_base_odds,
        pdo=cfg.score_pdo,
    ))

    if prob <= cfg.threshold_approve:
        decision = "APPROVE"
    elif prob >= cfg.threshold_decline:
        decision = "DECLINE"
    else:
        decision = "REVIEW"

    # ── SHAP ──────────────────────────────────────────────────────────────────
    shap_values: dict[str, float] = {}
    top_factors: list[RiskFactor] = []
    base_value: float = 0.0

    shap_explainer = getattr(request.app.state, "shap_explainer", None)
    if shap_explainer is not None:
        try:
            shap_result = shap_explainer.explain(X, top_n=5)
            shap_values = shap_result["shap_values"]
            base_value = shap_result["base_value"]
            top_factors = [
                RiskFactor(
                    feature=f["feature"],
                    value=data.get(f["feature"], "N/A"),
                    shap_impact=f["shap_impact"],
                    direction=f["direction"],
                    rank=f["rank"],
                )
                for f in shap_result["top_factors"]
            ]
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("SHAP failed (non-fatal): %s", exc)
            # Fallback: return empty but don't crash
            shap_values = {}
            top_factors = []

    # ── LIME ──────────────────────────────────────────────────────────────────
    lime_explanation: dict[str, float] = {}
    lime_explainer = getattr(request.app.state, "lime_explainer", None)
    if lime_explainer is not None:
        try:
            lime_explanation = lime_explainer.explain(X, top_n=8)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("LIME failed (non-fatal): %s", exc)

    # ── Counterfactual ────────────────────────────────────────────────────────
    cf_changes: list[CounterfactualChange] = []
    cf_decision: str | None = None

    cf_generator = getattr(request.app.state, "cf_generator", None)
    if cf_generator is not None and decision != "APPROVE":
        try:
            cf_result = cf_generator.generate(X, target_decision="APPROVE")
            cf_decision = cf_result["counterfactual_decision"]
            for ch in cf_result["changes"]:
                cf_changes.append(CounterfactualChange(
                    feature=ch["feature"],
                    current_value=ch["current_value"],
                    suggested_value=ch["suggested_value"],
                    change_description=ch["change_description"],
                ))
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Counterfactual failed (non-fatal): %s", exc)

    return ExplanationResponse(
        application_id=app_id,
        decision=decision,
        default_probability=round(prob, 6),
        risk_score=score,
        shap_values=shap_values,
        top_risk_factors=top_factors,
        shap_base_value=base_value,
        lime_explanation=lime_explanation,
        counterfactual=cf_changes,
        counterfactual_decision=cf_decision,
        timestamp=datetime.now(timezone.utc),
    )
