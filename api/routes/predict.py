"""POST /predict — credit risk prediction endpoint."""

from __future__ import annotations

import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

_PKG = Path(__file__).resolve().parents[2]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from api.model_registry import ModelRegistry, ModelNotLoadedError, get_registry
from api.schemas import LoanApplication, PredictionResponse
from db.prediction_log import PredictionLogger
from ml.evaluate import probability_to_score

router = APIRouter()


def _get_logger(request: Request) -> PredictionLogger:
    return request.app.state.prediction_logger


def _make_dataframe(app: LoanApplication) -> pd.DataFrame:
    """Convert Pydantic model to a single-row DataFrame."""
    data = app.model_dump()
    return pd.DataFrame([data])


def _confidence(prob: float, threshold_approve: float, threshold_decline: float) -> float:
    """Distance from the nearest decision boundary (0–1)."""
    if prob <= threshold_approve:
        return round(1.0 - (prob / threshold_approve), 4)
    elif prob >= threshold_decline:
        return round((prob - threshold_decline) / (1.0 - threshold_decline), 4)
    else:
        mid = (threshold_approve + threshold_decline) / 2
        span = (threshold_decline - threshold_approve) / 2
        return round(abs(prob - mid) / span, 4)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Credit risk prediction",
    description="Score a loan application and return probability, credit score, and decision.",
)
async def predict(
    application: LoanApplication,
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    request: Request,
) -> PredictionResponse:
    """Run credit risk assessment on a loan application.

    Returns default probability, 300–850 risk score, and APPROVE/REVIEW/DECLINE decision.
    Every prediction is logged to SQLite for audit and drift detection.
    """
    t_start = time.perf_counter()

    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded — service starting up.")

    app_id = application.application_id or f"REQ-{uuid.uuid4().hex[:12].upper()}"
    application.application_id = app_id

    try:
        X = _make_dataframe(application)
        prob = float(registry.predict_proba(X)[0])
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    from config.settings import get_settings
    cfg = get_settings()

    score = int(probability_to_score(
        prob,
        base_score=cfg.score_base_score,
        base_odds=cfg.score_base_odds,
        pdo=cfg.score_pdo,
        score_min=cfg.score_min,
        score_max=cfg.score_max,
    ))

    if prob <= cfg.threshold_approve:
        decision = "APPROVE"
    elif prob >= cfg.threshold_decline:
        decision = "DECLINE"
    else:
        decision = "REVIEW"

    latency_ms = (time.perf_counter() - t_start) * 1000
    confidence = _confidence(prob, cfg.threshold_approve, cfg.threshold_decline)

    # Audit log
    try:
        pred_logger: PredictionLogger = _get_logger(request)
        features = application.model_dump(exclude={"application_id"})
        pred_logger.log(
            application_id=app_id,
            features=features,
            default_prob=prob,
            risk_score=score,
            decision=decision,
            confidence=confidence,
            model_version=registry.champion_version,
            latency_ms=latency_ms,
        )
    except Exception as log_exc:
        # Log failure must never break the prediction response
        import logging
        logging.getLogger(__name__).warning("Audit log failed: %s", log_exc)

    return PredictionResponse(
        application_id=app_id,
        default_probability=round(prob, 6),
        risk_score=score,
        decision=decision,
        confidence=confidence,
        model_version=registry.champion_version,
        processing_time_ms=round(latency_ms, 2),
        timestamp=datetime.now(timezone.utc),
    )
