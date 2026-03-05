"""GET /health, GET /metrics, GET /drift, POST /retrain endpoints."""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

_PKG = Path(__file__).resolve().parents[2]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from api.model_registry import ModelRegistry, get_registry
from api.schemas import (
    DriftResponse,
    FeatureDrift,
    HealthResponse,
    MetricsResponse,
    ModelInfo,
    RetrainResponse,
)

router = APIRouter()


# ── /health ───────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> HealthResponse:
    """Return service health status and loaded model metadata."""
    from config.settings import get_settings
    cfg = get_settings()

    status = "healthy" if registry.is_loaded else "unhealthy"

    return HealthResponse(
        status=status,
        version=cfg.app_version,
        model=ModelInfo(
            model_name=cfg.champion_model_name,
            version=registry.champion_version,
            loaded=registry.is_loaded,
            artifact_path=registry.champion_path,
        ),
        uptime_seconds=round(registry.uptime_seconds, 1),
        timestamp=datetime.now(timezone.utc),
    )


# ── /metrics ──────────────────────────────────────────────────────────────────

@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Operational metrics",
)
async def metrics(request: Request) -> MetricsResponse:
    """Return Prometheus-style operational metrics from the prediction log."""
    pred_logger = request.app.state.prediction_logger

    recent_24h = pred_logger.recent(hours=24)
    recent_1h  = pred_logger.recent(hours=1)
    total      = pred_logger.total_count()
    latency_p  = pred_logger.latency_percentiles(hours=1)

    n = len(recent_24h)
    decisions  = [r["decision"] for r in recent_24h]
    probs      = [r["default_prob"] for r in recent_24h]
    scores     = [r["risk_score"] for r in recent_24h]
    latencies  = [r["latency_ms"] for r in recent_24h]

    def rate(d: str) -> float:
        return round(decisions.count(d) / n, 4) if n > 0 else 0.0

    return MetricsResponse(
        total_predictions=total,
        predictions_last_hour=len(recent_1h),
        approval_rate=rate("APPROVE"),
        review_rate=rate("REVIEW"),
        decline_rate=rate("DECLINE"),
        avg_default_probability=round(sum(probs) / n, 4) if n > 0 else 0.0,
        avg_risk_score=round(sum(scores) / n, 1) if n > 0 else 0.0,
        avg_latency_ms=round(sum(latencies) / n, 2) if n > 0 else 0.0,
        p95_latency_ms=latency_p["p95"],
        p99_latency_ms=latency_p["p99"],
        timestamp=datetime.now(timezone.utc),
    )


# ── /drift ────────────────────────────────────────────────────────────────────

@router.get(
    "/drift",
    response_model=DriftResponse,
    summary="Feature drift report",
)
async def drift(request: Request) -> DriftResponse:
    """Compute PSI drift scores for all features using recent predictions."""
    import pandas as pd

    drift_detector = getattr(request.app.state, "drift_detector", None)
    if drift_detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialised.")

    pred_logger = request.app.state.prediction_logger
    recent_features = pred_logger.get_feature_vectors(hours=168)  # 7 days

    if len(recent_features) < 10:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough predictions for drift analysis ({len(recent_features)} < 10).",
        )

    current_df = pd.DataFrame(recent_features)
    report = drift_detector.compute(
        current_df,
        reference_period="training",
        current_period="last_7_days",
    )

    return DriftResponse(
        overall_status=report.overall_status,
        n_features_checked=report.n_features_checked,
        n_alerts=report.n_alerts,
        features=[
            FeatureDrift(feature=f.feature, psi=round(f.psi, 4), status=f.status)
            for f in report.features
        ],
        reference_period=report.reference_period,
        current_period=report.current_period,
        timestamp=report.timestamp,
    )


# ── /retrain ──────────────────────────────────────────────────────────────────

@router.post(
    "/retrain",
    response_model=RetrainResponse,
    summary="Trigger async model retraining",
)
async def retrain(request: Request) -> RetrainResponse:
    """Queue an async retraining job.

    Retrains in the background using the prediction log as additional
    training data. Returns immediately with a job ID to poll status.
    """
    job_id = f"RETRAIN-{uuid.uuid4().hex[:8].upper()}"

    async def _run_retrain() -> None:
        """Background retraining task."""
        import logging
        log = logging.getLogger("loanguard.retrain")
        log.info("Retraining job %s started", job_id)
        try:
            # Import here to avoid circular imports at module load time
            from ml.train import train as run_train
            # Run synchronously in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: run_train(n_trials=20))
            log.info("Retraining job %s completed", job_id)
            # Reload champion
            registry = get_registry()
            from config.settings import get_settings
            cfg = get_settings()
            registry.load(cfg.model_dir)
            log.info("Model reloaded after retraining")
        except Exception as exc:
            log.error("Retraining job %s failed: %s", job_id, exc)

    asyncio.create_task(_run_retrain())

    return RetrainResponse(
        job_id=job_id,
        status="queued",
        message="Retraining job queued. Model will be hot-reloaded on completion.",
        triggered_at=datetime.now(timezone.utc),
    )
