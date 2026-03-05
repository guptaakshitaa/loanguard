"""LoanGuard AI — FastAPI application entry point.

Startup sequence:
    1. Load config
    2. Load model artifact into ModelRegistry
    3. Initialise SHAP/LIME/counterfactual explainers
    4. Initialise drift detector with training reference
    5. Open prediction audit logger (SQLite)
    6. Register all routes

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Or via Docker:
    docker-compose up
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import pandas as pd
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Path fix so imports work regardless of working directory ──────────────────
_PKG = Path(__file__).resolve().parent.parent   # loanguard/
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from api.model_registry import get_registry
from api.routes.explain import router as explain_router
from api.routes.health import router as health_router
from api.routes.predict import router as predict_router
from config.settings import get_settings
from db.prediction_log import PredictionLogger
from monitoring.drift_detector import DriftDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("loanguard.api")


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise all heavy resources on startup, clean up on shutdown."""
    cfg = get_settings()
    logger.info("LoanGuard AI starting up (env=%s)", cfg.environment)

    # 1. Load model
    registry = get_registry()
    try:
        registry.load(cfg.model_dir)
        logger.info("Model loaded: version=%s", registry.champion_version)
    except FileNotFoundError as exc:
        logger.error("Could not load model: %s", exc)
        logger.warning("API will start in degraded mode — /predict will return 503")

    # 2. Prediction logger
    db_path = Path("db/predictions.db")
    app.state.prediction_logger = PredictionLogger(db_path)
    logger.info("Prediction logger ready: %s", db_path)

    # 3. Drift detector (needs reference data — load from training parquet if available)
    app.state.drift_detector = None
    ref_path = Path(cfg.data_dir) / cfg.dataset_filename
    if ref_path.exists():
        try:
            from data.loader import load_dataset, NUMERIC_FEATURES, CATEGORICAL_FEATURES
            X_ref, _, _ = load_dataset(ref_path, drop_id=True)
            app.state.drift_detector = DriftDetector(
                X_ref,
                numeric_cols=NUMERIC_FEATURES,
                categorical_cols=CATEGORICAL_FEATURES,
            )
            logger.info("Drift detector ready: %d reference rows", len(X_ref))
        except Exception as exc:
            logger.warning("Drift detector init failed (non-fatal): %s", exc)

    # 4. SHAP explainer
    app.state.shap_explainer = None
    if registry.is_loaded:
        try:
            from explainability.shap_explainer import SHAPExplainer
            app.state.shap_explainer = SHAPExplainer(registry.champion)
            logger.info("SHAP explainer ready")
        except Exception as exc:
            logger.warning("SHAP init failed (non-fatal): %s", exc)

    # 5. LIME explainer (needs a sample of training data)
    app.state.lime_explainer = None
    if registry.is_loaded and ref_path.exists():
        try:
            from data.loader import load_dataset
            from explainability.lime_explainer import LIMEExplainer
            X_ref_lime, _, _ = load_dataset(ref_path, drop_id=False)
            X_sample = X_ref_lime.sample(min(500, len(X_ref_lime)), random_state=42)
            app.state.lime_explainer = LIMEExplainer(registry.champion, X_sample)
            logger.info("LIME explainer ready")
        except Exception as exc:
            logger.warning("LIME init failed (non-fatal): %s", exc)

    # 6. Counterfactual generator
    app.state.cf_generator = None
    if registry.is_loaded:
        try:
            from explainability.counterfactual import CounterfactualGenerator
            app.state.cf_generator = CounterfactualGenerator(
                registry.champion,
                threshold_approve=cfg.threshold_approve,
                threshold_decline=cfg.threshold_decline,
            )
            logger.info("Counterfactual generator ready")
        except Exception as exc:
            logger.warning("Counterfactual init failed (non-fatal): %s", exc)

    logger.info("LoanGuard AI startup complete — serving on port %d", cfg.port)
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("LoanGuard AI shutting down")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title="LoanGuard AI",
        description=(
            "Production credit risk assessment API.\n\n"
            "Predicts default probability, converts to a 300–850 credit score, "
            "and returns APPROVE / REVIEW / DECLINE with full SHAP + LIME explanations."
        ),
        version=cfg.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if cfg.environment != "production" else [],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Request ID + latency middleware ───────────────────────────────────────
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: object) -> Response:
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        t0 = time.perf_counter()
        response: Response = await call_next(request)  # type: ignore[operator]
        latency_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
        return response

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if cfg.debug else "Contact support.",
            },
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(predict_router, tags=["Prediction"])
    app.include_router(explain_router, tags=["Explainability"])
    app.include_router(health_router,  tags=["Operations"])

    # Root redirect
    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {"service": "LoanGuard AI", "docs": "/docs", "health": "/health"}

    return app


app = create_app()
