"""LoanGuard AI — API test suite.

Tests all routes with valid inputs, edge cases, and error conditions.
Uses a mock model registry so no trained artifact is needed for CI.

Run:
    pytest tests/ -v
    pytest tests/ -v --cov=loanguard --cov-report=term-missing
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_APPLICATION: dict[str, Any] = {
    "age": 35,
    "annual_income": 75000.0,
    "employment_length_years": 5.0,
    "loan_amount": 15000.0,
    "interest_rate": 12.5,
    "loan_term_months": 36,
    "num_open_accounts": 8,
    "num_derog_records": 0,
    "num_credit_inquiries": 2,
    "credit_utilization_ratio": 0.35,
    "debt_to_income": 0.18,
    "months_since_last_delinq": None,
    "revolving_balance": 5000.0,
    "total_accounts": 15,
    "home_ownership": "RENT",
    "loan_purpose": "debt_consolidation",
    "grade": "B",
    "verification_status": "Verified",
}

HIGH_RISK_APPLICATION: dict[str, Any] = {
    **VALID_APPLICATION,
    "credit_utilization_ratio": 0.98,
    "debt_to_income": 0.75,
    "num_derog_records": 5,
    "grade": "F",
    "interest_rate": 28.0,
    "num_credit_inquiries": 15,
}

LOW_RISK_APPLICATION: dict[str, Any] = {
    **VALID_APPLICATION,
    "credit_utilization_ratio": 0.05,
    "debt_to_income": 0.08,
    "num_derog_records": 0,
    "grade": "A",
    "interest_rate": 6.0,
    "annual_income": 200000.0,
    "num_credit_inquiries": 0,
}


def _make_mock_registry(prob: float = 0.25) -> MagicMock:
    """Create a mock ModelRegistry that returns a fixed probability."""
    registry = MagicMock()
    registry.is_loaded = True
    registry.champion_version = "test_v1"
    registry.champion_path = "/mock/champion_latest.pkl"
    registry.predict_proba.return_value = np.array([prob])
    registry.uptime_seconds = 60.0
    registry.champion_meta = {"version": "test_v1"}
    return registry


@pytest.fixture
def mock_registry_approve():
    return _make_mock_registry(prob=0.10)  # → APPROVE


@pytest.fixture
def mock_registry_review():
    return _make_mock_registry(prob=0.50)  # → REVIEW


@pytest.fixture
def mock_registry_decline():
    return _make_mock_registry(prob=0.85)  # → DECLINE


@pytest.fixture
def mock_pred_logger():
    logger = MagicMock()
    logger.recent.return_value = []
    logger.total_count.return_value = 0
    logger.latency_percentiles.return_value = {"p50": 10.0, "p95": 25.0, "p99": 40.0}
    logger.decision_counts.return_value = {}
    logger.get_feature_vectors.return_value = []
    return logger


@pytest.fixture
def test_client(mock_registry_approve, mock_pred_logger):
    """FastAPI TestClient with mocked dependencies."""
    from fastapi.testclient import TestClient
    from api.model_registry import get_registry
    from api.main import create_app

    app = create_app()

    # Inject mocks into app state
    app.state.prediction_logger = mock_pred_logger
    app.state.shap_explainer = None
    app.state.lime_explainer = None
    app.state.cf_generator = None
    app.state.drift_detector = None

    app.dependency_overrides[get_registry] = lambda: mock_registry_approve

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# ── Schemas ───────────────────────────────────────────────────────────────────

class TestSchemas:

    def test_valid_application_parses(self):
        from api.schemas import LoanApplication
        app = LoanApplication(**VALID_APPLICATION)
        assert app.age == 35
        assert app.grade == "B"

    def test_missing_required_field_raises(self):
        from pydantic import ValidationError
        from api.schemas import LoanApplication
        data = {k: v for k, v in VALID_APPLICATION.items() if k != "annual_income"}
        with pytest.raises(ValidationError):
            LoanApplication(**data)

    def test_age_below_minimum_raises(self):
        from pydantic import ValidationError
        from api.schemas import LoanApplication
        with pytest.raises(ValidationError):
            LoanApplication(**{**VALID_APPLICATION, "age": 15})

    def test_invalid_grade_raises(self):
        from pydantic import ValidationError
        from api.schemas import LoanApplication
        with pytest.raises(ValidationError):
            LoanApplication(**{**VALID_APPLICATION, "grade": "Z"})

    def test_credit_utilization_out_of_range_raises(self):
        from pydantic import ValidationError
        from api.schemas import LoanApplication
        with pytest.raises(ValidationError):
            LoanApplication(**{**VALID_APPLICATION, "credit_utilization_ratio": 1.5})

    def test_optional_fields_accept_none(self):
        from api.schemas import LoanApplication
        app = LoanApplication(**{
            **VALID_APPLICATION,
            "employment_length_years": None,
            "months_since_last_delinq": None,
            "revolving_balance": None,
            "num_derog_records": None,
        })
        assert app.employment_length_years is None


# ── /predict ──────────────────────────────────────────────────────────────────

class TestPredict:

    def test_approve_decision(self, test_client):
        resp = test_client.post("/predict", json=LOW_RISK_APPLICATION)
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "APPROVE"
        assert 300 <= data["risk_score"] <= 850
        assert 0.0 <= data["default_probability"] <= 1.0
        assert data["processing_time_ms"] > 0

    def test_response_has_all_fields(self, test_client):
        resp = test_client.post("/predict", json=VALID_APPLICATION)
        assert resp.status_code == 200
        data = resp.json()
        required = {
            "application_id", "default_probability", "risk_score",
            "decision", "confidence", "model_version", "processing_time_ms", "timestamp"
        }
        assert required.issubset(data.keys())

    def test_application_id_echoed(self, test_client):
        payload = {**VALID_APPLICATION, "application_id": "TEST-001"}
        resp = test_client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["application_id"] == "TEST-001"

    def test_auto_generated_id_when_missing(self, test_client):
        resp = test_client.post("/predict", json=VALID_APPLICATION)
        assert resp.status_code == 200
        assert resp.json()["application_id"].startswith("REQ-")

    def test_score_in_valid_range(self, test_client):
        for payload in [LOW_RISK_APPLICATION, VALID_APPLICATION, HIGH_RISK_APPLICATION]:
            resp = test_client.post("/predict", json=payload)
            if resp.status_code == 200:
                score = resp.json()["risk_score"]
                assert 300 <= score <= 850, f"Score {score} out of range"

    def test_missing_required_field_returns_422(self, test_client):
        payload = {k: v for k, v in VALID_APPLICATION.items() if k != "annual_income"}
        resp = test_client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_invalid_grade_returns_422(self, test_client):
        resp = test_client.post("/predict", json={**VALID_APPLICATION, "grade": "X"})
        assert resp.status_code == 422

    def test_null_optional_fields_accepted(self, test_client):
        payload = {
            **VALID_APPLICATION,
            "employment_length_years": None,
            "months_since_last_delinq": None,
            "revolving_balance": None,
            "num_derog_records": None,
        }
        resp = test_client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_model_not_loaded_returns_503(self, mock_pred_logger):
        from fastapi.testclient import TestClient
        from api.model_registry import get_registry
        from api.main import create_app

        unloaded_registry = MagicMock()
        unloaded_registry.is_loaded = False

        app = create_app()
        app.state.prediction_logger = mock_pred_logger
        app.state.shap_explainer = None
        app.state.lime_explainer = None
        app.state.cf_generator = None
        app.state.drift_detector = None
        app.dependency_overrides[get_registry] = lambda: unloaded_registry

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/predict", json=VALID_APPLICATION)
            assert resp.status_code == 503

    def test_decline_decision(self, mock_pred_logger):
        from fastapi.testclient import TestClient
        from api.model_registry import get_registry
        from api.main import create_app

        decline_registry = _make_mock_registry(prob=0.85)
        app = create_app()
        app.state.prediction_logger = mock_pred_logger
        app.state.shap_explainer = None
        app.state.lime_explainer = None
        app.state.cf_generator = None
        app.state.drift_detector = None
        app.dependency_overrides[get_registry] = lambda: decline_registry

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/predict", json=HIGH_RISK_APPLICATION)
            assert resp.status_code == 200
            assert resp.json()["decision"] == "DECLINE"

    def test_review_decision(self, mock_pred_logger):
        from fastapi.testclient import TestClient
        from api.model_registry import get_registry
        from api.main import create_app

        review_registry = _make_mock_registry(prob=0.50)
        app = create_app()
        app.state.prediction_logger = mock_pred_logger
        app.state.shap_explainer = None
        app.state.lime_explainer = None
        app.state.cf_generator = None
        app.state.drift_detector = None
        app.dependency_overrides[get_registry] = lambda: review_registry

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/predict", json=VALID_APPLICATION)
            assert resp.status_code == 200
            assert resp.json()["decision"] == "REVIEW"


# ── /explain ──────────────────────────────────────────────────────────────────

class TestExplain:

    def test_explain_returns_200(self, test_client):
        resp = test_client.post("/explain", json=VALID_APPLICATION)
        assert resp.status_code == 200

    def test_explain_has_required_fields(self, test_client):
        resp = test_client.post("/explain", json=VALID_APPLICATION)
        data = resp.json()
        required = {
            "application_id", "decision", "default_probability", "risk_score",
            "shap_values", "top_risk_factors", "lime_explanation",
            "counterfactual", "timestamp"
        }
        assert required.issubset(data.keys())

    def test_explain_shap_empty_without_explainer(self, test_client):
        # test_client has shap_explainer=None → shap_values should be {}
        resp = test_client.post("/explain", json=VALID_APPLICATION)
        assert resp.status_code == 200
        assert resp.json()["shap_values"] == {}

    def test_explain_missing_field_returns_422(self, test_client):
        payload = {k: v for k, v in VALID_APPLICATION.items() if k != "loan_amount"}
        resp = test_client.post("/explain", json=payload)
        assert resp.status_code == 422


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_returns_200(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200

    def test_health_status_healthy(self, test_client):
        resp = test_client.get("/health")
        assert resp.json()["status"] == "healthy"

    def test_health_contains_model_info(self, test_client):
        resp = test_client.get("/health")
        data = resp.json()
        assert "model" in data
        assert data["model"]["loaded"] is True
        assert data["model"]["version"] == "test_v1"


# ── /metrics ──────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_metrics_returns_200(self, test_client):
        resp = test_client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_has_rate_fields(self, test_client):
        resp = test_client.get("/metrics")
        data = resp.json()
        assert "approval_rate" in data
        assert "decline_rate" in data
        assert "total_predictions" in data
        assert "p95_latency_ms" in data


# ── /drift ────────────────────────────────────────────────────────────────────

class TestDrift:

    def test_drift_no_data_returns_422(self, test_client):
        # mock_pred_logger returns [] for get_feature_vectors
        resp = test_client.get("/drift")
        assert resp.status_code in {422, 503}

    def test_drift_no_detector_returns_503(self, test_client):
        # test_client has drift_detector=None
        resp = test_client.get("/drift")
        assert resp.status_code == 503


# ── Scorecard ─────────────────────────────────────────────────────────────────

class TestScorecard:

    def test_score_range(self):
        from ml.evaluate import probability_to_score
        for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            score = probability_to_score(p)
            assert 300 <= score <= 850, f"Score {score} out of range for p={p}"

    def test_high_risk_lower_score(self):
        from ml.evaluate import probability_to_score
        high_risk = probability_to_score(0.9)
        low_risk  = probability_to_score(0.1)
        assert high_risk < low_risk

    def test_score_array_input(self):
        from ml.evaluate import probability_to_score
        scores = probability_to_score(np.array([0.1, 0.5, 0.9]))
        assert len(scores) == 3
        assert all(300 <= s <= 850 for s in scores)


# ── Drift detector ────────────────────────────────────────────────────────────

class TestDriftDetector:

    def test_stable_psi(self):
        from monitoring.drift_detector import DriftDetector
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(5, 2, 500)})
        cur = pd.DataFrame({"x": rng.normal(0, 1, 100), "y": rng.normal(5, 2, 100)})
        detector = DriftDetector(ref, numeric_cols=["x", "y"], categorical_cols=[])
        report = detector.compute(cur)
        assert report.overall_status in {"stable", "monitor"}

    def test_shifted_distribution_triggers_alert(self):
        from monitoring.drift_detector import DriftDetector
        rng = np.random.default_rng(0)
        ref = pd.DataFrame({"x": rng.normal(0, 1, 500)})
        cur = pd.DataFrame({"x": rng.normal(5, 1, 100)})  # large shift
        detector = DriftDetector(ref, numeric_cols=["x"], categorical_cols=[])
        report = detector.compute(cur)
        assert report.overall_status == "alert"


# ── Prediction logger ─────────────────────────────────────────────────────────

class TestPredictionLogger:

    def test_log_and_retrieve(self, tmp_path):
        from db.prediction_log import PredictionLogger
        pl = PredictionLogger(tmp_path / "test.db")
        pl.log(
            application_id="TEST-001",
            features={"age": 35, "income": 50000},
            default_prob=0.12,
            risk_score=580,
            decision="APPROVE",
            confidence=0.8,
            model_version="v1",
            latency_ms=12.5,
        )
        rows = pl.recent(hours=1)
        assert len(rows) == 1
        assert rows[0]["application_id"] == "TEST-001"
        assert rows[0]["decision"] == "APPROVE"

    def test_total_count(self, tmp_path):
        from db.prediction_log import PredictionLogger
        pl = PredictionLogger(tmp_path / "test.db")
        assert pl.total_count() == 0
        pl.log("A1", {}, 0.1, 600, "APPROVE", 0.9, "v1", 10.0)
        pl.log("A2", {}, 0.8, 400, "DECLINE", 0.8, "v1", 11.0)
        assert pl.total_count() == 2

    def test_decision_counts(self, tmp_path):
        from db.prediction_log import PredictionLogger
        pl = PredictionLogger(tmp_path / "test.db")
        pl.log("A1", {}, 0.1, 600, "APPROVE", 0.9, "v1", 10.0)
        pl.log("A2", {}, 0.8, 400, "DECLINE", 0.8, "v1", 10.0)
        pl.log("A3", {}, 0.5, 500, "APPROVE", 0.5, "v1", 10.0)
        counts = pl.decision_counts(hours=1)
        assert counts.get("APPROVE", 0) == 2
        assert counts.get("DECLINE", 0) == 1
