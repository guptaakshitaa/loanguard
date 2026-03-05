"""Pydantic request and response schemas for LoanGuard AI API.

Every field includes validation constraints and descriptions
so the auto-generated /docs page is self-documenting.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── Request ───────────────────────────────────────────────────────────────────

class LoanApplication(BaseModel):
    """Incoming loan application for credit risk assessment."""

    # Identity (not used by model — for audit trail only)
    application_id: Optional[str] = Field(
        default=None,
        description="Optional caller-supplied ID. Auto-generated if omitted.",
        examples=["APP0001234"],
    )

    # Numeric features
    age: float = Field(..., ge=18, le=100, description="Applicant age in years")
    annual_income: float = Field(..., gt=0, le=10_000_000, description="Gross annual income (USD)")
    employment_length_years: Optional[float] = Field(
        default=None, ge=0, le=60, description="Years at current employer (null if unemployed)"
    )
    loan_amount: float = Field(..., gt=0, le=500_000, description="Requested loan amount (USD)")
    interest_rate: float = Field(..., gt=0, le=40, description="Annual interest rate (%)")
    loan_term_months: int = Field(..., ge=12, le=84, description="Loan duration in months")
    num_open_accounts: int = Field(..., ge=0, le=100, description="Number of open credit accounts")
    num_derog_records: Optional[int] = Field(
        default=None, ge=0, le=50, description="Number of derogatory records"
    )
    num_credit_inquiries: int = Field(..., ge=0, le=50, description="Hard credit inquiries in last 12 months")
    credit_utilization_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Revolving credit utilization (0–1)"
    )
    debt_to_income: float = Field(..., ge=0.0, le=2.0, description="Monthly debt / monthly income")
    months_since_last_delinq: Optional[float] = Field(
        default=None, ge=0, le=300, description="Months since last delinquency (null if never)"
    )
    revolving_balance: Optional[float] = Field(
        default=None, ge=0, description="Total revolving balance (USD)"
    )
    total_accounts: int = Field(..., ge=1, le=200, description="Total number of credit accounts ever")

    # Categorical features
    home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"] = Field(
        ..., description="Home ownership status"
    )
    loan_purpose: Literal[
        "debt_consolidation", "credit_card", "home_improvement",
        "other", "major_purchase", "medical"
    ] = Field(..., description="Purpose of the loan")
    grade: Literal["A", "B", "C", "D", "E", "F"] = Field(
        ..., description="Lender-assigned credit grade"
    )
    verification_status: Literal["Not Verified", "Source Verified", "Verified"] = Field(
        ..., description="Income verification status"
    )

    @field_validator("debt_to_income")
    @classmethod
    def dti_reasonable(cls, v: float) -> float:
        if v > 1.0:
            # Allow but flag — very high DTI is a valid (risky) input
            pass
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "annual_income": 75000,
                "employment_length_years": 5,
                "loan_amount": 15000,
                "interest_rate": 12.5,
                "loan_term_months": 36,
                "num_open_accounts": 8,
                "num_derog_records": 0,
                "num_credit_inquiries": 2,
                "credit_utilization_ratio": 0.35,
                "debt_to_income": 0.18,
                "months_since_last_delinq": None,
                "revolving_balance": 5000,
                "total_accounts": 15,
                "home_ownership": "RENT",
                "loan_purpose": "debt_consolidation",
                "grade": "B",
                "verification_status": "Verified",
            }
        }
    }


# ── Predict response ──────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Credit risk assessment result."""

    application_id: str = Field(..., description="Unique prediction ID (echoed or generated)")
    default_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of default (0–1)")
    risk_score: int = Field(..., ge=300, le=850, description="Credit score (300=high risk, 850=low risk)")
    decision: Literal["APPROVE", "REVIEW", "DECLINE"] = Field(..., description="Lending decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence (distance from decision boundary)")
    model_version: str = Field(..., description="Model artifact version used")
    processing_time_ms: float = Field(..., description="End-to-end latency in milliseconds")
    timestamp: datetime = Field(..., description="UTC timestamp of prediction")


# ── Explain response ──────────────────────────────────────────────────────────

class RiskFactor(BaseModel):
    """A single feature's contribution to the prediction."""
    feature: str
    value: Any
    shap_impact: float = Field(..., description="SHAP value — positive increases default risk")
    direction: Literal["increases_risk", "decreases_risk"]
    rank: int = Field(..., description="Rank by absolute SHAP impact (1 = most influential)")


class CounterfactualChange(BaseModel):
    """A suggested change that would flip the decision."""
    feature: str
    current_value: Any
    suggested_value: Any
    change_description: str


class ExplanationResponse(BaseModel):
    """Full explainability payload for a prediction."""

    application_id: str
    decision: Literal["APPROVE", "REVIEW", "DECLINE"]
    default_probability: float
    risk_score: int

    # SHAP
    shap_values: dict[str, float] = Field(..., description="Feature → SHAP value mapping")
    top_risk_factors: list[RiskFactor] = Field(..., description="Top 5 features by SHAP magnitude")
    shap_base_value: float = Field(..., description="SHAP expected value (model baseline)")

    # LIME
    lime_explanation: dict[str, float] = Field(
        ..., description="LIME feature → local weight mapping"
    )

    # Counterfactual
    counterfactual: list[CounterfactualChange] = Field(
        ..., description="Changes that would flip the decision"
    )
    counterfactual_decision: Optional[str] = Field(
        default=None, description="Decision after applying counterfactual changes"
    )

    timestamp: datetime


# ── Health response ───────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    model_name: str
    version: str
    loaded: bool
    artifact_path: str


class HealthResponse(BaseModel):
    """Service health status."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    model: ModelInfo
    uptime_seconds: float
    timestamp: datetime


# ── Drift response ────────────────────────────────────────────────────────────

class FeatureDrift(BaseModel):
    feature: str
    psi: float
    status: Literal["stable", "monitor", "alert"]


class DriftResponse(BaseModel):
    """Feature drift report based on PSI scores."""
    overall_status: Literal["stable", "monitor", "alert"]
    n_features_checked: int
    n_alerts: int
    features: list[FeatureDrift]
    reference_period: str
    current_period: str
    timestamp: datetime


# ── Metrics response ──────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    """Prometheus-style operational metrics."""
    total_predictions: int
    predictions_last_hour: int
    approval_rate: float
    review_rate: float
    decline_rate: float
    avg_default_probability: float
    avg_risk_score: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    timestamp: datetime


# ── Retrain response ──────────────────────────────────────────────────────────

class RetrainResponse(BaseModel):
    """Response to an async retraining trigger."""
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    message: str
    triggered_at: datetime


# ── Error response ────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error envelope."""
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime
