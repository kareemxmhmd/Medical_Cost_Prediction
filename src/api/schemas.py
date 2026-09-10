from typing import Any, Literal
from pydantic import BaseModel, Field


class ApplicantProfile(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body mass index (BMI)")
    children: int = Field(..., ge=0, le=20, description="Number of dependents / children")
    smoker: Literal["yes", "no"] = Field(..., description="Tobacco smoking status")
    region: Literal["northeast", "northwest", "southeast", "southwest"] = Field(
        ..., description="US Census geographic region"
    )


class ReasonCode(BaseModel):
    feature: str
    impact: float
    direction: Literal["increases cost", "decreases cost"]


class PredictionResponse(BaseModel):
    predicted_cost: float
    interval_80: tuple[float, float]
    premium_tier: Literal["Standard", "Loaded", "Refer to Underwriter"]
    reason_codes: list[ReasonCode]
    model_version: str
    decision_timestamp: str


class BatchPredictionRequest(BaseModel):
    applicants: list[ApplicantProfile] = Field(..., max_length=100, description="Batch of applicant profiles (max 100)")


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class DriftRequest(BaseModel):
    applicants: list[ApplicantProfile] = Field(
        default_factory=list,
        max_length=1000,
        description="Optional batch of applicants to evaluate (max 1000). If empty, live audit log is evaluated."
    )


class DriftFeatureMetric(BaseModel):
    ks_statistic: float | None = None
    chi2_statistic: float | None = None
    p_value: float
    psi: float | None = None
    drift_detected: bool
    sample_size: int | None = None
    warning: str | None = None


class DriftResponse(BaseModel):
    evaluation_source: str
    sample_size: int
    reference_sample_size: int
    min_sample_size_met: bool = True
    overall_drift_detected: bool = False
    drift_metrics: dict[str, DriftFeatureMetric]
    categorical_drift: dict[str, Any] | None = None
    prediction_drift: dict[str, Any] | None = None


class AuditLogEntry(BaseModel):
    id: int
    timestamp: str
    age: int
    bmi: float
    children: int
    smoker: str
    region: str
    predicted_cost: float
    interval_lower: float
    interval_upper: float
    premium_tier: str
    model_version: str


class AuditLogsResponse(BaseModel):
    total_logged: int
    records: list[AuditLogEntry]

