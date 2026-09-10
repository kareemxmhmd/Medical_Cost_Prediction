import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.models.inference import (
    DEFAULT_BASELINE_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_INFO_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_POLICY_PATH,
    UnderwritingInferenceEngine,
)
from src.monitoring.audit_logger import (
    get_prediction_count,
    get_recent_predictions,
    init_db,
    log_prediction,
    log_predictions_batch,
)

from .schemas import (
    ApplicantProfile,
    AuditLogEntry,
    AuditLogsResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriftFeatureMetric,
    DriftRequest,
    DriftResponse,
    PredictionResponse,
    ReasonCode,
)

from fastapi import Depends, Security
from fastapi.security import APIKeyHeader
import secrets

API_KEY = os.environ.get("API_KEY", "dev-api-key-change-in-production")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)):
    """Validates API key for protected endpoints using constant-time comparison."""
    if api_key is None or not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")
    return api_key

logger = logging.getLogger("api_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

MODEL_PATH = DEFAULT_MODEL_PATH
INFO_PATH = DEFAULT_INFO_PATH
BASELINE_DATA_PATH = DEFAULT_BASELINE_PATH
DB_PATH = DEFAULT_DB_PATH
POLICY_PATH = DEFAULT_POLICY_PATH

# Global inference engine instance
engine: UnderwritingInferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    init_db(DB_PATH)
    engine = UnderwritingInferenceEngine(
        model_path=MODEL_PATH,
        info_path=INFO_PATH,
        baseline_path=BASELINE_DATA_PATH,
        db_path=DB_PATH,
        policy_path=POLICY_PATH,
    )
    if engine.is_ready:
        logger.info("UnderwritingInferenceEngine initialized successfully.")
    else:
        logger.warning("Model artifacts missing. Train pipeline before scoring.")
    yield


app = FastAPI(
    title="Medical Cost Prediction & Pricing API",
    description="Actuarial cost prediction, uncertainty intervals, SHAP reason codes, and drift governance.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="Service Health Check")
def health_check():
    is_loaded = engine is not None and engine.is_ready
    version = engine.model_info.get("version", "unknown") if (engine and engine.model_info) else None
    return {
        "status": "ok",
        "model_loaded": is_loaded,
        "model_version": version,
    }


@app.get("/model-info", summary="Model Metadata & Evaluation Metrics")
def get_model_info(_api_key: str = Depends(verify_api_key)):
    if not engine or not engine.model_info:
        raise HTTPException(status_code=503, detail="Model metadata not loaded.")
    return engine.model_info


@app.post("/predict", response_model=PredictionResponse, summary="Score Single Applicant Profile")
def predict(applicant: ApplicantProfile, _api_key: str = Depends(verify_api_key)):
    if not engine or not engine.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded. Train pipeline first.")
    
    input_dict = applicant.model_dump()
    result = engine.predict_single(input_dict, log_to_audit=True)
    
    reason_codes = [ReasonCode(**rc) for rc in result["reason_codes"]]
    
    return PredictionResponse(
        predicted_cost=result["predicted_cost"],
        interval_80=result["interval_80"],
        premium_tier=result["premium_tier"],
        reason_codes=reason_codes,
        model_version=result["model_version"],
        decision_timestamp=result["decision_timestamp"],
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="Score Batch of Applicants")
def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(verify_api_key),
):
    if not engine or not engine.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded. Train pipeline first.")

    applicants_data = [app.model_dump() for app in request.applicants]
    results = engine.predict_batch(applicants_data, log_to_audit=False)

    if results:
        batch_audit_records = [
            {
                "applicant": applicants_data[i],
                "predicted_cost": results[i]["predicted_cost"],
                "interval_80": results[i]["interval_80"],
                "premium_tier": results[i]["premium_tier"],
                "model_version": results[i]["model_version"],
                "decision_timestamp": results[i]["decision_timestamp"],
            }
            for i in range(len(results))
        ]
        # Asynchronously persist batch audit log in background task
        background_tasks.add_task(log_predictions_batch, batch_audit_records, db_path=DB_PATH)

    predictions = [
        PredictionResponse(
            predicted_cost=r["predicted_cost"],
            interval_80=r["interval_80"],
            premium_tier=r["premium_tier"],
            reason_codes=[ReasonCode(**rc) for rc in r["reason_codes"]],
            model_version=r["model_version"],
            decision_timestamp=r["decision_timestamp"],
        )
        for r in results
    ]

    return BatchPredictionResponse(predictions=predictions)


@app.post("/drift", response_model=DriftResponse, summary="Evaluate Population Covariate Drift")
def check_drift(
    request: DriftRequest | None = None,
    _api_key: str = Depends(verify_api_key),
):
    if not engine or engine.baseline_data is None:
        raise HTTPException(status_code=503, detail="Baseline reference dataset not available.")
        
    applicants = request.applicants if request else []
    
    if applicants:
        eval_df = pd.DataFrame([app.model_dump() for app in applicants])
    else:
        eval_df = None
        
    try:
        drift_result = engine.evaluate_drift(eval_df)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift evaluation error: {e}")

    drift_metrics_dict = {
        feat: DriftFeatureMetric(**vals)
        for feat, vals in drift_result.get("drift_metrics", {}).items()
    }
        
    return DriftResponse(
        evaluation_source=drift_result["evaluation_source"],
        sample_size=drift_result["sample_size"],
        reference_sample_size=drift_result["reference_sample_size"],
        min_sample_size_met=drift_result.get("min_sample_size_met", True),
        overall_drift_detected=drift_result.get("overall_drift_detected", False),
        drift_metrics=drift_metrics_dict,
        categorical_drift=drift_result.get("categorical_drift"),
        prediction_drift=drift_result.get("prediction_drift"),
    )


@app.get("/drift", response_model=DriftResponse, summary="Evaluate Drift from Production Audit Log")
def get_production_drift(_api_key: str = Depends(verify_api_key)):
    return check_drift(DriftRequest(applicants=[]))


@app.get("/audit-logs", response_model=AuditLogsResponse, summary="Query Ingestion Audit Trail")
def get_audit_logs(
    limit: int = Query(default=100, ge=1, le=1000),
    _api_key: str = Depends(verify_api_key),
):
    records_df = get_recent_predictions(limit=limit, db_path=DB_PATH)
    total_count = get_prediction_count(db_path=DB_PATH)
    
    records = []
    for _, row in records_df.iterrows():
        records.append(
            AuditLogEntry(
                id=int(row["id"]),
                timestamp=str(row["timestamp"]),
                age=int(row["age"]),
                bmi=round(float(row["bmi"]), 2),
                children=int(row["children"]),
                smoker=str(row["smoker"]),
                region=str(row["region"]),
                predicted_cost=round(float(row["predicted_cost"]), 2),
                interval_lower=round(float(row["interval_lower"]), 2),
                interval_upper=round(float(row["interval_upper"]), 2),
                premium_tier=str(row["premium_tier"]),
                model_version=str(row["model_version"]),
            )
        )
        
    return AuditLogsResponse(total_logged=total_count, records=records)
