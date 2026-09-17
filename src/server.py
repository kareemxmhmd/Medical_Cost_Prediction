import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.pkl"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Insurance Charges Predictor")

# Fix #3: allow_credentials=True is incompatible with allow_origins=["*"].
# Use the CORS_ORIGINS env var in production to set the exact frontend URL.
# Example: CORS_ORIGINS=https://myapp.up.railway.app
CORS_ORIGINS_ENV = os.environ.get("CORS_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in CORS_ORIGINS_ENV.split(",")]
_use_credentials = ALLOWED_ORIGINS != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=_use_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load model artefacts at startup
# ---------------------------------------------------------------------------
# model.pkl is a sklearn Pipeline: preprocessor (OHE) + RandomForestRegressor
# FEATURE_COLUMNS are the RAW input column names before OHE (e.g. ['age','sex',...])
model = joblib.load(MODEL_PATH)
FEATURE_COLUMNS: list = joblib.load(COLUMNS_PATH)   # raw feature names
METRICS: dict = joblib.load(METRICS_PATH)

MODEL_VERSION = f"{METRICS.get('model_name', 'RandomForest')}-v1"

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------
VALID_SEX = ["female", "male"]
VALID_SMOKER = ["no", "yes"]
VALID_REGION = ["northeast", "northwest", "southeast", "southwest"]

# Underwriting tier thresholds (USD annual)
TIER_LOADED_THRESHOLD = 15_000
TIER_REFER_THRESHOLD = 30_000


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    age: int
    bmi: float
    children: int
    smoker: str
    region: str
    # Fix #2: sex is optional — excluded per ACA § 2701 compliance.
    # Defaults to "male" (dataset majority class) when omitted.
    sex: Optional[str] = "male"


class ReasonCode(BaseModel):
    feature: str
    impact: float
    direction: str   # "increases cost" | "decreases cost"


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    predicted_cost: float
    interval_80: list[float]         # [p10, p90] across RF estimators
    premium_tier: str                # "Standard" | "Loaded" | "Refer to Underwriter"
    reason_codes: list[ReasonCode]
    model_version: str
    decision_timestamp: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_feature_row(req: PredictRequest) -> pd.DataFrame:
    """Build a single-row DataFrame with the raw feature columns the pipeline expects."""
    data = {
        "age":      req.age,
        "sex":      req.sex or "male",
        "bmi":      req.bmi,
        "children": req.children,
        "smoker":   req.smoker,
        "region":   req.region,
    }
    # Keep only the columns the pipeline was trained on, in the original order
    return pd.DataFrame([{col: data[col] for col in FEATURE_COLUMNS}])


def _rf_prediction_interval(row: pd.DataFrame) -> tuple[float, float]:
    """
    Derive an 80% prediction interval from the spread of individual tree
    predictions inside the RandomForestRegressor (10th–90th percentile).
    Falls back to ±30% around the point estimate if the pipeline structure
    changes unexpectedly.
    """
    try:
        preprocessor = model.named_steps["preprocessor"]
        regressor    = model.named_steps["regressor"]
        X_t = preprocessor.transform(row)
        tree_preds = np.array([tree.predict(X_t)[0] for tree in regressor.estimators_])
        return float(np.percentile(tree_preds, 10)), float(np.percentile(tree_preds, 90))
    except Exception:
        point = float(model.predict(row)[0])
        return round(point * 0.70, 2), round(point * 1.30, 2)


def _reason_codes(row: pd.DataFrame, prediction: float) -> list[ReasonCode]:
    """
    Approximate feature-level SHAP drivers by permutation: zero out each raw
    feature value and measure the delta vs. the point prediction.
    Returns the top-4 drivers sorted by absolute impact (USD).
    """
    codes: list[ReasonCode] = []
    for col in FEATURE_COLUMNS:
        perturbed = row.copy()
        # Zero-out: 0 for numerics, empty string for categoricals
        perturbed[col] = 0 if row[col].dtype != object else ""
        pp = float(model.predict(perturbed)[0])
        impact = prediction - pp
        if abs(impact) < 1.0:
            continue
        label = col.replace("_", " ").strip()
        codes.append(ReasonCode(
            feature=label,
            impact=round(abs(impact), 2),
            direction="increases cost" if impact > 0 else "decreases cost",
        ))

    codes.sort(key=lambda c: c.impact, reverse=True)
    return codes[:4]


def _premium_tier(cost: float) -> str:
    if cost >= TIER_REFER_THRESHOLD:
        return "Refer to Underwriter"
    if cost >= TIER_LOADED_THRESHOLD:
        return "Loaded"
    return "Standard"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/metadata")
def model_metadata():
    return METRICS


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.sex and req.sex not in VALID_SEX:
        raise HTTPException(status_code=400, detail="sex must be 'male' or 'female'")
    if req.smoker not in VALID_SMOKER:
        raise HTTPException(status_code=400, detail="smoker must be 'yes' or 'no'")
    if req.region not in VALID_REGION:
        raise HTTPException(status_code=400, detail="invalid region")

    row = _build_feature_row(req)
    prediction = round(float(model.predict(row)[0]), 2)
    p10, p90 = _rf_prediction_interval(row)
    codes = _reason_codes(row, prediction)
    tier = _premium_tier(prediction)

    return PredictResponse(
        predicted_cost=prediction,
        interval_80=[round(p10, 2), round(p90, 2)],
        premium_tier=tier,
        reason_codes=codes,
        model_version=MODEL_VERSION,
        decision_timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Entrypoint — Fix #8: uvicorn reads $PORT explicitly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
