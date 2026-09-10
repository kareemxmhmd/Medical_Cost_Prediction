import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import joblib
import numpy as np
import pandas as pd
import shap

from src.data.data_loader import load_and_validate_data
from src.monitoring.audit_logger import (
    get_prediction_count,
    get_recent_predictions,
    init_db,
    log_prediction,
    log_predictions_batch,
)
from src.monitoring.drift_monitor import calculate_comprehensive_drift, calculate_ks_drift

logger = logging.getLogger("inference_engine")

DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "pricing_model_v1.pkl"
DEFAULT_INFO_PATH = PROJECT_ROOT / "artifacts" / "model_info.json"
DEFAULT_BASELINE_PATH = PROJECT_ROOT / "data" / "insurance.csv"
DEFAULT_DB_PATH = PROJECT_ROOT / "audit_predictions.db"
DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "underwriting_policy.json"


def load_policy_thresholds(policy_path: Path = DEFAULT_POLICY_PATH) -> tuple[float, float]:
    """Loads tier decision thresholds from config with fallback to actuarial baseline constants."""
    if policy_path.exists():
        try:
            with open(policy_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return (
                    float(data.get("tier_standard_max_cost", 16640.0)),
                    float(data.get("tier_loaded_max_cost", 41180.0)),
                )
        except Exception as e:
            logger.warning("Could not read policy config: %s. Using default thresholds.", e)
    return 16640.0, 41180.0


TIER_STANDARD_MAX_COST, TIER_LOADED_MAX_COST = load_policy_thresholds()


def determine_tier(
    cost: float,
    standard_max: float | None = None,
    loaded_max: float | None = None,
) -> str:
    """Classifies an applicant's predicted annual cost into an underwriting tier."""
    s_max = standard_max if standard_max is not None else TIER_STANDARD_MAX_COST
    l_max = loaded_max if loaded_max is not None else TIER_LOADED_MAX_COST
    if cost > l_max:
        return "Refer to Underwriter"
    if cost > s_max:
        return "Loaded"
    return "Standard"


FEATURE_PARENT_MAP = {
    "smoker_no": "smoker",
    "smoker_yes": "smoker",
    "region_northeast": "region",
    "region_northwest": "region",
    "region_southeast": "region",
    "region_southwest": "region",
    "is_obese": "obesity (BMI >= 30)",
    "obese_smoker": "smoking & obesity interaction",
}


def generate_reason_codes(
    shap_values: np.ndarray,
    feature_names: list[str],
    applicant_dict: dict,
    max_codes: int = 3,
) -> list[dict[str, Any]]:
    """Extracts dominant risk drivers using SHAP marginal dollar contributions aggregated by feature."""
    feature_impacts: dict[str, float] = {}

    for i, feature in enumerate(feature_names):
        impact = float(shap_values[i])
        if feature in FEATURE_PARENT_MAP:
            feature_key = FEATURE_PARENT_MAP[feature]
        elif "_" in feature:
            prefix = feature.split("_")[0]
            feature_key = prefix if prefix in applicant_dict else feature
        else:
            feature_key = feature

        feature_impacts[feature_key] = feature_impacts.get(feature_key, 0.0) + impact

    codes = []
    for feat, total_impact in feature_impacts.items():
        if abs(total_impact) > 10.0:
            direction = "increases cost" if total_impact > 0 else "decreases cost"
            codes.append(
                {
                    "feature": feat,
                    "impact": round(abs(total_impact), 2),
                    "direction": direction,
                }
            )

    codes.sort(key=lambda x: x["impact"], reverse=True)
    return codes[:max_codes]


class UnderwritingInferenceEngine:
    """Production inference engine for health insurance actuarial cost prediction."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        info_path: Path | str = DEFAULT_INFO_PATH,
        baseline_path: Path | str = DEFAULT_BASELINE_PATH,
        db_path: Path | str = DEFAULT_DB_PATH,
        policy_path: Path | str = DEFAULT_POLICY_PATH,
    ):
        self.model_path = Path(model_path)
        self.info_path = Path(info_path)
        self.baseline_path = Path(baseline_path)
        self.db_path = Path(db_path)
        self.policy_path = Path(policy_path)

        self.model_artifact: dict | None = None
        self.model_info: dict | None = None
        self.explainer: shap.TreeExplainer | None = None
        self.baseline_data: pd.DataFrame | None = None

        self.tier_standard_max, self.tier_loaded_max = load_policy_thresholds(self.policy_path)
        self.load()

    def load(self) -> None:
        """Initializes SQLite database schema and loads model artifacts into memory."""
        init_db(self.db_path)

        if self.model_path.exists() and self.info_path.exists():
            logger.info("Loading model artifact from %s", self.model_path)
            self.model_artifact = joblib.load(self.model_path)
            with open(self.info_path, "r", encoding="utf-8") as f:
                self.model_info = json.load(f)
            logger.info("Model and SHAP explainer loaded successfully.")
        else:
            logger.error(
                "Model artifacts missing at %s or %s. Runtimes must fail fast without side-effects.",
                self.model_path,
                self.info_path,
            )

        if self.baseline_path.exists():
            try:
                self.baseline_data = load_and_validate_data(self.baseline_path)
                logger.info("Loaded baseline data (%d rows)", len(self.baseline_data))
                if self.is_ready and "predicted_cost" not in self.baseline_data.columns:
                    preprocessor = self.model_artifact["preprocessor"]
                    X_raw = preprocessor.transform(self.baseline_data)
                    feature_names = self.model_artifact.get("feature_names", [])
                    X_proc = pd.DataFrame(X_raw, columns=feature_names) if feature_names else X_raw
                    self.baseline_data["predicted_cost"] = np.maximum(
                        0.0, self.model_artifact["mid_model"].predict(X_proc)
                    )
            except Exception as e:
                logger.warning("Failed to load baseline data: %s", e)

    @property
    def is_ready(self) -> bool:
        return self.model_artifact is not None

    def _ensure_explainer(self) -> None:
        """Lazily initializes SHAP TreeExplainer on first prediction call."""
        if self.explainer is None and self.model_artifact is not None:
            self.explainer = shap.TreeExplainer(self.model_artifact["mid_model"])

    def predict_single(
        self,
        applicant: dict[str, Any],
        log_to_audit: bool = True,
    ) -> dict[str, Any]:
        """Runs end-to-end actuarial inference on a single applicant profile."""
        if not self.is_ready:
            raise RuntimeError("Inference engine model artifacts are not loaded.")

        df = pd.DataFrame([applicant])
        preprocessor = self.model_artifact["preprocessor"]
        X_raw = preprocessor.transform(df)
        feature_names = self.model_artifact.get("feature_names", [])
        X_processed = pd.DataFrame(X_raw, columns=feature_names) if feature_names else X_raw

        raw_mid = float(self.model_artifact["mid_model"].predict(X_processed)[0])
        pred_mid = float(np.maximum(0.0, raw_mid))
        raw_lower = float(self.model_artifact["lower_model"].predict(X_processed)[0])
        raw_upper = float(self.model_artifact["upper_model"].predict(X_processed)[0])

        # Enforce non-negativity and monotonic bounds (preventing quantile crossing)
        pred_lower = float(np.maximum(0.0, min(raw_lower, pred_mid)))
        pred_upper = float(max(raw_upper, pred_mid))

        self._ensure_explainer()
        shap_vals = self.explainer.shap_values(X_processed)[0]
        reason_codes = generate_reason_codes(
            shap_vals, self.model_artifact["feature_names"], applicant
        )

        tier = determine_tier(pred_mid, self.tier_standard_max, self.tier_loaded_max)
        version = self.model_info.get("version", "v1.0") if self.model_info else "v1.0"
        timestamp_iso = datetime.now(timezone.utc).isoformat()

        if log_to_audit:
            try:
                log_prediction(
                    applicant=applicant,
                    predicted_cost=pred_mid,
                    interval_80=(pred_lower, pred_upper),
                    premium_tier=tier,
                    model_version=version,
                    db_path=self.db_path,
                )
            except Exception as e:
                logger.error("Audit log persistence failed: %s", e)

        return {
            "predicted_cost": round(pred_mid, 2),
            "interval_80": (round(pred_lower, 2), round(pred_upper, 2)),
            "premium_tier": tier,
            "reason_codes": reason_codes,
            "model_version": version,
            "decision_timestamp": timestamp_iso,
        }

    def predict_batch(
        self,
        applicants: list[dict[str, Any]] | pd.DataFrame,
        log_to_audit: bool = False,
    ) -> list[dict[str, Any]]:
        """Scores a batch of applicant profiles efficiently via vectorized matrix operations."""
        if not self.is_ready:
            raise RuntimeError("Inference engine model artifacts are not loaded.")

        if isinstance(applicants, pd.DataFrame):
            df = applicants.copy()
        else:
            df = pd.DataFrame(applicants)

        if df.empty:
            return []

        preprocessor = self.model_artifact["preprocessor"]
        X_raw = preprocessor.transform(df)
        feature_names = self.model_artifact.get("feature_names", [])
        X_processed = pd.DataFrame(X_raw, columns=feature_names) if feature_names else X_raw

        raw_mids = self.model_artifact["mid_model"].predict(X_processed)
        preds_mid = np.maximum(0.0, raw_mids)
        raw_lowers = self.model_artifact["lower_model"].predict(X_processed)
        raw_uppers = self.model_artifact["upper_model"].predict(X_processed)

        preds_lower = np.maximum(0.0, np.minimum(raw_lowers, preds_mid))
        preds_upper = np.maximum(raw_uppers, preds_mid)

        self._ensure_explainer()
        shap_matrix = self.explainer.shap_values(X_processed)
        version = self.model_info.get("version", "v1.0") if self.model_info else "v1.0"
        timestamp_iso = datetime.now(timezone.utc).isoformat()

        records_dict = df.to_dict(orient="records")
        results = []
        batch_audit_records = []

        for i in range(len(df)):
            mid = float(preds_mid[i])
            low = float(preds_lower[i])
            upp = float(preds_upper[i])
            tier = determine_tier(mid, self.tier_standard_max, self.tier_loaded_max)
            app_dict = records_dict[i]
            rc = generate_reason_codes(shap_matrix[i], feature_names, app_dict)

            rec = {
                "predicted_cost": round(mid, 2),
                "interval_80": (round(low, 2), round(upp, 2)),
                "premium_tier": tier,
                "reason_codes": rc,
                "model_version": version,
                "decision_timestamp": timestamp_iso,
            }
            results.append(rec)

            if log_to_audit:
                batch_audit_records.append({
                    "applicant": app_dict,
                    "predicted_cost": mid,
                    "interval_80": (low, upp),
                    "premium_tier": tier,
                    "model_version": version,
                    "decision_timestamp": timestamp_iso,
                })

        if log_to_audit and batch_audit_records:
            try:
                log_predictions_batch(batch_audit_records, db_path=self.db_path)
            except Exception as e:
                logger.error("Batch audit log persistence failed: %s", e)

        return results

    def evaluate_drift(
        self,
        production_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Calculates multi-dimensional drift metrics comparing production data against baseline reference."""
        if self.baseline_data is None:
            raise RuntimeError("Baseline reference data is not available.")

        if production_df is None or production_df.empty:
            eval_source = "sqlite_audit_log"
            eval_df = get_recent_predictions(limit=500, db_path=self.db_path)
        else:
            eval_source = "custom_production_batch"
            eval_df = production_df

        if eval_df.empty:
            raise ValueError("No production scoring data available to evaluate drift.")

        drift_result = calculate_comprehensive_drift(self.baseline_data, eval_df)

        return {
            "evaluation_source": eval_source,
            "sample_size": drift_result["sample_size"],
            "reference_sample_size": drift_result["reference_sample_size"],
            "min_sample_size_met": drift_result["min_sample_size_met"],
            "overall_drift_detected": drift_result["overall_drift_detected"],
            "drift_metrics": drift_result["numerical_drift"],
            "categorical_drift": drift_result["categorical_drift"],
            "prediction_drift": drift_result["prediction_drift"],
        }

    def get_audit_trail(self, limit: int = 100) -> dict[str, Any]:
        """Retrieves recent audit logs from SQLite."""
        records_df = get_recent_predictions(limit=limit, db_path=self.db_path)
        total_count = get_prediction_count(db_path=self.db_path)
        return {
            "total_logged": total_count,
            "records": records_df.to_dict(orient="records"),
        }
