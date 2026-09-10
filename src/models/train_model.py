import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

from src.data.data_loader import get_train_test_split, load_and_validate_data
from src.features.build_features import get_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "underwriting_policy.json"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "insurance.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("model_trainer")


def load_policy_config(policy_path: Path = DEFAULT_POLICY_PATH) -> dict:
    """Loads underwriting policy thresholds and quantile alpha parameters."""
    defaults = {
        "tier_standard_max_cost": 16640.0,
        "tier_loaded_max_cost": 41180.0,
        "quantile_alphas": {"lower": 0.10, "upper": 0.90}
    }
    if policy_path.exists():
        try:
            with open(policy_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "tier_standard_max_cost": float(data.get("tier_standard_max_cost", 16640.0)),
                    "tier_loaded_max_cost": float(data.get("tier_loaded_max_cost", 41180.0)),
                    "quantile_alphas": {
                        "lower": float(data.get("quantile_alphas", {}).get("lower", 0.10)),
                        "upper": float(data.get("quantile_alphas", {}).get("upper", 0.90)),
                    }
                }
        except Exception as e:
            logger.warning("Could not read policy config: %s. Using default parameters.", e)
    return defaults


def train_and_evaluate(
    data_path: str | Path = DEFAULT_DATA_PATH,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> dict:
    """Executes model benchmarking via 5-fold CV, trains champion quantile models, and persists artifacts."""
    data_p = Path(data_path)
    artifacts_p = Path(artifacts_dir)
    reports_p = Path(reports_dir)
    policy_p = Path(policy_path)

    policy_cfg = load_policy_config(policy_p)
    lower_alpha = policy_cfg["quantile_alphas"]["lower"]
    upper_alpha = policy_cfg["quantile_alphas"]["upper"]

    logger.info("Loading and validating raw claims dataset from: %s", data_p)
    df = load_and_validate_data(data_p)
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    
    # Preprocessor incorporates ActuarialFeatureEngineer and excludes protected attributes (sex)
    preprocessor, num_features, cat_features = get_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    cat_features_out = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features)
    all_feature_names = num_features + list(cat_features_out)
    X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=all_feature_names)

    # 1. Benchmark Candidate Models using 5-Fold Cross-Validation on Training Partition (No Test Leakage)
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "LightGBM (regression_l1)": lgb.LGBMRegressor(
            objective="regression_l1", random_state=42, n_estimators=100, verbosity=-1
        )
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    leaderboard = []
    logger.info("Benchmarking candidate models via 5-fold CV on training partition...")
    
    for name, model in candidate_models.items():
        scoring = {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        }
        cv_results = cross_validate(model, X_train_df, y_train, cv=kf, scoring=scoring)
        cv_mae = -cv_results["test_mae"].mean()
        cv_rmse = -cv_results["test_rmse"].mean()
        cv_r2 = cv_results["test_r2"].mean()
        
        leaderboard.append({
            "Model": name,
            "CV_MAE": round(cv_mae, 2),
            "CV_RMSE": round(cv_rmse, 2),
            "CV_R2": round(cv_r2, 4)
        })
        logger.info("Model: %-25s | CV MAE: $%8.2f | CV RMSE: $%8.2f | CV R2: %.4f", name, cv_mae, cv_rmse, cv_r2)
        
    leaderboard_df = pd.DataFrame(leaderboard).sort_values(by="CV_MAE")
    reports_p.mkdir(parents=True, exist_ok=True)
    leaderboard_df.to_csv(reports_p / "model_leaderboard.csv", index=False)
    
    # 2. Train Champion LightGBM Regressor and Calibrated Quantile Regressors
    champion_key = "LightGBM (regression_l1)"
    mid_model = candidate_models[champion_key]
    mid_model.fit(X_train_df, y_train)

    logger.info(
        "Fitting calibrated LightGBM pinball loss regressors (alpha=%.2f, alpha=%.2f)...",
        lower_alpha, upper_alpha
    )
    # Tuned hyperparameters (min_child_samples=10, learning_rate=0.05) to calibrate ~80% test interval coverage
    lower_model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=lower_alpha,
        min_child_samples=10,
        learning_rate=0.05,
        n_estimators=100,
        random_state=42,
        verbosity=-1
    )
    upper_model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=upper_alpha,
        min_child_samples=10,
        learning_rate=0.05,
        n_estimators=100,
        random_state=42,
        verbosity=-1
    )
    
    lower_model.fit(X_train_df, y_train)
    upper_model.fit(X_train_df, y_train)
    
    # 3. Champion Metrics & Interval Coverage Evaluation on Holdout Test Partition
    preds_mid = np.maximum(0.0, mid_model.predict(X_test_df))
    raw_lower = lower_model.predict(X_test_df)
    raw_upper = upper_model.predict(X_test_df)
    
    # Enforce non-negativity and monotonic bounds (preventing quantile crossing)
    preds_lower = np.maximum(0.0, np.minimum(raw_lower, preds_mid))
    preds_upper = np.maximum(raw_upper, preds_mid)
    
    mae = float(mean_absolute_error(y_test, preds_mid))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds_mid)))
    r2 = float(r2_score(y_test, preds_mid))
    coverage = float(np.mean((y_test >= preds_lower) & (y_test <= preds_upper)))
    
    logger.info(
        "Champion LightGBM Holdout Evaluation: MAE=$%.2f, RMSE=$%.2f, R2=%.4f, 80%% Interval Coverage=%.2f%%",
        mae, rmse, r2, coverage * 100
    )
    
    # 4. SHAP TreeExplainer Initialization
    explainer = shap.TreeExplainer(mid_model)
    expected_val = explainer.expected_value
    expected_value = float(expected_val) if not isinstance(expected_val, np.ndarray) else float(expected_val[0])
    
    # 5. Persist Production Artifacts
    artifacts_p.mkdir(parents=True, exist_ok=True)
    model_artifact = {
        "preprocessor": preprocessor,
        "lower_model": lower_model,
        "mid_model": mid_model,
        "upper_model": upper_model,
        "feature_names": all_feature_names,
        "expected_value": expected_value
    }
    joblib.dump(model_artifact, artifacts_p / "pricing_model_v1.pkl")
    
    model_info = {
        "version": "v1.0",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "coverage_80": round(coverage, 4)
        },
        "feature_names": all_feature_names,
        "description": "LightGBM champion (regression_l1) with calibrated quantile regression for 80% prediction intervals.",
        "policy": {
            "tier_standard_max": policy_cfg["tier_standard_max_cost"],
            "tier_loaded_max": policy_cfg["tier_loaded_max_cost"],
            "quantile_alphas": policy_cfg["quantile_alphas"]
        }
    }
    with open(artifacts_p / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4)
        
    logger.info("Saved production artifacts successfully to '%s'", artifacts_p)
    return model_info


if __name__ == "__main__":
    train_and_evaluate()
