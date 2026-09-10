from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp

DEFAULT_NUMERICAL_FEATURES = ["age", "bmi", "children"]
DEFAULT_CATEGORICAL_FEATURES = ["smoker", "region"]
DRIFT_SIGNIFICANCE_ALPHA = 0.05
MIN_RECOMMENDED_SAMPLE_SIZE = 30


def calculate_psi(
    ref_series: pd.Series, 
    prod_series: pd.Series, 
    epsilon: float = 1e-4
) -> float:
    """Calculates Population Stability Index (PSI) between reference and production distributions.
    
    PSI Interpretability Standards:
    - PSI < 0.10: Insignificant distribution shift / stable.
    - 0.10 <= PSI < 0.25: Moderate shift / cautionary drift.
    - PSI >= 0.25: Significant population drift / model review recommended.
    """
    ref_counts = ref_series.value_counts(normalize=True)
    prod_counts = prod_series.value_counts(normalize=True)
    all_categories = sorted(set(ref_counts.index).union(set(prod_counts.index)))

    psi_val = 0.0
    for cat in all_categories:
        p = float(ref_counts.get(cat, epsilon))
        q = float(prod_counts.get(cat, epsilon))
        # Smooth zero frequencies with epsilon to prevent division-by-zero or log(0)
        p = max(p, epsilon)
        q = max(q, epsilon)
        psi_val += (q - p) * np.log(q / p)

    return round(float(psi_val), 4)


def calculate_ks_drift(
    reference_data: pd.DataFrame, 
    production_data: pd.DataFrame, 
    features: list[str] | None = None,
    alpha: float = DRIFT_SIGNIFICANCE_ALPHA,
) -> dict[str, dict[str, Any]]:
    """Calculates Kolmogorov-Smirnov (KS) two-sample test statistic for continuous covariates.
    
    Args:
        reference_data: Historical baseline dataframe (e.g. training set).
        production_data: Live or batch inference dataframe.
        features: Numerical feature names to evaluate. Defaults to age, bmi, children.
        alpha: Statistical significance threshold.
        
    Returns:
        dict mapping feature names to ks_statistic, p_value, and drift_detected flag.
    """
    if features is None:
        features = DEFAULT_NUMERICAL_FEATURES
        
    drift_metrics: dict[str, dict[str, Any]] = {}
    
    if production_data.empty:
        return drift_metrics

    for feature in features:
        if feature in reference_data.columns and feature in production_data.columns:
            ref_col = reference_data[feature].dropna()
            prod_col = production_data[feature].dropna()
            
            if len(prod_col) < 2:
                drift_metrics[feature] = {
                    "ks_statistic": 0.0,
                    "p_value": 1.0,
                    "drift_detected": False,
                    "sample_size": len(prod_col),
                    "warning": "Sample size too small for statistical hypothesis test (N < 2).",
                }
                continue
                
            stat, p_value = ks_2samp(ref_col, prod_col)
            drift_metrics[feature] = {
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": bool(p_value < alpha),
                "sample_size": len(prod_col),
            }
            if len(prod_col) < MIN_RECOMMENDED_SAMPLE_SIZE:
                drift_metrics[feature]["warning"] = (
                    f"Sample size ({len(prod_col)}) below recommended threshold ({MIN_RECOMMENDED_SAMPLE_SIZE}). "
                    "Results may exhibit high variance."
                )
            
    return drift_metrics


def calculate_categorical_drift(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
    features: list[str] | None = None,
    alpha: float = DRIFT_SIGNIFICANCE_ALPHA,
) -> dict[str, dict[str, Any]]:
    """Evaluates categorical covariate drift using Chi-Square goodness-of-fit and PSI.
    
    Args:
        reference_data: Historical baseline dataframe.
        production_data: Production scoring dataframe.
        features: Categorical feature names. Defaults to smoker, region.
        alpha: Significance threshold for Chi-Square test.
        
    Returns:
        dict mapping feature names to chi2_statistic, p_value, psi, and drift_detected flag.
    """
    if features is None:
        features = DEFAULT_CATEGORICAL_FEATURES

    categorical_metrics: dict[str, dict[str, Any]] = {}

    if production_data.empty:
        return categorical_metrics

    for feature in features:
        if feature in reference_data.columns and feature in production_data.columns:
            ref_col = reference_data[feature].dropna().astype(str)
            prod_col = production_data[feature].dropna().astype(str)

            if len(prod_col) < 2:
                categorical_metrics[feature] = {
                    "chi2_statistic": 0.0,
                    "p_value": 1.0,
                    "psi": 0.0,
                    "drift_detected": False,
                    "sample_size": len(prod_col),
                    "warning": "Sample size too small for Chi-Square test (N < 2).",
                }
                continue

            # Calculate PSI
            psi_score = calculate_psi(ref_col, prod_col)

            # Calculate Chi-Square Goodness-of-Fit
            ref_probs = ref_col.value_counts(normalize=True)
            categories = sorted(ref_probs.index)
            prod_counts = prod_col.value_counts()

            # Align observed and expected frequencies across reference categories
            obs_counts = [float(prod_counts.get(cat, 0)) for cat in categories]
            total_prod = float(len(prod_col))

            # If new unseen categories appeared in production, add them to observed
            unseen_count = sum(float(cnt) for cat, cnt in prod_counts.items() if cat not in categories)
            if unseen_count > 0:
                obs_counts.append(unseen_count)
                # expected for unseen category is small proportional share
                exp_counts = [total_prod * float(ref_probs[cat]) * (1.0 - 1e-4) for cat in categories] + [total_prod * 1e-4]
            else:
                exp_counts = [total_prod * float(ref_probs[cat]) for cat in categories]

            try:
                stat, p_value = chisquare(f_obs=obs_counts, f_exp=exp_counts)
                chi2_stat = round(float(stat), 4)
                chi2_p = round(float(p_value), 4)
            except Exception:
                chi2_stat = 0.0
                chi2_p = 1.0

            # Suppress statistical hypothesis testing and PSI alerting on small sample sizes
            is_sample_adequate = len(prod_col) >= MIN_RECOMMENDED_SAMPLE_SIZE
            if is_sample_adequate:
                drift_detected = bool((chi2_p < alpha) or (psi_score >= 0.25))
            else:
                drift_detected = False

            categorical_metrics[feature] = {
                "chi2_statistic": chi2_stat,
                "p_value": chi2_p,
                "psi": psi_score,
                "drift_detected": drift_detected,
                "sample_size": len(prod_col),
            }

            if not is_sample_adequate:
                categorical_metrics[feature]["warning"] = (
                    f"Sample size ({len(prod_col)}) below recommended threshold ({MIN_RECOMMENDED_SAMPLE_SIZE}). "
                    "Drift hypothesis testing suppressed due to small sample size."
                )

    return categorical_metrics


def calculate_comprehensive_drift(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
    alpha: float = DRIFT_SIGNIFICANCE_ALPHA,
) -> dict[str, Any]:
    """Runs end-to-end drift audit over numerical, categorical, and output prediction dimensions."""
    num_drift = calculate_ks_drift(reference_data, production_data, alpha=alpha)
    cat_drift = calculate_categorical_drift(reference_data, production_data, alpha=alpha)

    # Check for prediction drift by comparing production predictions against baseline model predictions
    pred_drift: dict[str, Any] | None = None
    ref_col_name = "predicted_cost" if "predicted_cost" in reference_data.columns else "charges"
    if "predicted_cost" in production_data.columns and ref_col_name in reference_data.columns:
        prod_preds = production_data["predicted_cost"].dropna()
        ref_costs = reference_data[ref_col_name].dropna()
        if len(prod_preds) >= 2:
            stat, p_val = ks_2samp(ref_costs, prod_preds)
            is_sample_adequate = len(prod_preds) >= MIN_RECOMMENDED_SAMPLE_SIZE
            pred_drift = {
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_val), 4),
                "drift_detected": bool(p_val < alpha) if is_sample_adequate else False,
                "sample_size": len(prod_preds),
            }
            if not is_sample_adequate:
                pred_drift["warning"] = (
                    f"Sample size ({len(prod_preds)}) below recommended threshold ({MIN_RECOMMENDED_SAMPLE_SIZE}). "
                    "Prediction drift test suppressed."
                )

    # Apply Bonferroni correction for multiple hypothesis testing
    total_tests = len(num_drift) + len(cat_drift) + (1 if pred_drift else 0)
    corrected_alpha = alpha / max(total_tests, 1)
    
    min_size_met = len(production_data) >= MIN_RECOMMENDED_SAMPLE_SIZE
    
    any_num_drift = any(
        v.get("p_value", 1.0) < corrected_alpha and v.get("drift_detected", False)
        for v in num_drift.values()
    )
    any_cat_drift = any(
        v.get("drift_detected", False) for v in cat_drift.values()
    )
    any_pred_drift = bool(
        pred_drift and pred_drift.get("drift_detected", False) and pred_drift.get("p_value", 1.0) < corrected_alpha
    )

    overall_drift = bool(min_size_met and (any_num_drift or any_cat_drift or any_pred_drift))

    return {
        "sample_size": len(production_data),
        "reference_sample_size": len(reference_data),
        "min_sample_size_met": min_size_met,
        "overall_drift_detected": overall_drift,
        "corrected_alpha": round(corrected_alpha, 6),
        "numerical_drift": num_drift,
        "categorical_drift": cat_drift,
        "prediction_drift": pred_drift,
    }
