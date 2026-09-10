import os
import pandas as pd
import pytest
from src.monitoring.audit_logger import (
    init_db,
    log_prediction,
    log_predictions_batch,
    get_recent_predictions,
    get_prediction_count
)
from src.monitoring.drift_monitor import (
    calculate_categorical_drift,
    calculate_comprehensive_drift,
    calculate_ks_drift,
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_audit.db"
    init_db(db_file)
    return str(db_file)


def test_sqlite_audit_logger(temp_db):
    assert get_prediction_count(temp_db) == 0
    
    applicant = {
        "age": 32,
        "bmi": 24.5,
        "children": 1,
        "smoker": "no",
        "region": "northeast"
    }
    
    row_id = log_prediction(
        applicant=applicant,
        predicted_cost=4250.0,
        interval_80=(3500.0, 5200.0),
        premium_tier="Standard",
        model_version="v1.0",
        db_path=temp_db
    )
    assert row_id == 1
    assert get_prediction_count(temp_db) == 1
    
    records = get_recent_predictions(limit=10, db_path=temp_db)
    assert len(records) == 1
    assert records.iloc[0]["age"] == 32
    assert records.iloc[0]["predicted_cost"] == 4250.0
    assert records.iloc[0]["premium_tier"] == "Standard"


def test_ks_drift_detection():
    # Baseline data
    ref_df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45, 50, 55, 60],
        "bmi": [22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0],
        "children": [0, 1, 2, 0, 1, 2, 0, 1, 2]
    })
    
    # Identical distribution -> No drift
    prod_similar = pd.DataFrame({
        "age": [21, 26, 31, 36, 41, 46, 51, 56, 61],
        "bmi": [22.5, 24.5, 26.5, 28.5, 30.5, 32.5, 34.5, 36.5, 38.5],
        "children": [0, 1, 2, 0, 1, 2, 0, 1, 2]
    })
    
    metrics = calculate_ks_drift(ref_df, prod_similar)
    assert "age" in metrics
    assert not metrics["age"]["drift_detected"]
    
    # Severely shifted distribution -> Drift detected
    prod_shifted = pd.DataFrame({
        "age": [85, 90, 92, 95, 98, 88, 89, 91, 94],
        "bmi": [52.0, 55.0, 58.0, 53.0, 56.0, 54.0, 57.0, 59.0, 51.0],
        "children": [10, 12, 11, 10, 14, 12, 10, 11, 13]
    })
    
    drift_metrics = calculate_ks_drift(ref_df, prod_shifted)
    assert drift_metrics["age"]["drift_detected"]
    assert drift_metrics["bmi"]["drift_detected"]


def test_batch_audit_logger(temp_db):
    records = [
        {
            "applicant": {"age": 25, "bmi": 22.0, "children": 0, "smoker": "no", "region": "northeast"},
            "predicted_cost": 3100.0,
            "interval_80": (2500.0, 4000.0),
            "premium_tier": "Standard",
            "model_version": "v1.0",
        },
        {
            "applicant": {"age": 55, "bmi": 35.0, "children": 2, "smoker": "yes", "region": "southeast"},
            "predicted_cost": 42000.0,
            "interval_80": (36000.0, 48000.0),
            "premium_tier": "Refer to Underwriter",
            "model_version": "v1.0",
        },
    ]
    inserted = log_predictions_batch(records, db_path=temp_db)
    assert inserted == 2
    assert get_prediction_count(temp_db) == 2


def test_categorical_drift_detection():
    # Baseline: 80% non-smokers, 20% smokers
    ref_df = pd.DataFrame({
        "smoker": ["no"] * 80 + ["yes"] * 20,
        "region": ["northeast"] * 25 + ["northwest"] * 25 + ["southeast"] * 25 + ["southwest"] * 25,
    })

    # Production similar: 80% non-smokers, 20% smokers
    prod_similar = pd.DataFrame({
        "smoker": ["no"] * 40 + ["yes"] * 10,
        "region": ["northeast"] * 12 + ["northwest"] * 13 + ["southeast"] * 12 + ["southwest"] * 13,
    })
    cat_metrics = calculate_categorical_drift(ref_df, prod_similar)
    assert "smoker" in cat_metrics
    assert not cat_metrics["smoker"]["drift_detected"]

    # Production shifted: 90% smokers, 10% non-smokers
    prod_shifted = pd.DataFrame({
        "smoker": ["no"] * 5 + ["yes"] * 45,
        "region": ["southeast"] * 50,
    })
    shifted_metrics = calculate_categorical_drift(ref_df, prod_shifted)
    assert shifted_metrics["smoker"]["drift_detected"]
    assert shifted_metrics["region"]["drift_detected"]


def test_comprehensive_drift_small_sample_suppressed():
    ref_df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45] * 10,
        "bmi": [22.0, 24.0, 26.0, 28.0, 30.0] * 10,
        "children": [0, 1, 0, 1, 2] * 10,
        "smoker": ["no", "no", "no", "yes", "no"] * 10,
        "region": ["northeast", "northwest", "southeast", "southwest", "northeast"] * 10,
        "predicted_cost": [3000.0, 4000.0, 5000.0, 15000.0, 6000.0] * 10,
    })
    # Tiny sample of 2 records
    prod_tiny = pd.DataFrame({
        "age": [26, 31],
        "bmi": [23.0, 25.0],
        "children": [0, 1],
        "smoker": ["no", "no"],
        "region": ["northeast", "northwest"],
        "predicted_cost": [3200.0, 4100.0],
    })
    drift_res = calculate_comprehensive_drift(ref_df, prod_tiny)
    assert not drift_res["min_sample_size_met"]
    assert not drift_res["overall_drift_detected"]


def test_comprehensive_drift_no_false_positive_on_reference():
    ref_df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45, 50, 55, 60] * 10,
        "bmi": [22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0] * 10,
        "children": [0, 1, 2, 0, 1, 2, 0, 1, 2] * 10,
        "smoker": ["no"] * 70 + ["yes"] * 20,
        "region": ["northeast"] * 25 + ["northwest"] * 25 + ["southeast"] * 20 + ["southwest"] * 20,
        "predicted_cost": [4000.0 + i * 100 for i in range(90)],
    })
    prod_identical = ref_df.copy()
    drift_res = calculate_comprehensive_drift(ref_df, prod_identical)
    assert drift_res["min_sample_size_met"]
    assert not drift_res["overall_drift_detected"]

