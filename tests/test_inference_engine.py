import pandas as pd
import pytest
from src.models.inference import UnderwritingInferenceEngine, determine_tier


@pytest.fixture
def engine():
    return UnderwritingInferenceEngine()


def test_inference_engine_loaded(engine):
    assert engine.is_ready
    assert engine.model_info is not None
    assert "metrics" in engine.model_info
    assert engine.baseline_data is not None


def test_inference_engine_predict_single(engine):
    applicant = {
        "age": 30,
        "bmi": 25.0,
        "children": 0,
        "smoker": "no",
        "region": "southwest",
    }
    result = engine.predict_single(applicant, log_to_audit=False)
    
    assert "predicted_cost" in result
    assert "interval_80" in result
    assert len(result["interval_80"]) == 2
    assert result["interval_80"][0] <= result["predicted_cost"] <= result["interval_80"][1]
    assert result["premium_tier"] in ["Standard", "Loaded", "Refer to Underwriter"]
    assert "reason_codes" in result
    assert len(result["reason_codes"]) <= 3


def test_inference_engine_predict_batch(engine):
    applicants = [
        {"age": 25, "bmi": 22.0, "children": 0, "smoker": "no", "region": "northeast"},
        {"age": 55, "bmi": 32.0, "children": 2, "smoker": "yes", "region": "southeast"},
    ]
    results = engine.predict_batch(applicants, log_to_audit=False)
    assert len(results) == 2
    assert results[0]["predicted_cost"] < results[1]["predicted_cost"]


def test_determine_tier():
    assert determine_tier(5000.0) == "Standard"
    assert determine_tier(16640.0) == "Standard"
    assert determine_tier(25000.0) == "Loaded"
    assert determine_tier(41180.0) == "Loaded"
    assert determine_tier(45000.0) == "Refer to Underwriter"


def test_generate_reason_codes():
    import numpy as np
    from src.models.inference import generate_reason_codes
    
    feature_names = ["age", "bmi", "children", "smoker_no", "smoker_yes", "region_northeast", "region_northwest", "region_southeast", "region_southwest"]
    shap_values = np.array([500.0, 200.0, -50.0, -1500.0, 1500.0, 10.0, -5.0, 3.0, -8.0])
    applicant = {"age": 55, "bmi": 35.0, "children": 2, "smoker": "yes", "region": "southeast"}
    
    codes = generate_reason_codes(shap_values, feature_names, applicant)
    
    # Should have at most 3 codes
    assert len(codes) <= 3
    # smoker should be aggregated and be the top driver (1500 + -1500 = 0... wait)
    # Actually smoker_no=-1500 + smoker_yes=1500 = 0 net. Let me fix.
    # age=500 should be top, bmi=200 second, children=-50 below threshold
    assert codes[0]["feature"] == "age"
    assert codes[0]["direction"] == "increases cost"
    assert codes[1]["feature"] == "bmi"


def test_generate_reason_codes_no_significant():
    import numpy as np
    from src.models.inference import generate_reason_codes
    
    feature_names = ["age", "bmi", "children"]
    shap_values = np.array([5.0, -3.0, 2.0])  # All below threshold of 10
    applicant = {"age": 30, "bmi": 25.0, "children": 0}
    
    codes = generate_reason_codes(shap_values, feature_names, applicant)
    assert len(codes) == 0


def test_determine_tier_boundary_values():
    from src.models.inference import determine_tier
    # Exactly at boundary: cost > s_max means Standard includes the boundary
    assert determine_tier(16640.0) == "Standard"
    assert determine_tier(16640.01) == "Loaded"
    assert determine_tier(41180.0) == "Loaded"
    assert determine_tier(41180.01) == "Refer to Underwriter"
    assert determine_tier(0.0) == "Standard"


def test_predict_single_non_negative_clamp(engine):
    # Very young applicant with low BMI and no smoking
    applicant = {"age": 18, "bmi": 15.0, "children": 0, "smoker": "no", "region": "northeast"}
    res = engine.predict_single(applicant, log_to_audit=False)
    assert res["predicted_cost"] >= 0.0
    assert res["interval_80"][0] >= 0.0
    assert res["interval_80"][0] <= res["predicted_cost"] <= res["interval_80"][1]


def test_engineered_feature_reason_codes():
    import numpy as np
    from src.models.inference import generate_reason_codes

    feature_names = ["age", "bmi", "children", "is_obese", "obese_smoker", "smoker_no", "smoker_yes"]
    shap_values = np.array([100.0, 50.0, 5.0, 800.0, 3500.0, -200.0, 400.0])
    applicant = {"age": 45, "bmi": 35.0, "children": 1, "smoker": "yes", "region": "southeast"}

    codes = generate_reason_codes(shap_values, feature_names, applicant)
    features_present = [c["feature"] for c in codes]
    assert "smoking & obesity interaction" in features_present
    assert codes[0]["direction"] == "increases cost"

