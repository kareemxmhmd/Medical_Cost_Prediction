import pandas as pd
import pytest
from src.data.data_loader import get_train_test_split, load_and_validate_data
from src.features.build_features import get_preprocessor


def test_load_and_validate_data():
    df = load_and_validate_data("data/insurance.csv")
    assert isinstance(df, pd.DataFrame)
    assert "charges" in df.columns
    # 1338 raw rows minus 1 duplicate row = 1337 deduplicated records
    assert len(df) == 1337


def test_get_train_test_split():
    df = load_and_validate_data("data/insurance.csv")
    X_train, X_test, y_train, y_test = get_train_test_split(df, test_size=0.2, random_state=42)
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
    assert "charges" not in X_train.columns


def test_get_preprocessor_pipeline():
    preprocessor, num_features, cat_features = get_preprocessor()
    assert num_features == ["age", "bmi", "children", "is_obese", "obese_smoker"]
    assert cat_features == ["smoker", "region"]
    assert "sex" not in cat_features
    assert "sex" not in num_features

    # Test transformation on sample dataframe with all 4 regions
    sample_df = pd.DataFrame({
        "age": [25, 45, 30, 50],
        "bmi": [22.4, 31.0, 28.0, 35.0],
        "children": [0, 2, 1, 3],
        "smoker": ["no", "yes", "no", "yes"],
        "region": ["northeast", "southwest", "northwest", "southeast"]
    })
    transformed = preprocessor.fit_transform(sample_df)
    # 5 num (3 raw + 2 engineered) + 2 smoker + 4 region = 11 features
    assert transformed.shape[1] == 5 + 2 + 4

    # Verify unknown category is ignored without conflating with first category
    unseen_df = pd.DataFrame({
        "age": [25],
        "bmi": [22.4],
        "children": [0],
        "smoker": ["no"],
        "region": ["unknown_region"]
    })
    unseen_transformed = preprocessor.transform(unseen_df)
    assert unseen_transformed.shape == (1, 11)
    # The 4 region columns (indices 7 to 11) should all be 0 for unknown category
    assert (unseen_transformed[0, 7:11] == 0).all()


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_and_validate_data("data/nonexistent_file.csv")


def test_load_data_missing_columns(tmp_path):
    csv_path = tmp_path / "bad_data.csv"
    csv_path.write_text("age,sex,bmi,children,smoker,region\n25,male,22.0,0,no,northeast\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        load_and_validate_data(str(csv_path))


def test_load_data_null_values(tmp_path):
    csv_path = tmp_path / "null_data.csv"
    csv_path.write_text("age,sex,bmi,children,smoker,region,charges\n25,male,,0,no,northeast,1200.0\n")
    with pytest.raises(ValueError, match="contains null values"):
        load_and_validate_data(str(csv_path))


def test_load_data_invalid_range(tmp_path):
    csv_path = tmp_path / "invalid_range.csv"
    csv_path.write_text("age,sex,bmi,children,smoker,region,charges\n150,male,25.0,0,no,northeast,1200.0\n")
    with pytest.raises(ValueError, match="Age values must be numeric between 18 and 100"):
        load_and_validate_data(str(csv_path))


def test_train_and_evaluate_pipeline(tmp_path):
    from src.models.train_model import train_and_evaluate
    df = load_and_validate_data("data/insurance.csv").sample(200, random_state=42)
    sample_csv = tmp_path / "sample_insurance.csv"
    df.to_csv(sample_csv, index=False)
    
    art_dir = tmp_path / "artifacts"
    rep_dir = tmp_path / "reports"
    
    info = train_and_evaluate(
        data_path=sample_csv,
        artifacts_dir=art_dir,
        reports_dir=rep_dir,
    )
    assert (art_dir / "pricing_model_v1.pkl").exists()
    assert (art_dir / "model_info.json").exists()
    assert (rep_dir / "model_leaderboard.csv").exists()
    assert "metrics" in info
    assert info["metrics"]["r2"] > 0.60
