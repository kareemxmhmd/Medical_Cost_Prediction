from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
VALID_SMOKER = {"yes", "no"}
VALID_REGIONS = {"northeast", "northwest", "southeast", "southwest"}


def load_and_validate_data(filepath: str | Path) -> pd.DataFrame:
    """Loads dataset, validates schema and domain constraints, and deduplicates records.
    
    Args:
        filepath: Path to raw or processed insurance claims CSV.
        
    Returns:
        pd.DataFrame: Validated and deduplicated dataframe.
        
    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If mandatory schema columns are missing, null values exist,
                    or values breach domain bounds.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path.resolve()}")
        
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {sorted(missing)}")

    # Null value verification
    if df.isna().any().any():
        null_counts = df.isna().sum().to_dict()
        raise ValueError(f"Dataset contains null values: {null_counts}")

    # Domain value range validation
    if not pd.api.types.is_numeric_dtype(df["age"]) or (df["age"] < 18).any() or (df["age"] > 100).any():
        raise ValueError("Age values must be numeric between 18 and 100.")
    if not pd.api.types.is_numeric_dtype(df["bmi"]) or (df["bmi"] < 10.0).any() or (df["bmi"] > 70.0).any():
        raise ValueError("BMI values must be numeric between 10.0 and 70.0.")
    if not pd.api.types.is_numeric_dtype(df["children"]) or (df["children"] < 0).any() or (df["children"] > 20).any():
        raise ValueError("Children count must be numeric between 0 and 20.")
    if not pd.api.types.is_numeric_dtype(df["charges"]) or (df["charges"] < 0.0).any():
        raise ValueError("Charges must be non-negative numeric values.")

    invalid_smokers = set(df["smoker"].unique()) - VALID_SMOKER
    if invalid_smokers:
        raise ValueError(f"Invalid smoker values: {invalid_smokers}")

    invalid_regions = set(df["region"].unique()) - VALID_REGIONS
    if invalid_regions:
        raise ValueError(f"Invalid region values: {invalid_regions}")

    # Deduplicate identical records to prevent train/test partition leakage
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def get_train_test_split(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits dataframe into train and holdout evaluation partitions.
    
    Performs stratified partitioning on smoker status to preserve critical 
    sub-population risk distribution across splits.
    """
    X = df.drop(columns=["charges"])
    y = df["charges"]
    
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=df["smoker"]
    )

