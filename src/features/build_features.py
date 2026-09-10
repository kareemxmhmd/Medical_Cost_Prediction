import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_NUMERICAL_FEATURES = ["age", "bmi", "children"]
ENGINEERED_NUMERICAL_FEATURES = ["is_obese", "obese_smoker"]
NUMERICAL_FEATURES = RAW_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES
CATEGORICAL_FEATURES = ["smoker", "region"]


class ActuarialFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineers actuarial risk factors: obesity classification and smoking-obesity interaction."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=["age", "bmi", "children", "smoker", "region"])
            
        is_obese = (df["bmi"] >= 30.0).astype(float)
        is_smoker = (df["smoker"].astype(str).str.lower() == "yes").astype(float)
        
        df["is_obese"] = is_obese
        df["obese_smoker"] = is_obese * is_smoker
        return df


class PreprocessingPipeline(Pipeline):
    """Pipeline exposing named_transformers_ for backward compatibility with existing tests."""

    @property
    def named_transformers_(self):
        return self.named_steps["transform"].named_transformers_


def get_preprocessor() -> tuple[PreprocessingPipeline, list[str], list[str]]:
    """Builds the feature preprocessing pipeline for underwriting cost modeling.
    
    Protected demographic attributes (e.g. sex/gender) are excluded per 
    Affordable Care Act (ACA) § 2701 non-discrimination rating mandates.
    
    Returns:
        tuple containing:
            - PreprocessingPipeline: Configured preprocessing pipeline
            - list[str]: Numerical feature names (including engineered features)
            - list[str]: Categorical feature names
    """
    col_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    
    pipeline = PreprocessingPipeline(
        steps=[
            ("engineer", ActuarialFeatureEngineer()),
            ("transform", col_transformer),
        ]
    )
    return pipeline, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

