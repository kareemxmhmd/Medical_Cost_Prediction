import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from data_loader import load_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.pkl"


def split_features_target(df, target_col="charges"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def get_preprocessor(X):
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])


MODELS = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
}


def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    results = {}

    for name, reg in MODELS.items():
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', reg)
        ])
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)

        print(f"{name} -> mae: {mae:.2f}, rmse: {rmse:.2f}, r2: {r2:.4f}")
        results[name] = {"model": model, "mae": mae, "rmse": rmse, "r2": r2}

    return results


def get_best_model(results):
    best_name = max(results, key=lambda name: results[name]["r2"])
    best = results[best_name]
    print(f"\nBest model: {best_name}")
    return best_name, best


def main():
    df = load_data()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = get_preprocessor(X)
    results = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
    best_name, best = get_best_model(results)

    joblib.dump(best["model"], MODEL_PATH)
    joblib.dump(X.columns.tolist(), COLUMNS_PATH)
    joblib.dump(
        {"model_name": best_name, "mae": best["mae"], "rmse": best["rmse"], "r2": best["r2"]},
        METRICS_PATH,
    )

    print(f"Saved best model to {MODEL_PATH}")
    print(f"Saved feature columns to {COLUMNS_PATH}")


if __name__ == "__main__":
    main()