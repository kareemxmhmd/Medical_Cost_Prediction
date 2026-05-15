# Medical Cost Prediction

Predicting individual medical insurance charges using a **Random Forest Regressor** trained on demographic and lifestyle features.

## Dataset

Insurance dataset — 1,338 records with 7 features:

| Feature    | Description                          |
|------------|--------------------------------------|
| `age`      | Age of the insured                   |
| `sex`      | Gender (male / female)               |
| `bmi`      | Body Mass Index                      |
| `children` | Number of dependents                 |
| `smoker`   | Smoking status (yes / no)            |
| `region`   | Residential region in the US         |
| `charges`  | Medical cost billed (target)         |

## Project Structure

```
├── main.ipynb      # Full pipeline: EDA → preprocessing → modeling → evaluation
├── predict.py      # Inference script for new predictions
├── model.pkl       # Saved Random Forest pipeline
├── insurance.csv   # Dataset
└── README.md
```

## Workflow

1. **EDA** — Distribution analysis, correlation heatmap, outlier detection
2. **Preprocessing** — Scikit-learn `ColumnTransformer` (scaling + one-hot encoding)
3. **Modeling** — Random Forest Regressor via `Pipeline`
4. **Evaluation** — MAE, RMSE, R² scoring with actual vs. predicted plots
5. **Interpretation** — Feature importance ranking and residual analysis

## Results

| Metric | Score |
|--------|-------|
| R²     | ~0.86 |
| MAE    | ~2,500 |

> **Key finding:** `smoker` status is the dominant cost driver, followed by `age` and `bmi`.

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Joblib
