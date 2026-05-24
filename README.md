# Medical Cost Prediction

Predicting individual medical insurance charges using a **Random Forest Regressor** — built as an end-to-end ML project covering EDA, preprocessing, modeling, evaluation, and inference.

---

## Overview

Medical insurance costs vary widely based on personal factors like age, BMI, smoking status, and region. This project builds a predictive model that estimates annual insurance charges for an individual, given those inputs.

The full pipeline goes from raw data exploration all the way to a saved model with an interactive inference script — meaning you can actually run predictions on new inputs without touching the notebook.

---

## Project Structure

```
Medical_Cost_Prediction/
│
├── main.ipynb        # Full pipeline: EDA → preprocessing → modeling → evaluation
├── predict.py        # Inference script — run predictions from the terminal
├── model.pkl         # Saved Random Forest pipeline (preprocessing + model)
├── insurance.csv     # Raw dataset
└── README.md
```

---

## Technical Approach

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis for all numerical features (`age`, `bmi`, `charges`)
- Correlation heatmap to identify relationships with the target
- Outlier detection — `charges` is right-skewed, driven heavily by smokers
- Visual breakdown of cost by smoking status, region, and age group

### 2. Preprocessing
Built a scikit-learn `ColumnTransformer` inside a `Pipeline` to handle:
- **Numerical features** → `StandardScaler`
- **Categorical features** → `OneHotEncoder` (handles `sex`, `smoker`, `region`)

Using a Pipeline ensures preprocessing is part of the saved model — no data leakage, no manual re-transformation at inference time.

### 3. Modeling
- **Algorithm:** `RandomForestRegressor`
- **Why Random Forest?** Handles mixed feature types well, robust to outliers, and gives feature importance out of the box — useful for interpreting what actually drives costs.
- Train/test split: 80/20

### 4. Evaluation
Evaluated on the held-out test set using three regression metrics:

| Metric | What it tells you |
|---|---|
| **R²** | How much variance the model explains (1.0 = perfect) |
| **MAE** | Average absolute prediction error in dollars |
| **RMSE** | Penalizes large errors more heavily than MAE |

### 5. Interpretation
- Feature importance ranking from the trained forest
- Residual plot (actual vs. predicted) to check for systematic bias
- Confirmed that the model underperforms on extreme high-cost cases (outlier smokers)

---

## Results

| Metric | Score |
|---|---|
| **R²** | **0.86** |
| **MAE** | ~$2,500 |
| **RMSE** | ~$4,800 |

**The model explains 86% of the variance in insurance charges** — strong performance given the small dataset and no hyperparameter tuning beyond defaults.

> **Key finding:** `smoker` status is by far the dominant cost driver, followed by `age` and `bmi`. Region and number of children have minimal predictive weight.
