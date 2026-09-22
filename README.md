# Medical Cost Prediction

**Live App:** [medical-cost.app](https://medical-cost-app.vercel.app/)

## 1. The Problem & The Solution

Insurance companies need to estimate how much a new policyholder will cost them in medical charges *before* they set the premium. Doing this manually with flat lookup tables is slow, inconsistent, and misses the non-linear interactions between risk factors (a smoker with high BMI costs far more than either factor alone would suggest).

This project predicts annual medical costs from six basic inputs: age, BMI, number of children, smoking status, sex, and region. The prediction isn't just a number — the API returns everything an underwriter needs to make a decision:

| Output | What It Tells You |
|---|---|
| Predicted Cost (USD) | Point estimate of expected annual charges |
| 80% Prediction Interval | Low–high range from individual trees, so you know how confident the estimate is |
| Premium Tier | Automatic routing: *Standard* / *Loaded* / *Refer to Underwriter* |
| Reason Codes | Top-4 features driving the cost up or down, so the decision is explainable |

The tier thresholds are straightforward:

| Tier | Predicted Cost | Action |
|---|---|---|
| Standard | < $15,000 | Auto-approve, standard premium |
| Loaded | $15,000 – $30,000 | Apply risk loading to premium |
| Refer to Underwriter | ≥ $30,000 | Route to human review |

This way, low-risk applicants get quoted instantly, and underwriter time is reserved for the cases that actually need it.

---

## 2. How the Data Was Handled

Starting from the Medical Cost Personal Dataset (`insurance.csv`) — 1,338 patient records, 7 columns:

1. **No missing values** — the dataset is clean out of the box, no imputation needed.
2. **No duplicates** — all 1,338 records are unique patients.
3. **Feature/target split** — separated `charges` (what we're predicting) from the 6 input features (`age`, `sex`, `bmi`, `children`, `smoker`, `region`).
4. **Train/test split** — 80/20 with a fixed seed (`random_state=42`) so results are reproducible every time.
5. **Encoding is inside the pipeline** — categorical columns (`sex`, `smoker`, `region`) are one-hot encoded via a `ColumnTransformer` that's saved *as part of the model itself*. Numeric columns (`age`, `bmi`, `children`) pass through untouched. This means there's zero chance of train/serve skew — the exact same transformations run during training and prediction.

Result: 1,070 rows for training, 268 rows for testing.

---

## 3. Why These Models

Three models were trained and compared on the same data split:

| Model | MAE (USD) | RMSE (USD) | R² Score |
|---|---|---|---|
| Linear Regression | 4,181.19 | 5,796.28 | 0.7836 |
| XGBoost | 2,861.04 | 4,973.90 | 0.8406 |
| **Random Forest (chosen)** | **2,559.90** | **4,586.94** | **0.8645** |

**Linear Regression** was included as a baseline — if a straight line fits the data well enough, there's no reason to add complexity. It doesn't here, because medical costs are heavily right-skewed and driven by interaction effects (smoker × BMI) that a linear model can't capture.

**XGBoost** is usually the go-to for tabular data. It performed well, but slightly underperformed Random Forest on this dataset. With only 1,338 rows, XGBoost's sequential error-correction doesn't have enough signal to outshine bagging — it typically needs larger datasets to show its edge.

**Random Forest** won because it had the lowest MAE and highest R², meaning it makes the smallest average errors *and* explains the most variance. It also gives us a bonus: since it's an ensemble of 200 trees, we can look at the spread of individual tree predictions to derive a prediction interval — something you can't easily get from a single model.

---

## 4. Output Quality

- **R² of 0.8645** — the model explains 86.5% of the variance in medical costs. The remaining 13.5% comes from factors not in the dataset (pre-existing conditions, lifestyle details, etc.), which is expected.
- **MAE of \$2,560** — on average, predictions are off by about \$2,560. For individual insurance pricing where charges range from \$1,122 to \$63,770, this is well within acceptable actuarial tolerance.
- **Prediction intervals** — beyond the point estimate, the API returns the 10th–90th percentile range across all 200 tree predictions. If the interval is tight, the model is confident. If it's wide, the underwriter knows to apply more scrutiny.
- **Reason codes** — each prediction comes with the top-4 features driving the cost, computed by permutation (zero out each feature and measure the delta). The top driver is consistently `smoker`, which makes real business sense — smokers cost dramatically more.

---

## 5. How the Model Was Trained

```
cd src
python train_model.py
```

This one command runs the full pipeline:
1. Loads `data/insurance.csv` via `src/data_loader.py`.
2. Splits it 80/20 into train/test.
3. Builds a preprocessing pipeline (one-hot encoding for categoricals, passthrough for numerics).
4. Trains all three models (Linear Regression, XGBoost, Random Forest).
5. Automatically picks the best one by R² score.
6. Saves the trained pipeline, feature list, and metrics into `models/`.

Expected output:
```
Loaded 1338 rows, 7 columns
LinearRegression -> mae: 4181.19, rmse: 5796.28, r2: 0.7836
RandomForest     -> mae: 2559.90, rmse: 4586.94, r2: 0.8645
XGBoost          -> mae: 2861.04, rmse: 4973.90, r2: 0.8406

Best model: RandomForest
Saved best model to models/model.pkl
```

To retrain on new data, just replace `data/insurance.csv` and re-run the same command.

---

## 6. How to Use It

### Live App
**https://medical-cost-app.vercel.app/**

No installation needed — open the link and start predicting. Fill in a patient's profile:

- **age** — patient's age
- **sex** — male or female
- **bmi** — Body Mass Index
- **children** — number of dependents covered
- **smoker** — yes or no
- **region** — northeast, northwest, southeast, or southwest

Click predict to instantly get the estimated annual cost, the 80% confidence interval, the premium tier, and the reason codes explaining the prediction.

### Run the API yourself

```
pip install -r requirements.txt
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

- `POST /predict` — send one patient's data, get back predicted cost, interval, tier, and reason codes.
- `GET /health` — check the API is running.
- `GET /model/metadata` — see the model's saved performance metrics.