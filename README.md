# Medical Cost Prediction & Underwriting Web Application

A production-grade machine learning web application for actuarial healthcare cost prediction, uncertainty quantification, and underwriting risk assessment. Built with **LightGBM**, **FastAPI**, and **Next.js + React**.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Overview](#project-overview)
- [Machine Learning Approach](#machine-learning-approach)
  - [Champion Model Architecture](#champion-model-architecture)
  - [Uncertainty Quantification (80% Prediction Intervals)](#uncertainty-quantification-80-prediction-intervals)
  - [SHAP Explainability & Reason Codes](#shap-explainability--reason-codes)
  - [Regulatory Compliance (ACA § 2701)](#regulatory-compliance-aca--2701)
  - [Underwriting Risk Policy](#underwriting-risk-policy)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
  - [Authentication](#authentication)
  - [Endpoints](#endpoints)
  - [Example Request & Response](#example-request--response)
- [Frontend Web Application](#frontend-web-application)
- [Local Development Setup](#local-development-setup)
  - [1. Backend (FastAPI)](#1-backend-fastapi)
  - [2. Frontend (Next.js)](#2-frontend-nextjs)
- [Docker & Containerized Deployment](#docker--containerized-deployment)
- [Automated Testing & Governance](#automated-testing--governance)
- [Continuous Integration (CI)](#continuous-integration-ci)

---

## Problem Statement

Accurate pricing of health insurance policies requires estimating anticipated individual healthcare expenditures while quantifying predictive uncertainty. Traditional linear pricing models fail to capture complex non-linear risk interactions (such as the combined impact of smoking and high BMI), leading to adverse selection or mispriced coverage. Furthermore, actuarial workflows require transparent reason codes to explain automated risk ratings to underwriters and applicants.

---

## Project Overview

This system transforms raw applicant demographic and clinical characteristics into:
1. **Expected Annual Medical Costs**: Point prediction of medical expenditures.
2. **80% Prediction Intervals**: Calibrated lower (10th percentile) and upper (90th percentile) cost boundaries preventing quantile crossing.
3. **Automated Risk Tiers**: Categorization into `Standard`, `Loaded`, or `Refer to Underwriter` based on policy thresholds.
4. **SHAP Feature Attribution**: Dollar-denominated reason codes explaining key drivers behind individual predictions.
5. **Drift Governance & Audit Trail**: Real-time statistical drift monitoring (KS-test, $\chi^2$, PSI) and an immutable SQLite audit log.

---

## Machine Learning Approach

### Champion Model Architecture
- **Algorithm**: LightGBM Regressor with `regression_l1` (Mean Absolute Error) objective.
- **Benchmark Evaluation**: Evaluated against Linear Regression and Random Forest Regressors via 5-fold cross-validation on training partitions without test leakage.
- **Performance Metrics**:
  - Holdout MAE: **$1,201.55**
  - Holdout RMSE: **$3,367.99**
  - Coefficient of Determination ($R^2$): **0.9213**

### Uncertainty Quantification (80% Prediction Intervals)
- Dual LightGBM quantile regression models trained with pinball loss at $\alpha = 0.10$ and $\alpha = 0.90$.
- Monotonic boundary clamping is applied during inference to prevent quantile crossing:
  $$\text{lower} = \max(0, \min(\text{pred}_{10}, \text{pred}_{\text{mid}}))$$
  $$\text{upper} = \max(\text{pred}_{90}, \text{pred}_{\text{mid}})$$
- **Observed Empirical Test Coverage**: **83.96%** (calibrated within the target $[75\%, 90\%]$ range).

### SHAP Explainability & Reason Codes
- TreeSHAP (`shap.TreeExplainer`) computes exact marginal contributions for every feature in dollar values.
- One-hot encoded features (e.g., region and smoking indicators) and engineered terms are aggregated into human-readable reason codes highlighting whether each factor increases or decreases expected expense.

### Regulatory Compliance (ACA § 2701)
In accordance with Section 2701 of the Affordable Care Act (ACA), protected demographic attributes—specifically biological sex/gender—are strictly excluded from the model feature set. Permissible rating variables utilized:
- **Age** (18–100 years)
- **Body Mass Index (BMI)** (10.0–60.0 kg/m²)
- **Number of Dependents / Children** (0–20)
- **Tobacco Smoker Status** (`yes`, `no`)
- **Geographic Rating Area / Region** (`northeast`, `northwest`, `southeast`, `southwest`)

### Underwriting Risk Policy
Configured in [`config/underwriting_policy.json`](file:///C:/Users/dxt/Desxtop/ML_Engineer/Medical_Cost_Prediction/config/underwriting_policy.json):
- **Standard**: Expected Cost $\le \$16,640.00$ (automated policy issuance eligible).
- **Loaded**: $\$16,640.00 < \text{Cost} \le \$41,180.00$ (risk surcharge applied).
- **Refer to Underwriter**: $\text{Cost} > \$41,180.00$ (manual actuarial review required).

---

## System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                 Single-Page Web Application                 │
│                   (Next.js 14 + React 18)                   │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / JSON (X-API-Key)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI REST Service                    │
│             (CORS Enabled, OpenAPI/Swagger Docs)            │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────────┐ ┌────────────────────────────┐
│ UnderwritingInferenceEngine  │ │    SQLite Audit Logger     │
│ (LightGBM + Quantile + SHAP) │ │   (audit_predictions.db)   │
└──────────────┬───────────────┘ └────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Model Artifacts        │
│  - pricing_model_v1.pkl      │
│  - model_info.json           │
│  - underwriting_policy.json  │
└──────────────────────────────┘
```

---

## Project Structure

```text
Medical_Cost_Prediction/
├── .github/
│   └── workflows/
│       └── ci.yml                         # Automated CI pipeline (pytest, governance, docker)
├── artifacts/
│   ├── model_info.json                    # Serialized metrics and hyperparameters
│   └── pricing_model_v1.pkl               # Champion LightGBM models & preprocessor
├── config/
│   └── underwriting_policy.json           # Policy thresholds and quantile parameters
├── data/
│   └── insurance.csv                      # Baseline reference dataset
├── frontend/
│   ├── app/
│   │   ├── globals.css                    # Tailwind CSS configuration
│   │   ├── layout.tsx                     # Root page layout
│   │   └── page.tsx                       # Single-page quote calculator UI
│   ├── public/                            # Static assets
│   ├── Dockerfile                         # Multi-stage production container build
│   ├── package.json                       # Frontend dependencies
│   ├── tailwind.config.ts                 # Styling configuration
│   └── tsconfig.json                      # TypeScript configuration
├── reports/
│   ├── figures/                           # Exploratory data analysis charts
│   └── model_leaderboard.csv              # CV benchmark results
├── src/
│   ├── api/
│   │   ├── schemas.py                     # Pydantic request/response schemas
│   │   └── server.py                      # FastAPI REST application
│   ├── data/
│   │   └── data_loader.py                 # Dataset loading and validation logic
│   ├── features/
│   │   └── build_features.py              # Actuarial feature engineering pipeline
│   ├── models/
│   │   ├── inference.py                   # Unified inference & SHAP explanation engine
│   │   └── train_model.py                 # Model training and artifact persistence
│   ├── monitoring/
│   │   ├── audit_logger.py                # SQLite prediction audit logging
│   │   └── drift_monitor.py               # Statistical drift monitor (KS, Chi2, PSI)
│   ├── eda.py                             # Exploratory data analysis generation
│   └── run_all.py                         # End-to-end pipeline orchestration runner
├── tests/
│   ├── test_api.py                        # FastAPI integration tests
│   ├── test_data_features.py              # Data pipeline and feature tests
│   ├── test_inference_engine.py           # Inference, quantile, and reason code tests
│   └── test_monitoring.py                 # Drift and audit logger tests
├── audit_predictions.db                   # SQLite inference audit trail database
├── Dockerfile                             # FastAPI container build
├── docker-compose.yml                     # Multi-service local composition (API + Web)
├── pytest.ini                             # Pytest configuration
├── requirements.txt                       # Python dependencies
└── README.md                              # Technical documentation
```

---

## API Documentation

### Authentication
Protected endpoints require the `X-API-Key` HTTP header:
- Default development key: `dev-api-key-change-in-production`
- Configurable via the `API_KEY` environment variable.

### Endpoints

| Method | Endpoint | Description | Auth Required |
|---|---|---|:---:|
| `GET` | `/health` | Service and model readiness check | No |
| `GET` | `/model-info` | Model metadata, training date, and CV metrics | Yes |
| `POST` | `/predict` | Score single applicant profile with reason codes | Yes |
| `POST` | `/predict/batch` | Score batch of applicants (up to 100 profiles) | Yes |
| `POST` | `/drift` | Evaluate covariate drift against baseline data | Yes |
| `GET` | `/drift` | Evaluate drift directly from the production audit log | Yes |
| `GET` | `/audit-logs` | Retrieve recent inference audit records | Yes |

Interactive Swagger documentation is accessible at `http://localhost:8000/docs`.

### Example Request & Response

#### Request (`POST /predict`):
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key-change-in-production" \
  -d '{
    "age": 35,
    "bmi": 28.5,
    "children": 1,
    "smoker": "no",
    "region": "southwest"
  }'
```

#### Response (`200 OK`):
```json
{
  "predicted_cost": 5195.27,
  "interval_80": [4691.33, 8998.66],
  "premium_tier": "Standard",
  "reason_codes": [
    {
      "feature": "smoker",
      "impact": 3414.11,
      "direction": "decreases cost"
    },
    {
      "feature": "age",
      "impact": 1483.24,
      "direction": "decreases cost"
    },
    {
      "feature": "smoking & obesity interaction",
      "impact": 1210.67,
      "direction": "decreases cost"
    }
  ],
  "model_version": "v1.0",
  "decision_timestamp": "2026-09-10T04:43:12.634792+00:00"
}
```

---

## Frontend Web Application

The frontend is a lightweight, single-page application built with **Next.js 14** and **Tailwind CSS**. It provides a frictionless experience for end users:
- **Intuitive Inputs**: Validated form controls for age, BMI, dependents, smoker status, and geographic region.
- **Real-Time Actuarial Quotes**: Instant display of expected annual cost, calibrated 80% prediction interval, and underwriting tier badge (`Standard`, `Loaded`, `Refer to Underwriter`).
- **Explainability**: Clear visual breakdown of top risk drivers derived from TreeSHAP values.
- **Zero Configuration**: Reads `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_API_KEY` from environment variables, defaulting to local development settings.

---

## Local Development Setup

### Prerequisites
- Python 3.11+
- Node.js 20+ and npm

### 1. Backend (FastAPI)
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run the FastAPI application
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```
The API will be available at `http://localhost:8000`.

### 2. Frontend (Next.js)
```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start the Next.js development server
npm run dev
```
Open `http://localhost:3000` in your browser.

---

## Docker & Containerized Deployment

Run both the FastAPI backend and Next.js frontend with a single command:

```bash
docker compose up --build
```

- **Web Frontend**: `http://localhost:3000`
- **FastAPI Backend**: `http://localhost:8000`
- **Interactive OpenAPI Docs**: `http://localhost:8000/docs`

To stop the containers:
```bash
docker compose down
```

---

## Automated Testing & Governance

Run the comprehensive pytest suite covering API endpoints, data pipelines, model inference, prediction intervals, SHAP reason codes, and drift monitoring:

```bash
python -m pytest tests/ -v
```

All 32 test suites pass with 100% success.

To verify the Next.js frontend build and TypeScript types:
```bash
cd frontend
npm run build
```

---

## Continuous Integration (CI)

The GitHub Actions workflow ([`.github/workflows/ci.yml`](file:///C:/Users/dxt/Desxtop/ML_Engineer/Medical_Cost_Prediction/.github/workflows/ci.yml)) validates every pull request and push:
1. **Code Quality & Tests**: Runs complete pytest suite with code coverage.
2. **Frontend Build**: Builds Next.js application and validates TypeScript types.
3. **Model Governance Audit**: Verifies model metadata thresholds ($MAE \le \$1,800$, $R^2 \ge 0.85$, $0.75 \le \text{Coverage}_{80} \le 0.90$).
4. **Docker Verification**: Builds container images for both backend and frontend.
