# LoanGuard AI — Credit Risk Assessment System

A production-grade ML pipeline that predicts the probability of a borrower defaulting on a loan, converts it to a 300–850 credit score, and logs everything to MLflow.

---

## What It Does

Takes a loan application as input and outputs:
- **Default probability** (0 to 1)
- **Risk score** (300 to 850, higher = safer)
- **Decision** — APPROVE / REVIEW / DECLINE
- Full audit trail in MLflow

---

## Project Structure

```
loanguard/
├── config/
│   └── settings.py          # All configuration via environment variables
├── data/
│   ├── generator.py         # Synthetic dataset generation (50k samples)
│   └── loader.py            # Load CSV/Parquet, split into train/val/test
├── ml/
│   ├── features.py          # WoE encoding, IV filtering, derived features
│   ├── pipeline.py          # Full sklearn pipeline (imputer → WoE → XGBoost)
│   ├── evaluate.py          # AUC, KS, Gini, calibration, scorecard metrics
│   └── train.py             # Main training script (Optuna + MLflow)
├── models/                  # Saved model artifacts (.pkl)
├── mlruns/                  # MLflow experiment tracking
└── credit_risk_dataset.csv  # Your input dataset
```

---

## Pipeline Steps

1. **Load data** — your CSV or auto-generated synthetic data
2. **Feature engineering** — creates ratio features like `loan_amount / income`
3. **IV filtering** — drops features with low predictive power (IV < 0.02)
4. **WoE encoding** — converts raw values to risk scores
5. **Imputation + Scaling** — handles missing values, normalises features
6. **Optuna tuning** — runs 50 trials to find best XGBoost hyperparameters
7. **Final training** — trains champion (XGBoost) + challenger (LightGBM)
8. **Evaluation** — AUC-ROC, KS Statistic, Gini, F1, calibration curve
9. **Save** — versioned `.pkl` artifact + full MLflow log

---

## Quickstart

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Train on your dataset**
```bash
export PYTHONPATH=/path/to/loanguard
python3 ml/train.py --data-path credit_risk_dataset.csv
```

**3. Train on synthetic data (no dataset needed)**
```bash
python3 ml/train.py
```

**4. View results in MLflow**
```bash
mlflow ui --backend-store-uri mlruns --port 5000
```
Then open `http://localhost:5000` in your browser.

---

## Key Features Selected (from your run)

| Feature | IV | Strength |
|---|---|---|
| rate_x_dti (interest rate × debt-to-income) | 0.49 | Strong |
| debt_to_income | 0.39 | Strong |
| grade | 0.24 | Medium |
| credit_utilization_ratio | 0.20 | Medium |
| interest_rate | 0.11 | Medium |
| num_derog_records | 0.10 | Medium |

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| AUC-ROC | Overall discrimination ability (target > 0.75) |
| KS Statistic | Max separation between defaulters and non-defaulters |
| Gini Coefficient | 2 × AUC − 1 |
| F1 Score | Balance of precision and recall |
| PSI | Whether score distribution has shifted (< 0.1 = stable) |

---

## Scorecard

Probability is converted to a score using the standard PDO formula:

```
score = base_score - factor × log(p / 1-p)
```

- Score 300 = very high risk
- Score 850 = very low risk
- Mirrors FICO convention

---

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| LOANGUARD_OPTUNA_N_TRIALS | 50 | Number of tuning trials |
| LOANGUARD_CV_FOLDS | 5 | Cross-validation folds |
| LOANGUARD_TEST_SIZE | 0.20 | Fraction held out for testing |
| LOANGUARD_MLFLOW_TRACKING_URI | mlruns | MLflow backend |
| LOANGUARD_IV_FILTER_THRESHOLD | 0.02 | Minimum IV to keep a feature |

---

## Tech Stack

- **XGBoost** — champion model
- **LightGBM** — challenger model  
- **Optuna** — hyperparameter tuning
- **MLflow** — experiment tracking
- **scikit-learn** — pipeline and calibration
- **Pandas / NumPy** — data processing
