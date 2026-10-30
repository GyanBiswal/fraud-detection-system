# Transaction Fraud Detection System

A machine learning project that predicts whether a credit-card transaction is potentially fraudulent. It covers the full path from data exploration to a trained model, a FastAPI backend, and a React dashboard.

> **Demo only.** This project uses a public research dataset. It is not connected to any bank or real-time payment system, and it is not a production fraud detection system.

![Legitimate sample](reports/figures/demo-legit.png)
![Fraud sample](reports/figures/demo-fraud.png)

## Problem

Fraud is rare, so a model that always says "legitimate" looks very accurate while catching nothing. This project focuses on handling that imbalance properly: choosing suitable metrics, avoiding data leakage, and picking a decision threshold deliberately.

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284,807 transactions made by European cardholders over two days, with 492 frauds (0.17%).

- `V1` to `V28`: anonymized features produced by PCA, so their real meaning is unknown.
- `Amount`: transaction amount. `Time`: seconds since the first transaction. `Class`: 1 = fraud, 0 = legitimate.
- The CSV is not included in this repository. Download `creditcard.csv` from Kaggle and place it in `data/`.

## Methodology

1. **Cleaning:** removed 1,081 duplicate rows (they could appear in both train and test sets and inflate scores).
2. **Split:** 80/20 stratified split, so the fraud ratio is the same in both sets. The test set was used only once, at the end.
3. **Preprocessing:** scaled `Amount` with `StandardScaler`, fitted on training data only. Dropped `Time`, because it is a seconds-since-start counter that has no meaning for a new transaction.
4. **Models compared** on a validation set: a baseline that always predicts "legitimate", Logistic Regression, Decision Tree (with and without a depth limit), and Random Forest.
5. **Threshold:** the model outputs a fraud score between 0 and 1. A threshold of 0.3 was chosen on the validation set to catch more fraud at a small cost in false alarms.
6. **Pipeline:** the scaler and model are saved together, so training and prediction use identical preprocessing.

## Results

Validation set (used to compare models):

| Model | Precision | Recall |
|---|---|---|
| Baseline (always "legitimate") | n/a | 0.00 (99.8% accuracy) |
| Logistic Regression | 0.90 | 0.67 |
| Decision Tree, max depth 5 | 0.89 | 0.83 |
| Random Forest (threshold 0.5) | 0.94 | 0.83 |

Final held-out test set (95 frauds in 56,746 transactions), Random Forest with threshold 0.3:

| Precision | Recall | PR-AUC |
|---|---|---|
| 0.89 | 0.77 | 0.79 |

An unrestricted Decision Tree scored a perfect 1.0 recall on training data but only 0.80 on validation, which is overfitting. Limiting its depth removed the gap.

## Limitations

- The features are anonymized, so I can show which features matter but not why.
- The test set has only 95 frauds, so a few more or fewer misses change recall by several percentage points.
- One public dataset from two days in 2013 is not evidence of real-world performance.
- Real fraud patterns change over time; this model has no monitoring or retraining.
- The risk score comes from a Random Forest's votes and is a ranking signal, not a calibrated probability.

## Architecture

```
React (Vite)  --HTTP-->  FastAPI  -->  scikit-learn Pipeline (scaler + Random Forest)
```

The React app is a thin client. All prediction logic lives in the backend.

## Project structure

```
data/           dataset (not committed)
notebooks/      EDA, preprocessing, model comparison
src/            training script and prediction function
api/            FastAPI app
tests/          API tests
frontend/       React dashboard
models/         saved model (not committed)
```

## Run it locally

Requires Python 3.10+ and Node.js.

```bash
# 1. Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Put creditcard.csv in data/, then train and save the model
python src/train.py

# 3. Run the tests
python -m pytest -v

# 4. Start the API (http://127.0.0.1:8000/docs)
uvicorn api.main:app --reload
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The sample transactions in `frontend/src/samples.json` come from the test set and can be regenerated with `python src/make_samples.py`.

## API

`POST /predict` with `{"features": {"V1": ..., "V28": ..., "Amount": ...}}` returns:

```json
{"is_fraud": true, "risk_score": 0.52, "threshold": 0.3}
```

Missing features, negative amounts, and non-numeric values return a 422 error with a clear message.

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, joblib, FastAPI, React, Vite, pytest.