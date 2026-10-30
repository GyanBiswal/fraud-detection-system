from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fraud_model.joblib"


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict_transaction(transaction, bundle=None):
    bundle = bundle or load_model()
    features = bundle["features"]

    missing = [f for f in features if f not in transaction]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    row = pd.DataFrame([transaction])[features]
    score = float(bundle["pipeline"].predict_proba(row)[0, 1])

    return {
        "is_fraud": score >= bundle["threshold"],
        "risk_score": round(score, 4),
        "threshold": bundle["threshold"],
    }


if __name__ == "__main__":
    root = MODEL_PATH.parent.parent
    df = pd.read_csv(root / "data" / "creditcard.csv")
    features = load_model()["features"]

    fraud_row = df[df["Class"] == 1].iloc[0][features].to_dict()
    normal_row = df[df["Class"] == 0].iloc[0][features].to_dict()

    print("Known fraud transaction ->", predict_transaction(fraud_row))
    print("Known normal transaction ->", predict_transaction(normal_row))