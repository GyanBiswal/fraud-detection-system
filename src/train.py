from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
THRESHOLD = 0.3

# 1. Load and clean
df = pd.read_csv(ROOT / "data" / "creditcard.csv").drop_duplicates()
X = df.drop(columns=["Class", "Time"])
y = df["Class"]

# 2. Same split as the notebook
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Build the pipeline: scale Amount, then Random Forest
preprocess = ColumnTransformer(
    [("scale_amount", StandardScaler(), ["Amount"])],
    remainder="passthrough",
)
pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )),
])
pipeline.fit(X_train, y_train)

# 4. Quick check that the saved model matches the notebook result
probs = pipeline.predict_proba(X_test)[:, 1]
preds = (probs >= THRESHOLD).astype(int)
print("Precision:", round(precision_score(y_test, preds), 3))
print("Recall:", round(recall_score(y_test, preds), 3))
print("PR-AUC:", round(average_precision_score(y_test, probs), 3))

# 5. Save everything the app needs in one file
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(
    {"pipeline": pipeline, "threshold": THRESHOLD, "features": list(X.columns)},
    ROOT / "models" / "fraud_model.joblib",
)
print("Saved models/fraud_model.joblib")