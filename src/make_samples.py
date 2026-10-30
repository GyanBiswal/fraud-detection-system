import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "data" / "creditcard.csv").drop_duplicates()
X = df.drop(columns=["Class", "Time"])
y = df["Class"]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

fraud = X_test[y_test == 1].sample(3, random_state=7)
normal = X_test[y_test == 0].sample(3, random_state=7)

samples = []
for i, (_, row) in enumerate(normal.iterrows(), start=1):
    samples.append({
        "name": f"Sample {i}",
        "actual": "legitimate",
        "features": {k: round(float(v), 6) for k, v in row.items()},
    })
for i, (_, row) in enumerate(fraud.iterrows(), start=len(samples) + 1):
    samples.append({
        "name": f"Sample {i}",
        "actual": "fraud",
        "features": {k: round(float(v), 6) for k, v in row.items()},
    })

out = ROOT / "frontend" / "src" / "samples.json"
out.write_text(json.dumps(samples, indent=2))
print("Saved", len(samples), "samples to", out)