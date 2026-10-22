
#This file handles preprocessing, model training, 
#evaluation, comparison, and saving the best model.

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

DATA_FILE = Path("data/ai4i2020.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Dataset not found. Run: python download_data.py"
    )

df = pd.read_csv(DATA_FILE)

target = "Machine failure"

# Remove identifiers and target-related failure-mode columns.
# These columns must not be used as input features.
drop_columns = [
    "UDI",
    "Product ID",
    target,
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

X = df.drop(
    columns=[c for c in drop_columns if c in df.columns]
)
y = df[target].astype(int)

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
trained_models = {}

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    results.append(metrics)
    trained_models[name] = pipeline

    print(json.dumps(metrics, indent=2))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

# Prioritize F1-score for this imbalanced classification task.
# Inspect recall and precision as well before choosing a model.
best_result = max(results, key=lambda item: item["f1"])
best_name = best_result["model"]
best_model = trained_models[best_name]

joblib.dump(best_model, MODEL_DIR / "model.joblib")

with open(MODEL_DIR / "metrics.json", "w") as file:
    json.dump({
        "selected_model": best_name,
        "features": X.columns.tolist(),
        "metrics": results,
        "test_size": 0.20,
        "random_state": 42,
    }, file, indent=2)

print("\n" + "=" * 45)
print("TRAINING COMPLETE")
print("Selected model:", best_name)
print("Saved model: models/model.joblib")
print("Saved metrics: models/metrics.json")
print("=" * 45)
