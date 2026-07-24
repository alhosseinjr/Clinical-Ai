"""
Risk Prediction Agent's ML core.

Trains a logistic regression model on the real, public UCI Heart Disease
dataset (Cleveland Clinic subset -- see data/risk_model/DATASET_INFO.md for
provenance and column definitions). This replaced an earlier version that
trained on data generated at runtime from a hand-authored scoring rule;
that version could never report a meaningful accuracy number because the
"ground truth" was the same formula being fit. This version trains on real
labeled outcomes with a held-out test set, so the reported metrics mean
something.

Still a demo-scale model: 303 rows, single institution, ~40 years old.
Reported honestly in `outputs/risk_model_eval.md` (run
`python scripts/train_risk_model.py` to regenerate it).
"""

import os
import joblib
import numpy as np
import pandas as pd
from src.state import PipelineState
from src.utils.clinical_feature_mapper import map_clinical_features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(_ROOT, "data", "risk_model", "heart_disease.csv")
MODEL_PATH = os.path.join(_ROOT, "models", "risk_model.joblib")

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

FEATURE_LABELS = {
    "age": "age", "sex": "sex", "cp": "chest pain type",
    "trestbps": "resting systolic BP", "chol": "cholesterol",
    "fbs": "fasting blood sugar", "restecg": "resting ECG",
    "thalach": "max heart rate", "exang": "exercise-induced angina",
    "oldpeak": "ST depression", "slope": "ST segment slope",
    "ca": "major vessels (fluoroscopy)", "thal": "thalassemia result",
}

RANDOM_STATE = 42


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_NAMES].astype(float).values
    y = df["target"].astype(int).values
    return X, y


def train_and_evaluate():
    """Trains on a stratified 80/20 split and returns (pipeline, metrics_dict)."""
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred), 3),
        "recall": round(recall_score(y_test, y_pred), 3),
        "f1": round(f1_score(y_test, y_pred), 3),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 3),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return pipeline, metrics


def load_or_train():
    """Loads a persisted model if present, otherwise trains and persists one."""
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        return bundle["pipeline"], bundle["metrics"]

    pipeline, metrics = train_and_evaluate()
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metrics": metrics}, MODEL_PATH)
    return pipeline, metrics


class RiskModel:
    def __init__(self):
        self.pipeline, self.eval_metrics = load_or_train()
        self.clf = self.pipeline.named_steps["clf"]
        self.scaler = self.pipeline.named_steps["scaler"]

    def predict(self, patient: dict) -> dict:
        """
        Predict cardiovascular risk.

        Accepts:
        1. Already mapped numeric features
        2. Patient state containing extracted_entities from NLP agent

        If extracted_entities exist, they are converted into model features
        using clinical_feature_mapper.
        """

        # Convert NLP entities -> ML features
        if "extracted_entities" in patient:
            from src.utils.clinical_feature_mapper import map_clinical_features
            patient = map_clinical_features(patient)

        features = np.array(
            [[patient.get(name, 0) for name in FEATURE_NAMES]],
            dtype=float
        )

        proba = self.pipeline.predict_proba(features)[0][1]

        category = (
            "high"
            if proba >= 0.66
            else "moderate"
            if proba >= 0.33
            else "low"
        )

        # Explain prediction using model coefficients
        scaled = self.scaler.transform(features)[0]

        contributions = scaled * self.clf.coef_[0]

        top_idx = np.argsort(-np.abs(contributions))[:3]

        top_factors = [
            FEATURE_LABELS[FEATURE_NAMES[i]]
            for i in top_idx
        ]

        return {
            "risk_score": round(float(proba), 3),
            "risk_category": category,
            "top_factors": top_factors,
            "model_features_used": {
                FEATURE_NAMES[i]: float(features[0][i])
                for i in range(len(FEATURE_NAMES))
            }
        }
