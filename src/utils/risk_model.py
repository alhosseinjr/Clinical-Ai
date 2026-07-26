"""
Risk Prediction Agent's ML core (Upgraded to Random Forest).

Uses Random Forest to capture non-linear relationships between clinical features.
Trained on the UCI Heart Disease dataset. Stable on Apple Silicon.
"""

import os
import joblib
import numpy as np
import pandas as pd
from src.state import PipelineState
from src.utils.clinical_feature_mapper import map_clinical_features
from sklearn.ensemble import RandomForestClassifier
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
    """Trains a Random Forest model on a stratified 80/20 split."""
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Random Forest Pipeline (n_jobs=-1 uses all Mac cores for speed)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100, 
            max_depth=5, 
            random_state=RANDOM_STATE, 
            n_jobs=-1
        )),
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
            # Safely attempt to enhance features with NLP entities
        entities = patient.get("extracted_entities", {})
        if entities:
            try:
                # Pass both entities and the patient profile (the patient dict itself)
                mapped_features = map_clinical_features(entities, patient)
                # Update the patient dict with the mapped features
                patient.update(mapped_features)
            except Exception:
                # If mapping fails, we safely fall back to base vitals
                pass

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

        # Random Forest uses Feature Importance (Gini impurity decrease)
        importances = self.clf.feature_importances_
        top_idx = np.argsort(-importances)[:3]

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