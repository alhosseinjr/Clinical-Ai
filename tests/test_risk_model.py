"""
Unit tests for the Risk Prediction Agent's ML core (src/utils/risk_model.py).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.risk_model import RiskModel, FEATURE_NAMES, load_dataset


def test_dataset_loads_and_has_expected_shape():
    X, y = load_dataset()
    assert X.shape[1] == len(FEATURE_NAMES)
    assert X.shape[0] > 250  # the real UCI heart disease dataset has 303 rows
    assert set(y.tolist()) == {0, 1}


def test_model_reports_credible_eval_metrics():
    """Guards against silently regressing to a near-random or degenerate model."""
    model = RiskModel()
    metrics = model.eval_metrics
    assert metrics["accuracy"] > 0.7
    assert metrics["roc_auc"] > 0.75
    assert metrics["n_test"] > 0


def test_predict_returns_valid_shape():
    model = RiskModel()
    result = model.predict({name: 0 for name in FEATURE_NAMES})
    assert result["risk_category"] in {"low", "moderate", "high"}
    assert 0.0 <= result["risk_score"] <= 1.0
    assert len(result["top_factors"]) == 3


def test_high_risk_profile_scores_higher_than_low_risk_profile():
    """Feature directions here are verified against the corrected (un-inverted)
    target label -- see data/risk_model/DATASET_INFO.md 'label correction' note.
    Confirmed against an independent mirror of the same Cleveland cohort before
    trusting the direction (age, sex, thalach, exang, oldpeak, ca, thal all
    check out against clinical expectation)."""
    model = RiskModel()
    low_risk = model.predict({
        "age": 32, "sex": 0, "cp": 3, "trestbps": 110, "chol": 170,
        "fbs": 0, "restecg": 0, "thalach": 175, "exang": 0,
        "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 2,
    })
    high_risk = model.predict({
        "age": 68, "sex": 1, "cp": 0, "trestbps": 165, "chol": 290,
        "fbs": 1, "restecg": 2, "thalach": 105, "exang": 1,
        "oldpeak": 2.4, "slope": 2, "ca": 3, "thal": 3,
    })
    assert high_risk["risk_score"] > low_risk["risk_score"]
