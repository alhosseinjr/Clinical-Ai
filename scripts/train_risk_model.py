#!/usr/bin/env python3
"""
Trains the Risk Prediction Agent's logistic regression model on the real
UCI Heart Disease dataset and writes a human-readable evaluation report.

Usage:
    python scripts/train_risk_model.py [--force]

--force retrains even if models/risk_model.joblib already exists (e.g.
after changing the dataset or the feature set).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib

from src.utils.risk_model import (
    train_and_evaluate, MODEL_PATH, DATA_PATH, FEATURE_NAMES,
)

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "risk_model_eval.md",
)


def render_report(metrics: dict) -> str:
    cm = metrics["confusion_matrix"]
    return f"""# Risk Model Evaluation Report

**Dataset:** UCI Heart Disease (Cleveland subset), `{os.path.relpath(DATA_PATH)}`
See `data/risk_model/DATASET_INFO.md` for column definitions and honest
scope notes (this is a demo-scale, single-institution, ~40-year-old
dataset -- not a clinically validated model). Note: the original source
mirror had an inverted target label, corrected here after cross-checking
feature correlations against an independent mirror -- see DATASET_INFO.md
"Data quality note" section.

**Model:** Logistic Regression (scikit-learn), StandardScaler-normalized,
stratified 80/20 train/test split, `random_state=42`.

**Features ({len(FEATURE_NAMES)}):** {", ".join(FEATURE_NAMES)}

## Held-out test set results

| Metric | Value |
|---|---|
| Train size | {metrics['n_train']} |
| Test size | {metrics['n_test']} |
| Accuracy | {metrics['accuracy']} |
| Precision | {metrics['precision']} |
| Recall | {metrics['recall']} |
| F1 | {metrics['f1']} |
| ROC-AUC | {metrics['roc_auc']} |

**Confusion matrix** (rows = actual, cols = predicted, `[no disease, disease]`):

```
              pred_no   pred_yes
actual_no       {cm[0][0]:>5}      {cm[0][1]:>5}
actual_yes      {cm[1][0]:>5}      {cm[1][1]:>5}
```

## How to read this

These numbers are what you'd quote for the project, not marketing copy.
An accuracy in the low-to-mid 80s on a 303-row, decades-old, single-site
dataset is in line with published results on this exact benchmark -- it
demonstrates the pipeline can train and evaluate a real model properly, not
that this model is fit for clinical use.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Retrain even if a saved model exists.")
    args = parser.parse_args()

    if args.force and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    print(f"Training on {DATA_PATH} ...")
    pipeline, metrics = train_and_evaluate()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metrics": metrics}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    report = render_report(metrics)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Eval report written to {OUTPUT_PATH}\n")
    print(report)


if __name__ == "__main__":
    main()
