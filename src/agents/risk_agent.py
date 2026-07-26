"""
3. Risk Prediction Agent (ML)

Runs the patient's structured vitals + cardiac workup data through a
logistic regression model trained on the real UCI Heart Disease dataset
(src/utils/risk_model.py) to produce a risk score, category, and top
contributing factors.

Feature set matches the source dataset (see data/risk_model/DATASET_INFO.md).
Several fields (thalach, exang, oldpeak, slope, ca, thal) come from a
cardiac stress test / catheterization workup rather than a basic intake
visit -- sample_patients.json models these as a `cardiac_workup` block per
patient, representing results already on file. If a patient record has no
`cardiac_workup` block, we fall back to 0 for those fields and flag it in
the trace so it's visible in the report rather than silently treated as a
real "no" answer.
"""

from src.state import PipelineState
from src.utils.risk_model import RiskModel, FEATURE_NAMES
from src.utils.clinical_feature_mapper import map_clinical_features

_model = None

# Fields that realistically come from a stress test / cath workup rather
# than a basic intake visit -- used only to produce an honest trace note
# when they're missing.
_WORKUP_FIELDS = {"thalach", "exang", "oldpeak", "slope", "ca", "thal"}


def _get_model() -> RiskModel:
    global _model
    if _model is None:
        _model = RiskModel()
    return _model


def run(state: PipelineState) -> dict:
    profile = state.get("patient_profile", {})
    vitals = profile.get("vitals", {})
    workup = profile.get("cardiac_workup", {})

    sex_val = 1 if str(profile.get("sex", "")).upper().startswith("M") else 0

    source = {
        "age": profile.get("age"),
        "sex": sex_val,
        "cp": workup.get("chest_pain_type"),
        "trestbps": vitals.get("systolic_bp"),
        "chol": vitals.get("cholesterol_total"),
        "fbs": workup.get("fasting_blood_sugar_high"),
        "restecg": workup.get("resting_ecg"),
        "thalach": workup.get("max_heart_rate"),
        "exang": workup.get("exercise_induced_angina"),
        "oldpeak": workup.get("st_depression"),
        "slope": workup.get("st_slope"),
        "ca": workup.get("num_major_vessels"),
        "thal": workup.get("thalassemia"),
    }

    features = {}
    missing_workup = []

    for name in FEATURE_NAMES:
        val = source.get(name)
        if val is None:
            if name in _WORKUP_FIELDS:
                missing_workup.append(name)
            features[name] = 0
        else:
            features[name] = float(val)

    # Enhance structured features using NLP extraction
    # NOTE: 'risk' runs in parallel with 'nlp', so extracted_entities 
    # might not exist yet. We safely skip this if it's missing or incomplete.
    nlp_features = {}
    entities = state.get("extracted_entities")
    
    if entities:
        try:
            nlp_features = map_clinical_features(entities)
        except Exception:
            # If the mapper fails because entities are incomplete, just use base features
            pass

    for key, value in nlp_features.items():
        if key in features:
            if features[key] == 0:
                features[key] = value

    model = _get_model()
    result = model.predict(features)

    trace = [
        f"[Risk Prediction Agent] Predicted {result['risk_category']} risk "
        f"(score={result['risk_score']}), top factors: {', '.join(result['top_factors'])}."
    ]
    
    errors = []
    if missing_workup:
        trace.append(
            f"[Risk Prediction Agent] No cardiac workup data for: {', '.join(missing_workup)} "
            f"-- defaulted to 0 (treated as 'not present'), not as 'unknown'. Risk score may "
            f"be understated for this patient."
        )

    return {"risk_result": result, "trace": trace, "errors": errors}