"""
1. Patient Intake Agent

Normalizes a raw patient record (however it was loaded -- JSON file, form
submission, etc.) into a consistent internal `patient_profile` shape that
every downstream agent can rely on, and validates it well enough that a
malformed record produces a clear error instead of a confusing crash three
agents later.
"""

from src.state import PipelineState

REQUIRED_FIELDS = ["patient_id", "age", "sex"]
VALID_SEX_VALUES = {"M", "F"}


def _validate(raw: dict) -> list:
    """Returns a list of human-readable validation problems (empty if clean)."""
    problems = []

    for field in REQUIRED_FIELDS:
        if raw.get(field) in (None, ""):
            problems.append(f"missing required field '{field}'")

    age = raw.get("age")
    if age is not None and not isinstance(age, (int, float)):
        problems.append(f"'age' should be numeric, got {type(age).__name__}")
    elif isinstance(age, (int, float)) and not (0 <= age <= 130):
        problems.append(f"'age' value {age} is outside a plausible human range (0-130)")

    sex = raw.get("sex")
    if sex is not None and str(sex).upper() not in VALID_SEX_VALUES:
        problems.append(f"'sex' should be one of {sorted(VALID_SEX_VALUES)}, got '{sex}'")

    vitals = raw.get("vitals", {})
    if vitals and not isinstance(vitals, dict):
        problems.append("'vitals' should be an object")

    meds = raw.get("current_medications", [])
    if meds and not isinstance(meds, list):
        problems.append("'current_medications' should be a list")

    return problems


def run(state: PipelineState) -> dict:
    raw = state.get("patient_raw", {})
    # New entries only -- trace/errors use an operator.add reducer, so
    # LangGraph concatenates this delta onto the accumulated state itself.
    trace = []
    errors = []

    problems = _validate(raw)
    if problems:
        patient_id = raw.get("patient_id", "UNKNOWN")
        for p in problems:
            errors.append(f"[Intake Agent] Patient '{patient_id}': {p}")
        trace.append(
            f"[Intake Agent] Validation found {len(problems)} issue(s) for "
            f"'{patient_id}' -- proceeding with best-effort defaults, see errors."
        )

    vitals = raw.get("vitals", {}) if isinstance(raw.get("vitals", {}), dict) else {}
    workup = raw.get("cardiac_workup", {}) if isinstance(raw.get("cardiac_workup", {}), dict) else {}
    comorbidities = raw.get("comorbidities", []) if isinstance(raw.get("comorbidities", []), list) else []
    medications = raw.get("current_medications", []) if isinstance(raw.get("current_medications", []), list) else []

    profile = {
        "patient_id": raw.get("patient_id", "UNKNOWN"),
        "name": raw.get("name", "Unknown Patient"),
        "age": raw.get("age"),
        "sex": raw.get("sex"),
        "vitals": {
            "systolic_bp": vitals.get("systolic_bp"),
            "diastolic_bp": vitals.get("diastolic_bp"),
            "heart_rate": vitals.get("heart_rate"),
            "bmi": vitals.get("bmi"),
            "cholesterol_total": vitals.get("cholesterol_total"),
        },
        "cardiac_workup": {
            "chest_pain_type": workup.get("chest_pain_type"),
            "fasting_blood_sugar_high": workup.get("fasting_blood_sugar_high"),
            "resting_ecg": workup.get("resting_ecg"),
            "max_heart_rate": workup.get("max_heart_rate"),
            "exercise_induced_angina": workup.get("exercise_induced_angina"),
            "st_depression": workup.get("st_depression"),
            "st_slope": workup.get("st_slope"),
            "num_major_vessels": workup.get("num_major_vessels"),
            "thalassemia": workup.get("thalassemia"),
        },
        "comorbidities": comorbidities,
        "n_comorbidities": len(comorbidities),
        "smoker": bool(raw.get("smoker", False)),
        "current_medications": [m.lower() for m in medications],
        "intake_note": raw.get("intake_note", ""),
    }

    trace.append(f"[Intake Agent] Normalized profile for patient {profile['patient_id']} ({profile['name']}).")

    return {"patient_profile": profile, "trace": trace, "errors": errors}
