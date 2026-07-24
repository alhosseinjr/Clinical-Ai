"""
End-to-end and component tests for the clinical pipeline. Run with: pytest -v

These run entirely in mock mode (mock_mode=True in state), so no local
model, GPU, or network access is required.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph import build_graph

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_sample_patients():
    with open(os.path.join(DATA_DIR, "sample_patients.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def run_patient(app, patient, mock=True):
    return app.invoke({
        "patient_raw": patient,
        "mock_mode": mock,
        "trace": [],
        "errors": [],
    })


def test_full_pipeline_runs_in_mock_mode():
    app = build_graph()
    patients = load_sample_patients()
    patient = patients[0]

    final_state = run_patient(app, patient)

    assert "final_report" in final_state
    assert final_state["patient_profile"]["patient_id"] == patient["patient_id"]
    assert "risk_result" in final_state
    assert final_state["risk_result"]["risk_category"] in {"low", "moderate", "high"}
    assert len(final_state["trace"]) == 8  # one entry per agent
    assert final_state["errors"] == []  # clean sample data should validate with no issues


def test_pipeline_runs_for_all_sample_patients():
    """Every sample patient should complete the full graph without error,
    across the full diversity of ages, comorbidities, and medication counts
    now in the dataset (12 patients, 10 distinct comorbidities, 0-5 meds)."""
    app = build_graph()
    patients = load_sample_patients()
    assert len(patients) == 12

    for patient in patients:
        final_state = run_patient(app, patient)
        assert final_state["final_report"].startswith("# Clinical AI Pipeline Report"), (
            f"Report malformed for {patient['patient_id']}"
        )
        assert final_state["risk_result"]["risk_category"] in {"low", "moderate", "high"}


def test_polypharmacy_patient_flags_multiple_interactions():
    """P009 is on 5 meds including warfarin + allopurinol -- should surface
    at least one real interaction flag, not silently pass everyone through."""
    app = build_graph()
    patients = load_sample_patients()
    patient = next(p for p in patients if p["patient_id"] == "P009")

    final_state = run_patient(app, patient)
    assert final_state["drug_safety_result"]["flag_count"] >= 1


def test_healthy_baseline_patient_is_not_flagged_high_risk():
    """P004 has no comorbidities, no meds, normal vitals -- sanity check that
    the risk model doesn't default everyone to 'high'."""
    app = build_graph()
    patients = load_sample_patients()
    patient = next(p for p in patients if p["patient_id"] == "P004")

    final_state = run_patient(app, patient)
    assert final_state["risk_result"]["risk_category"] in {"low", "moderate"}
