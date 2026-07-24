"""
Unit tests for the Patient Intake Agent's normalization and validation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents import intake_agent


def test_valid_patient_produces_no_errors():
    raw = {
        "patient_id": "T001", "name": "Test Patient", "age": 40, "sex": "F",
        "vitals": {"systolic_bp": 120, "diastolic_bp": 80},
        "comorbidities": [], "current_medications": [],
    }
    result = intake_agent.run({"patient_raw": raw, "trace": [], "errors": []})
    assert result["errors"] == []
    assert result["patient_profile"]["patient_id"] == "T001"


def test_missing_required_field_is_flagged_not_silently_dropped():
    raw = {"name": "No ID Patient", "age": 40, "sex": "F"}  # missing patient_id
    result = intake_agent.run({"patient_raw": raw, "trace": [], "errors": []})
    assert any("patient_id" in e for e in result["errors"])
    # Pipeline should still produce a usable profile rather than crashing
    assert result["patient_profile"]["patient_id"] == "UNKNOWN"


def test_implausible_age_is_flagged():
    raw = {"patient_id": "T002", "age": 250, "sex": "M"}
    result = intake_agent.run({"patient_raw": raw, "trace": [], "errors": []})
    assert any("plausible" in e for e in result["errors"])


def test_invalid_sex_value_is_flagged():
    raw = {"patient_id": "T003", "age": 30, "sex": "unknown"}
    result = intake_agent.run({"patient_raw": raw, "trace": [], "errors": []})
    assert any("'sex'" in e for e in result["errors"])


def test_malformed_medications_field_does_not_crash():
    raw = {"patient_id": "T004", "age": 30, "sex": "F", "current_medications": "aspirin"}
    result = intake_agent.run({"patient_raw": raw, "trace": [], "errors": []})
    assert any("current_medications" in e for e in result["errors"])
    assert result["patient_profile"]["current_medications"] == []
