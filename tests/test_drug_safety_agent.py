"""
Unit tests for the Drug Safety Agent and the expanded interaction dataset.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents import drug_safety_agent

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def test_interaction_dataset_has_no_duplicate_pairs():
    with open(os.path.join(DATA_DIR, "drug_interactions.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    seen = set()
    for entry in data["interactions"]:
        pair = frozenset([entry["drug_a"].lower(), entry["drug_b"].lower()])
        assert pair not in seen, f"Duplicate interaction pair: {pair}"
        seen.add(pair)


def test_known_high_severity_interaction_is_flagged():
    state = {
        "patient_profile": {"current_medications": ["warfarin", "ibuprofen"]},
        "extracted_entities": {},
        "trace": [],
    }
    result = drug_safety_agent.run(state)
    flagged_pairs = [
        frozenset([i["drug_a"].lower(), i["drug_b"].lower()])
        for i in result["drug_safety_result"]["interactions"]
    ]
    assert frozenset(["warfarin", "ibuprofen"]) in flagged_pairs


def test_no_medications_produces_no_flags():
    state = {
        "patient_profile": {"current_medications": []},
        "extracted_entities": {},
        "trace": [],
    }
    result = drug_safety_agent.run(state)
    assert result["drug_safety_result"]["flag_count"] == 0


def test_unrelated_medications_produce_no_false_positive():
    state = {
        "patient_profile": {"current_medications": ["albuterol", "omeprazole"]},
        "extracted_entities": {},
        "trace": [],
    }
    result = drug_safety_agent.run(state)
    assert result["drug_safety_result"]["flag_count"] == 0
