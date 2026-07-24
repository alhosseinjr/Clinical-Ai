"""
5. Drug Safety Agent

Cross-checks the patient's current medications (plus any newly mentioned
medications from the NLP agent) against a local drug interaction dataset
and flags any matches.
"""

import json
import os
from src.state import PipelineState

DRUG_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "drug_interactions.json")

_interactions_cache = None


def _load_interactions():
    """Loads and parses data/drug_interactions.json once, then reuses the
    parsed list for every subsequent call/patient in this process -- the
    file never changes at runtime, so re-reading + re-parsing it on every
    single pipeline invocation was pure waste."""
    global _interactions_cache
    if _interactions_cache is None:
        with open(DRUG_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _interactions_cache = data.get("interactions", [])
    return _interactions_cache


def run(state: PipelineState) -> dict:
    profile = state.get("patient_profile", {})
    entities = state.get("extracted_entities", {})

    meds = set(profile.get("current_medications", []))
    meds |= {m.lower() for m in entities.get("mentioned_medications", [])}

    interactions_db = _load_interactions()
    flagged = []
    for entry in interactions_db:
        a, b = entry["drug_a"].lower(), entry["drug_b"].lower()
        if a in meds and b in meds:
            flagged.append(entry)

    result = {
        "medications_considered": sorted(meds),
        "interactions": flagged,
        "flag_count": len(flagged),
        "disclaimer": "Sample dataset for demo purposes only -- not a real drug safety database.",
    }

    trace = [f"[Drug Safety Agent] Checked {len(meds)} medication(s), found {len(flagged)} interaction flag(s)."]

    return {"drug_safety_result": result, "trace": trace}
