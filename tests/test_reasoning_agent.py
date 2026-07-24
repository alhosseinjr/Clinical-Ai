"""
Regression tests for the Clinical Reasoning Agent's output-cleanup helpers.

Locks in the fix for the garbled assessment text seen in production on
patient P006: the fine-tuned model (trained on MTSamples' '.,' pseudo-
newline convention) generated comma-joined numbered findings with a
leaked trailing 'PLAN:' fragment, and near-duplicate recommendations.
"""

from src.agents.reasoning_agent import _clean_assessment, _dedupe_recommendations


def test_clean_assessment_fixes_comma_joined_numbered_findings():
    raw = (
        "1. Stable ejection fraction.,2. Ejection fraction within normal "
        "range.,3. No evidence of acute ischemic change.,PLAN: ,Continue "
        "current management and reassess at next visit."
    )
    cleaned = _clean_assessment(raw)

    assert ",PLAN:" not in cleaned
    assert ".," not in cleaned
    assert not cleaned.startswith("1.")
    assert cleaned == (
        "Stable ejection fraction. Ejection fraction within normal range. "
        "No evidence of acute ischemic change."
    )


def test_clean_assessment_passthrough_for_clean_text():
    clean = "Patient presents with mild hypertension, well controlled on current regimen."
    assert _clean_assessment(clean) == clean


def test_clean_assessment_handles_missing_or_empty_input():
    assert _clean_assessment("") == ""
    assert _clean_assessment(None) is None


def test_dedupe_recommendations_merges_paraphrased_duplicate():
    recs = [
        "Follow up results of the EKG/ECG",
        "We will check the EKG/ECG",
        "If there is any significant change, we will reassess",
    ]
    deduped = _dedupe_recommendations(recs)

    assert len(deduped) == 2
    assert "If there is any significant change, we will reassess" in deduped
    # exactly one EKG-related item should remain
    assert sum("EKG" in r for r in deduped) == 1


def test_dedupe_recommendations_keeps_genuinely_distinct_items():
    recs = [
        "Increase lisinopril dose",
        "Monitor renal function after lisinopril increase",
    ]
    assert _dedupe_recommendations(recs) == recs


def test_dedupe_recommendations_handles_empty_list():
    assert _dedupe_recommendations([]) == []
