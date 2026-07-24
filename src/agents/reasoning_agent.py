"""
7. Clinical Reasoning Agent

Synthesizes every upstream agent's output (profile, extracted entities,
risk prediction, retrieved evidence, drug safety flags, guideline
verification) into a single coherent assessment with next-step
recommendations. This is the "senior reasoning" step before report writing.
"""

import json
import re
from src.state import PipelineState
from src.utils.llm import call_llm

# Shortened prompt as recommended for smaller models
SYSTEM_PROMPT = """
You are the Clinical Reasoning Agent.

Use ONLY the supplied pipeline outputs.

Never invent:
- diagnoses
- medications
- symptoms
- laboratory findings
- imaging findings
- family history
- treatments

If evidence is insufficient, state:
"Insufficient evidence to support additional conclusions."

Recommendations must come ONLY from:
- patient profile
- drug safety
- verified guideline
- retrieved evidence

Return ONLY valid JSON:

{
  "assessment":"2-4 sentence summary",
  "recommendations":["..."],
  "priority":"routine|elevated|urgent"
}
"""


def _mock_response():
    return json.dumps({
        "assessment": "Mock mode -- reasoning synthesis skipped. Train/point to a local model (see finetune/) for real output.",
        "recommendations": ["Train the local LoRA adapter (finetune/scripts/train_lora.py) and re-run to get real recommendations."],
        "priority": "routine",
    })


def _clean_assessment(text: str) -> str:
    """Defense-in-depth cleanup for the model's 'assessment' string.

    The current LoRA adapter was fine-tuned on MTSamples text that used
    '.,' as a pseudo-newline between numbered findings (e.g. '1. Stable
    ejection fraction.,2. Normal EF.,PLAN: ,Continue...'), so it sometimes
    reproduces that broken punctuation -- including a leaked trailing
    'PLAN:' fragment that belongs in recommendations, not the assessment
    sentence. This runs on every generation regardless of which adapter
    is loaded, so it holds even before/without a retrain.
    """
    if not isinstance(text, str) or not text:
        return text
    # Cut off a leaked header and anything after it.
    text = re.split(r"[.,]?\s*(?:PLAN|RECOMMENDATIONS?)\s*:\s*,?", text, maxsplit=1)[0]
    # '.,' -> '. ' (real sentence boundary instead of the training corpus's comma)
    text = re.sub(r"\.,\s*", ". ", text)
    # Strip embedded list numbering ("1. ", "2. ") now that items read as
    # consecutive sentences.
    text = re.sub(r"(?<=[.\s])\d+\.\s+", "", text)
    text = re.sub(r"^\d+\.\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


_STOPWORDS = {
    "the", "a", "an", "we", "will", "of", "to", "for", "and", "is", "are",
    "if", "there", "any", "at", "on", "with", "results", "significant",
}


def _tokens(item: str) -> set:
    words = re.sub(r"[^a-z0-9 ]", " ", item.lower()).split()
    return {w for w in words if w not in _STOPWORDS}


def _dedupe_recommendations(items, overlap_threshold: float = 0.4) -> list:
    """Drops near-duplicate recommendations (e.g. 'Follow up results of the
    EKG/ECG' and 'We will check the EKG/ECG') that the small fine-tuned
    model sometimes restates rather than merges. Uses token-overlap
    (Jaccard over content words, stopwords removed) rather than exact-match
    so paraphrased restatements are caught too. Threshold (0.4) was tuned
    against one real example (validated it merges "Follow up results of the
    EKG/ECG" + "We will check the EKG/ECG" [overlap 0.4] but does NOT merge
    two genuinely distinct lisinopril-related recommendations [overlap
    0.33]) -- treat as a reasonable starting point, not a proven cutoff.
    Keeps the longer/more specific phrasing of each duplicate pair.
    """
    if not isinstance(items, list):
        return items
    kept: list = []
    kept_tokens: list = []
    for item in items:
        if not isinstance(item, str):
            kept.append(item)
            kept_tokens.append(set())
            continue
        tok = _tokens(item)
        is_dup = False
        for i, existing_tok in enumerate(kept_tokens):
            if not tok or not existing_tok:
                continue
            overlap = len(tok & existing_tok) / len(tok | existing_tok)
            if overlap >= overlap_threshold:
                is_dup = True
                if len(item) > len(kept[i]):  # prefer the more specific phrasing
                    kept[i] = item
                    kept_tokens[i] = tok
                break
        if not is_dup:
            kept.append(item)
            kept_tokens.append(tok)
    return kept


def _validate_reasoning(reasoning: dict, payload: dict) -> dict:
    """
    Simple hallucination filter.
    Removes recommendations mentioning unsupported medications or diagnoses.
    """
    allowed_text = json.dumps(payload).lower()

    blocked_terms = [
        "aspirin",
        "beta blocker",
        "metoprolol",
        "clopidogrel",
        "angioplasty",
        "stent",
        "myocardial infarction",
        "stable angina",
        "unstable angina",
        "heart failure",
        "ecg",
        "ekg",
        "ejection fraction",
        "family history",
    ]

    cleaned = []

    for rec in reasoning.get("recommendations", []):
        if not isinstance(rec, str):
            continue

        text = rec.lower()

        unsupported = any(
            term in text and term not in allowed_text
            for term in blocked_terms
        )

        if not unsupported:
            cleaned.append(rec)

    reasoning["recommendations"] = cleaned
    return reasoning


def _summarize_evidence(evidence):
    """Summarizes retrieved evidence, including a text snippet for context."""
    summary = []
    for item in evidence:
        summary.append({
            "source": item.get("source"),
            "topic": item.get("query"),
            "score": round(item.get("score", 0), 3),
            "snippet": (item.get("text") or "")[:300],
        })
    return summary


def run(state: PipelineState) -> dict:
    mock = state.get("mock_mode", False)

    # Extract profile and build a compact version
    profile = state.get("patient_profile", {})

    compact_profile = {
        "age": profile.get("age"),
        "sex": profile.get("sex"),
        "smoker": profile.get("smoker"),
        "comorbidities": profile.get("comorbidities"),
        "current_medications": profile.get("current_medications", []),
    }

    # Build the final payload for the LLM
    payload = {
        "patient_profile": compact_profile,
        "extracted_entities": state.get("extracted_entities", {}),
        "risk_result": state.get("risk_result", {}),
        "retrieved_evidence": _summarize_evidence(
            state.get("retrieved_evidence", [])
        ),
        "drug_safety_result": state.get("drug_safety_result", {}),
        "guideline_verification": state.get("guideline_verification", {}),
    }

    raw_output = call_llm(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=json.dumps(payload, separators=(",", ":")),
        mock=mock,
        mock_response=_mock_response(),
        max_tokens=320,
    )

    try:
        reasoning = json.loads(raw_output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if match:
            try:
                reasoning = json.loads(match.group())
            except Exception:
                reasoning = {
                    "assessment": "parse_error: could not parse model output as JSON",
                    "recommendations": [],
                    "priority": "routine",
                    "raw_output": raw_output,
                }
        else:
            reasoning = {
                "assessment": "parse_error: could not parse model output as JSON",
                "recommendations": [],
                "priority": "routine",
                "raw_output": raw_output,
            }

    if "assessment" in reasoning:
        reasoning["assessment"] = _clean_assessment(reasoning["assessment"])

    if "recommendations" in reasoning:
        reasoning["recommendations"] = _dedupe_recommendations(
            reasoning["recommendations"]
        )
        reasoning = _validate_reasoning(reasoning, payload)

    # Validate priority to prevent invalid outputs like "High" or "Moderate"
    if reasoning.get("priority") not in {"routine", "elevated", "urgent"}:
        reasoning["priority"] = "routine"

    trace = [
        (
            f"[Clinical Reasoning Agent] "
            f"priority={reasoning.get('priority')}, "
            f"recommendations={len(reasoning.get('recommendations', []))}"
        )
    ]

    return {
        "clinical_reasoning": reasoning,
        "trace": trace,
    }