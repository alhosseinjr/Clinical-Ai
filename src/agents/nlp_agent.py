"""
2. Medical NLP Agent

Reads the free-text intake note and extracts structured entities
(symptoms, mentioned conditions, mentioned medications) using Claude.
Falls back to a deterministic mock response when running offline.
"""

import json
import re
from src.state import PipelineState
from src.utils.llm import call_llm
from src.orchestrator import should_call_nlp_llm
from src.utils.clinical_ner import extract_clinical_terms


def _parse_json_response(raw_output: str) -> dict:
    """
    Robust parser for small LLM outputs.
    """

    raw_output = raw_output.strip()

    # Normalize smart quotes
    raw_output = (
        raw_output
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
    )

    # Remove markdown fences
    raw_output = raw_output.replace("```json", "")
    raw_output = raw_output.replace("```", "")
    raw_output = raw_output.strip()

    # Extract JSON
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)

    if not match:
        raise json.JSONDecodeError(
            "No JSON object found",
            raw_output,
            0
        )

    data = json.loads(match.group())

    # Normalize keys
    key_mapping = {
        "mentioned Conditions": "mentioned_conditions",
        "mentioned Medications": "mentioned_medications",
        "noteworthy Flags": "notable_flags",
    }

    for old, new in key_mapping.items():
        if old in data:
            data[new] = data.pop(old)

    return data


SYSTEM_PROMPT = """You are a clinical NLP extraction agent.

Your task is to extract ONLY information explicitly mentioned in the patient intake note.

Rules:
- Do not infer or diagnose.
- Do not add information that is not written.
- Normalize medical terms when appropriate.
- Extract all mentioned symptoms, conditions, and medications.
- Include risk-relevant observations in notable_flags.

Return ONLY valid JSON.
Use exactly these keys:

{
  "symptoms": [],
  "mentioned_conditions": [],
  "mentioned_medications": [],
  "notable_flags": []
}

Examples:

Input:
"Patient reports headaches and fatigue. Has diabetes and takes metformin."

Output:
{
  "symptoms": ["headaches", "fatigue"],
  "mentioned_conditions": ["diabetes"],
  "mentioned_medications": ["metformin"],
  "notable_flags": []
}

Input:
"Patient has chest tightness on exertion, hypertension, smokes daily, and father had MI."

Output:
{
  "symptoms": ["chest tightness on exertion"],
  "mentioned_conditions": ["hypertension"],
  "mentioned_medications": [],
  "notable_flags": [
    "active smoker",
    "family history of coronary artery disease"
  ]
}

Important:
- Use normal double quotes.
- Do not translate keys.
- Do not use markdown.
"""


def _mock_response(note: str) -> str:
    return json.dumps({
        "symptoms": ["symptom extraction skipped in mock mode"],
        "mentioned_conditions": [],
        "mentioned_medications": [],
        "notable_flags": ["mock mode -- train/point to a local model (see finetune/) for real extraction"],
    })


def _empty_note_response() -> dict:
    """Deterministic result for a blank/whitespace-only intake note --
    there's nothing to extract, so no need to spend a generation pass."""
    return {
        "symptoms": [],
        "mentioned_conditions": [],
        "mentioned_medications": [],
        "notable_flags": ["intake note was empty -- extraction skipped"],
    }


def run(state: PipelineState) -> dict:
    profile = state.get("patient_profile", {})
    note = profile.get("intake_note", "")

    print("\n=== NLP INPUT ===")
    print(note)
    print("=================\n")

    if not should_call_nlp_llm(note):
        entities = _empty_note_response()
    else:
        # Deterministic clinical extraction
        entities = extract_clinical_terms(note)

    trace = [
    f"[Medical NLP Agent] Extracted "
    f"{len(entities['symptoms'])} symptoms, "
    f"{len(entities['mentioned_conditions'])} conditions, "
    f"{len(entities['mentioned_medications'])} medications."
]

    return {
        "extracted_entities": entities,
        "trace": trace
    }