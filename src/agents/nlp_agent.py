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


SYSTEM_PROMPT = """You are a medical NLP extraction agent. Extract symptoms, conditions, medications, and notable flags from the clinical note.

Respond ONLY with valid JSON in exactly this shape:
{
  "symptoms": ["symptom1", "symptom2"],
  "mentioned_conditions": ["condition1"],
  "mentioned_medications": ["medication1"],
  "notable_flags": ["flag1"],
  "extraction_confidence": "high"
}

Rules for 'extraction_confidence':
- "high": The note is clear, well-structured, and explicitly states the findings.
- "medium": The note is readable but some details are implied or slightly fragmented.
- "low": The note is very brief, fragmented, or ambiguous.

If a category has no items, use an empty list []. Do not include any text outside the JSON."""


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