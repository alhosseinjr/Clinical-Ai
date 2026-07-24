"""
2. Medical NLP Agent

Reads the free-text intake note and extracts structured entities
(symptoms, mentioned conditions, mentioned medications) using Claude.
Falls back to a deterministic mock response when running offline.
"""

import json
from src.state import PipelineState
from src.utils.llm import call_llm
from src.utils.llm_parser import parse_llm_json
from src.orchestrator import should_call_nlp_llm
from src.utils.clinical_ner import extract_clinical_terms

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
    """Deterministic result for a blank/whitespace-only intake note."""
    return {
        "symptoms": [],
        "mentioned_conditions": [],
        "mentioned_medications": [],
        "notable_flags": ["intake note was empty -- extraction skipped"],
    }


def run(state: PipelineState) -> dict:
    profile = state.get("patient_profile", {})
    note = profile.get("intake_note", "")
    mock = state.get("mock_mode", False)
    
    # Deterministic medical term extraction
    detected_entities = extract_clinical_terms(note)

    if not should_call_nlp_llm(note):
        entities = _empty_note_response()
    else:
        raw_output = call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=f"Intake note:\n\n{note}",
            mock=mock,
            mock_response=_mock_response(note),
            max_tokens=220,
        )

        # Use the shared, robust parser
        entities = parse_llm_json(
            raw_output,
            default={
                "symptoms": [],
                "mentioned_conditions": [],
                "mentioned_medications": [],
                "notable_flags": ["parse_error: could not parse model output as JSON"],
                "raw_output": raw_output,
            }
        )

        # Normalize keys (in case the LLM hallucinates slight variations)
        key_mapping = {
            "mentioned Conditions": "mentioned_conditions",
            "mentioned Medications": "mentioned_medications",
            "noteworthy Flags": "notable_flags",
        }
        for old, new in key_mapping.items():
            if old in entities:
                entities[new] = entities.pop(old)

        # Ensure all expected keys exist and are lists
        for key in ["symptoms", "mentioned_conditions", "mentioned_medications", "notable_flags"]:
            if key not in entities:
                entities[key] = []
            elif isinstance(entities[key], str):
                entities[key] = [entities[key]]
            
    # Merge rule-based extraction with LLM extraction
    for key in detected_entities:
        entities[key] = list(
            set(
                entities.get(key, [])
                + detected_entities[key]
            )
        )
        
    trace = [
        f"[Medical NLP Agent] Extracted {len(entities.get('symptoms', []))} symptom(s) "
        f"and {len(entities.get('notable_flags', []))} flag(s) from intake note."
    ]

    return {"extracted_entities": entities, "trace": trace}
