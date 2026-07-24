"""
6. Guideline Verification Agent

Uses Claude to check whether the retrieved evidence snippets are actually
relevant/consistent with the patient's profile and risk result, and
produces a short verification note with citations back to source files.
"""

import json
from src.state import PipelineState
from src.utils.llm import call_llm
from src.utils.llm_parser import parse_llm_json
from src.orchestrator import should_call_guideline_llm

SYSTEM_PROMPT = """You are a guideline verification agent in a clinical
decision-support pipeline. You receive a patient summary and a list of
retrieved guideline snippets (each with a source filename). Your job is to
judge whether the snippets are actually relevant to this patient, and note
any gaps. Respond ONLY with valid JSON in exactly this shape:

{
  "aligned": true | false,
  "notes": "1-3 sentence summary of how well the evidence fits the patient",
  "citations": ["source_filename.txt", "..."]
}

Only include a filename in "citations" if its snippet is genuinely relevant."""


def _mock_response(evidence):
    sources = [e["source"] for e in evidence[:5]]
    return json.dumps({
        "aligned": bool(sources),
        "notes": "Mock mode -- verification skipped. Train/point to a local model (see finetune/) for real analysis.",
        "citations": sources,
    })


def _no_evidence_response() -> dict:
    """Deterministic result when retrieval returned nothing -- there's no
    relevance judgment to make, so skip the generation pass."""
    return {
        "aligned": False,
        "notes": "No evidence was retrieved -- verification skipped.",
        "citations": [],
    }


def run(state: PipelineState) -> dict:
    profile = state.get("patient_profile", {})
    risk = state.get("risk_result", {})
    evidence = state.get("retrieved_evidence", [])
    mock = state.get("mock_mode", False)

    if not should_call_guideline_llm(evidence):
        verification = _no_evidence_response()
    else:
        top_evidence = sorted(
            evidence,
            key=lambda x: x.get("score", 0),
            reverse=True
        )[:5]

        evidence_block = "\n\n".join(
            f"Source: {e['source']}\n"
            f"Score: {e.get('score', 0):.3f}\n"
            f"Snippet: {(e.get('snippet') or e.get('text') or '')[:300]}"
            for e in top_evidence
        )

        user_prompt = f"""Patient summary:
- Age: {profile.get('age')}, Sex: {profile.get('sex')}
- Comorbidities: {profile.get('comorbidities')}
- Smoker: {profile.get('smoker')}
- Predicted risk category: {risk.get('risk_category')} (score {risk.get('risk_score')})

Retrieved evidence:
{evidence_block}"""

        raw_output = call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            mock=mock,
            mock_response=_mock_response(evidence),
            max_tokens=180,
        )

        # Single call replaces the entire try/regex/fallback block
        verification = parse_llm_json(
            raw_output,
            default={
                "aligned": False,
                "notes": "Unable to parse model output.",
                "citations": [],
            },
        )

        # Build allowed sources set
        allowed_sources = {
            e.get("source")
            for e in evidence
            if isinstance(e.get("source"), str)
        }

        # Filter and deduplicate citations
        citations = sorted({
            c
            for c in verification.get("citations", [])
            if isinstance(c, str) and c in allowed_sources
        })

        # Validate notes
        notes = verification.get("notes")
        if not isinstance(notes, str) or not notes.strip():
            notes = "Insufficient evidence."

        # Derive aligned from citations
        aligned = bool(citations)

        # Build sanitized verification object
        sanitized = {
            "aligned": aligned,
            "notes": notes,
            "citations": citations,
        }

        if "raw_output" in verification:
            sanitized["raw_output"] = verification["raw_output"]

        verification = sanitized

    trace = [
        f"[Guideline Verification] "
        f"evidence={len(evidence)} "
        f"citations={len(verification['citations'])} "
        f"aligned={verification['aligned']}"
    ]

    if len(evidence) > 5:
        trace.append(
            f"[Guideline Verification Agent] Using top 5 of {len(evidence)} retrieved documents."
        )

    return {"guideline_verification": verification, "trace": trace}