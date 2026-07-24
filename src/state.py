"""
Shared state definition for the clinical multi-agent pipeline.

Every agent (LangGraph node) receives the full PipelineState dict and
returns a partial dict with the keys it updates. LangGraph merges these
partial updates into the running state as the graph executes.

`trace` and `errors` are written by every agent, including agents that now
run in parallel branches (see src/graph.py). Without a reducer, two nodes
in the same superstep writing to the same key is an ambiguous update and
LangGraph raises InvalidUpdateError. `Annotated[..., operator.add]` tells
LangGraph how to merge concurrent partial updates: by concatenation, same
as the sequential behavior these lists already had.
"""

import operator
from typing import TypedDict, List, Dict, Any, Optional, Annotated


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer for agent_timings: shallow-merge two partial-update dicts."""
    merged = dict(a or {})
    merged.update(b or {})
    return merged


class PipelineState(TypedDict, total=False):
    # --- Config / control ---
    mock_mode: bool                     # if True, LLM agents return canned output (no API calls)

    # --- 1. Patient Intake Agent ---
    patient_raw: Dict[str, Any]         # raw patient record as loaded from source
    patient_profile: Dict[str, Any]     # normalized profile produced by intake agent

    # --- 2. Medical NLP Agent ---
    extracted_entities: Dict[str, Any]  # symptoms / conditions / meds pulled from free text

    # --- 3. Risk Prediction Agent (ML) ---
    risk_result: Dict[str, Any]         # {"risk_score": float, "risk_category": str, "top_factors": [...]}

    # --- 4. Evidence Retrieval Agent (RAG) ---
    retrieved_evidence: List[Dict[str, Any]]  # list of {"source": str, "snippet": str, "score": float}

    # --- 5. Drug Safety Agent ---
    drug_safety_result: Dict[str, Any]  # {"interactions": [...], "flags": [...]}

    # --- 6. Guideline Verification Agent ---
    guideline_verification: Dict[str, Any]  # {"aligned": bool, "notes": str, "citations": [...]}

    # --- 7. Clinical Reasoning Agent ---
    clinical_reasoning: Dict[str, Any]  # {"assessment": str, "recommendations": [...]}

    # --- 8. Report Generation Agent ---
    final_report: str                   # rendered markdown report

    # --- bookkeeping ---
    # Annotated with a reducer: safe to update from parallel branches (see graph.py).
    trace: Annotated[List[str], operator.add]   # human-readable log of what each agent did
    errors: Annotated[List[str], operator.add]  # any non-fatal errors collected along the way

    # --- orchestration / timing (populated by src/orchestrator.py) ---
    agent_timings: Annotated[Dict[str, float], merge_dicts]  # {agent_name: seconds}
