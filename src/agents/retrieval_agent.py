"""
4. Evidence Retrieval Agent (RAG)

Builds a query from the patient's comorbidities + extracted conditions and
retrieves the most relevant guideline snippets from the local corpus
(data/guidelines/) using semantic embedding similarity (sentence-transformers / all-MiniLM-L6-v2).
"""

import os
from src.state import PipelineState
from src.utils.vector_store import GuidelineRetriever

_retriever = None

GUIDELINES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "guidelines")


def _get_retriever() -> GuidelineRetriever:
    global _retriever
    if _retriever is None:
        _retriever = GuidelineRetriever(GUIDELINES_DIR)
    return _retriever


def run(state: PipelineState) -> dict:
    profile = state.get("patient_profile", {})
    entities = state.get("extracted_entities", {})

    query_terms = list(profile.get("comorbidities", []))
    query_terms += entities.get("mentioned_conditions", [])
    query_terms += entities.get("symptoms", [])
    query = " ".join(query_terms) or profile.get("intake_note", "")

    retriever = _get_retriever()
    results = retriever.retrieve(query, top_k=3)

    trace = [f"[Evidence Retrieval Agent] Retrieved {len(results)} guideline snippet(s) "
             f"for query terms: {query_terms}."]

    return {"retrieved_evidence": results, "trace": trace}
