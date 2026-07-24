"""
Unit tests for the guideline RAG retriever (src/utils/vector_store.py).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.vector_store import GuidelineRetriever

GUIDELINES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "guidelines")

EXPECTED_CONDITIONS = {
    "type_2_diabetes", "hypertension", "asthma", "atrial_fibrillation",
    "hyperlipidemia", "copd", "chronic_kidney_disease", "obesity",
    "depression", "gerd",
}


def test_all_expected_guideline_files_present():
    files = {f[:-4] for f in os.listdir(GUIDELINES_DIR) if f.endswith(".txt")}
    assert EXPECTED_CONDITIONS.issubset(files)


def test_retriever_finds_relevant_guideline():
    retriever = GuidelineRetriever(GUIDELINES_DIR)
    results = retriever.retrieve("hypertension high blood pressure smoker", top_k=2)
    assert len(results) > 0
    assert any("hypertension" in r["source"] for r in results)


def test_retriever_distinguishes_unrelated_conditions():
    retriever = GuidelineRetriever(GUIDELINES_DIR)
    results = retriever.retrieve("gerd heartburn reflux omeprazole", top_k=1)
    assert len(results) > 0
    assert "gerd" in results[0]["source"]


def test_retriever_handles_empty_query_gracefully():
    retriever = GuidelineRetriever(GUIDELINES_DIR)
    results = retriever.retrieve("", top_k=3)
    assert isinstance(results, list)
