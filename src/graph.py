"""
Builds the LangGraph StateGraph that wires the 8 agents together.

The pipeline used to be a single linear chain (Intake -> NLP -> Risk ->
Retrieval -> Drug Safety -> Guideline -> Reasoning -> Report), but that
ordering doesn't match what the agents actually depend on -- it was just
the order they happened to be written in. Reading each agent's `run()`,
the real data dependencies are:

  intake -> nlp            (needs patient_profile.intake_note)
  intake -> risk            (needs patient_profile.vitals/cardiac_workup --
                              NOT extracted_entities, so it doesn't need nlp)
  nlp -> retrieval          (needs extracted_entities + comorbidities)
  nlp -> drug_safety        (needs extracted_entities.mentioned_medications)
  risk, retrieval -> guideline_verification (needs risk_result + evidence,
                              NOT drug_safety_result)
  guideline_verification, drug_safety -> clinical_reasoning (needs both,
                              plus everything upstream, already in state)
  clinical_reasoning -> report

So `risk` can run in parallel with `nlp`, and `retrieval`/`drug_safety` can
run in parallel with each other once `nlp` finishes. `drug_safety` has no
downstream dependency until `reasoning`, so it also overlaps with
`guideline_verification`. This turns the critical path from 8 sequential
hops into 5: intake -> nlp -> {retrieval|drug_safety} -> guideline ->
reasoning -> report (risk overlaps intake->nlp's second half, drug_safety
overlaps guideline).

Each node is still a plain function (state: PipelineState) -> dict --
agent code is unchanged, only the wiring changed. LangGraph runs any nodes
whose upstream edges have all completed within the same superstep, and a
node with multiple incoming edges (e.g. guideline_verification) waits for
all of them before running -- no explicit join/barrier code needed.

Nodes are wrapped by src.orchestrator so every run is timed and NLP/
Guideline can skip their LLM call when there's nothing to do.
"""

from langgraph.graph import StateGraph, END

from src.state import PipelineState
from src.agents import (
    intake_agent,
    nlp_agent,
    risk_agent,
    retrieval_agent,
    drug_safety_agent,
    guideline_agent,
    reasoning_agent,
    report_agent,
)
from src.orchestrator import instrument


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("intake", instrument("intake", intake_agent.run))
    graph.add_node("nlp", instrument("nlp", nlp_agent.run))
    graph.add_node("risk", instrument("risk", risk_agent.run))
    graph.add_node("retrieval", instrument("retrieval", retrieval_agent.run))
    graph.add_node("drug_safety", instrument("drug_safety", drug_safety_agent.run))
    graph.add_node("guideline_verification", instrument("guideline_verification", guideline_agent.run))
    graph.add_node("clinical_reasoning", instrument("clinical_reasoning", reasoning_agent.run))
    graph.add_node("report", instrument("report", report_agent.run))

    graph.set_entry_point("intake")

    # Fan-out: intake unblocks both nlp and risk, which run in parallel.
    graph.add_edge("intake", "nlp")
    graph.add_edge("intake", "risk")

    # Fan-out: nlp unblocks both retrieval and drug_safety, which run in parallel.
    graph.add_edge("nlp", "retrieval")
    graph.add_edge("nlp", "drug_safety")

    # Fan-in: guideline_verification waits for BOTH risk and retrieval to
    # complete before running once. Passing a list to add_edge creates a
    # real join/barrier; two separate add_edge(x, target) calls do NOT --
    # LangGraph fires the target once per completed predecessor in that
    # case, which silently re-runs it multiple times with partial state.
    graph.add_edge(["risk", "retrieval"], "guideline_verification")

    # Fan-in: clinical_reasoning waits for BOTH guideline_verification and
    # drug_safety (same join semantics as above).
    graph.add_edge(["guideline_verification", "drug_safety"], "clinical_reasoning")

    graph.add_edge("clinical_reasoning", "report")
    graph.add_edge("report", END)

    return graph.compile()
