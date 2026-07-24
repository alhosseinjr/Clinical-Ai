"""
Orchestrator: the pipeline's control layer.

LangGraph already decides *when* a node is runnable (once its upstream
edges complete, see graph.py's DAG). This module adds the two things a
graph topology can't express on its own:

1. **Timing / failure isolation** -- `instrument()` wraps every node so we
   get a per-agent runtime in `state["agent_timings"]`, and so one agent
   raising doesn't take down the whole graph -- it's recorded in `errors`
   and the pipeline continues with whatever that agent would have
   defaulted to, instead of a hard crash three nodes before the report.

2. **Skip decisions** -- `should_call_nlp_llm()` / `should_call_guideline_llm()`
   are the "does this agent actually need to run its expensive part"
   logic the brief asked for. They're deliberately *not* wired as
   graph-level conditional edges: skipping a whole node would also skip
   its `trace` entry, which changes pipeline-shape assumptions tests and
   the report both rely on (one trace line per agent). Instead, the
   *agent itself* still runs (cheap, deterministic bookkeeping) but asks
   the orchestrator whether the expensive local-LLM call is worth making.
   This is the actual latency win -- LLM generation, not the Python
   function call around it -- with zero change to pipeline shape.
"""

import time
from functools import wraps
from typing import Callable, Dict


def instrument(name: str, fn: Callable) -> Callable:
    """Wrap a LangGraph node with timing + failure isolation.

    Records elapsed seconds into `agent_timings[name]` (merged via the
    `merge_dicts` reducer in state.py, so this is safe under parallel
    branches). If the wrapped agent raises, the exception is caught,
    logged into `errors`, and an empty partial update is returned so the
    graph can keep going instead of crashing the whole run.
    """

    @wraps(fn)
    def wrapped(state):
        start = time.perf_counter()
        try:
            result = fn(state)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any
            # agent failure should degrade gracefully, not kill the run.
            elapsed = time.perf_counter() - start
            return {
                "errors": [f"[Orchestrator] Agent '{name}' failed after {elapsed:.3f}s: {exc}"],
                "agent_timings": {name: round(elapsed, 4)},
            }
        elapsed = time.perf_counter() - start
        result = dict(result)
        result["agent_timings"] = {name: round(elapsed, 4)}
        return result

    return wrapped


def should_call_nlp_llm(intake_note: str) -> bool:
    """No point spending a generation pass extracting entities from an
    empty/whitespace-only intake note -- there's nothing to extract."""
    return bool(intake_note and intake_note.strip())


def should_call_guideline_llm(evidence: list) -> bool:
    """No point asking the model to judge relevance of zero snippets --
    the answer is deterministically "not aligned, nothing retrieved"."""
    return bool(evidence)


def summarize_timings(agent_timings: Dict[str, float]) -> str:
    """Human-readable one-liner for logs/traces, e.g. for CLI output."""
    if not agent_timings:
        return "no timing data"
    total = sum(agent_timings.values())
    parts = ", ".join(f"{k}={v:.3f}s" for k, v in sorted(agent_timings.items(), key=lambda kv: -kv[1]))
    return f"total={total:.3f}s ({parts})"
