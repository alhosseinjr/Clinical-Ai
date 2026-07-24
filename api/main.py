"""
FastAPI wrapper around the clinical multi-agent pipeline.

Endpoints:
    GET  /api/health              -> liveness check
    GET  /api/patients             -> summary list of sample patients
    POST /api/run                  -> run the full pipeline, return structured result

Also serves the static frontend in web/ at "/".

Run locally:
    uvicorn api.main:app --reload
"""

import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.graph import build_graph
from src.state import merge_dicts

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATIENTS_FILE = os.path.join(BASE_DIR, "data", "sample_patients.json")
WEB_DIR = os.path.join(BASE_DIR, "web")

app = FastAPI(title="Clinical AI Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def load_sample_patients():
    with open(PATIENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


class RunRequest(BaseModel):
    patient_id: Optional[str] = None
    patient: Optional[Dict[str, Any]] = None
    mock: bool = False


def _local_model_ready() -> bool:
    merged_path = os.environ.get("MERGED_MODEL_PATH", "models/clinical-llm-merged")
    adapter_path = os.environ.get("LORA_ADAPTER_PATH", "models/clinical-lora-adapter")
    return os.path.isdir(merged_path) or os.path.isdir(adapter_path)


def _resolve_patient_raw(req: "RunRequest") -> Dict[str, Any]:
    """Shared by /api/run and /api/run/stream so patient lookup only lives
    in one place."""
    if req.patient:
        return req.patient
    if req.patient_id:
        patients = load_sample_patients()
        match = next((p for p in patients if p["patient_id"] == req.patient_id), None)
        if not match:
            raise HTTPException(status_code=404, detail=f"Unknown patient_id '{req.patient_id}'")
        return match
    raise HTTPException(status_code=400, detail="Provide either 'patient_id' or a full 'patient' object.")


def _shape_result(final_state: Dict[str, Any], mock: bool) -> Dict[str, Any]:
    """Shared response shape for /api/run and the final SSE event of
    /api/run/stream."""
    return {
        "mock_mode": mock,
        "patient_profile": final_state.get("patient_profile"),
        "extracted_entities": final_state.get("extracted_entities"),
        "risk_result": final_state.get("risk_result"),
        "retrieved_evidence": final_state.get("retrieved_evidence"),
        "drug_safety_result": final_state.get("drug_safety_result"),
        "guideline_verification": final_state.get("guideline_verification"),
        "clinical_reasoning": final_state.get("clinical_reasoning"),
        "final_report": final_state.get("final_report"),
        "trace": final_state.get("trace", []),
        "agent_timings": final_state.get("agent_timings", {}),
    }


# Friendly per-node labels for the streaming endpoint's progress events.
_NODE_LABELS = {
    "intake": "Intake complete",
    "nlp": "NLP extraction complete",
    "risk": "Risk prediction complete",
    "retrieval": "Evidence retrieval complete",
    "drug_safety": "Drug safety check complete",
    "guideline_verification": "Guideline verification complete",
    "clinical_reasoning": "Clinical reasoning complete",
    "report": "Report generation complete",
}


@app.get("/api/health")
def health():
    return {"status": "ok", "local_model_ready": _local_model_ready()}


@app.get("/api/patients")
def list_patients():
    patients = load_sample_patients()
    return [
        {
            "patient_id": p["patient_id"],
            "name": p["name"],
            "age": p["age"],
            "sex": p["sex"],
            "comorbidities": p.get("comorbidities", []),
        }
        for p in patients
    ]


@app.post("/api/run")
def run_pipeline(req: RunRequest):
    patient_raw = _resolve_patient_raw(req)
    mock = req.mock or not _local_model_ready()

    graph = get_graph()
    final_state = graph.invoke({
        "patient_raw": patient_raw,
        "mock_mode": mock,
        "trace": [],
        "errors": [],
    })

    return _shape_result(final_state, mock)


@app.post("/api/run/stream")
def run_pipeline_stream(req: RunRequest):
    """
    Same inputs/output shape as POST /api/run, but streams a Server-Sent
    Event as each agent finishes instead of blocking until the whole
    pipeline completes -- lets the frontend show "checkmarks" progress
    (e.g. "Intake complete", "Risk prediction complete", ...) instead of
    a single spinner for the whole run.

    Purely additive: /api/run is unchanged, so existing clients (including
    the current web/ frontend) keep working without modification.

    Event stream:
      data: {"event": "agent_complete", "agent": "risk", "label": "...", "elapsed": 0.01}
      ... one per completed agent, in whatever order they actually finish
          (parallel branches may interleave) ...
      data: {"event": "done", "result": {...same shape as POST /api/run...}}
    """
    patient_raw = _resolve_patient_raw(req)
    mock = req.mock or not _local_model_ready()
    graph = get_graph()

    def event_stream():
        final_state: Dict[str, Any] = {}
        initial_state = {
            "patient_raw": patient_raw,
            "mock_mode": mock,
            "trace": [],
            "errors": [],
        }
        # stream_mode="updates" yields, per superstep, a dict of
        # {node_name: partial_update} for every node that finished in that
        # step -- parallel branches (e.g. nlp+risk) can share a chunk.
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            for node_name, update in chunk.items():
                # trace/errors/agent_timings use reducers in state.py
                # (concatenate / merge) rather than overwrite -- mirror
                # that here instead of a naive dict.update(), or we'd lose
                # every trace entry except the last node's.
                for key, value in update.items():
                    if key in ("trace", "errors"):
                        final_state[key] = final_state.get(key, []) + value
                    elif key == "agent_timings":
                        final_state[key] = merge_dicts(final_state.get(key, {}), value)
                    else:
                        final_state[key] = value

                label = _NODE_LABELS.get(node_name, node_name)
                elapsed = (update.get("agent_timings") or {}).get(node_name)
                payload = {
                    "event": "agent_complete",
                    "agent": node_name,
                    "label": f"\u2713 {label}",
                    "elapsed": elapsed,
                }
                yield f"data: {json.dumps(payload)}\n\n"

        yield f"data: {json.dumps({'event': 'done', 'result': _shape_result(final_state, mock)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Serve the frontend last so it doesn't shadow the /api routes above.
if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
