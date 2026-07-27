# 🏥 Clinical AI Multi-Agent Pipeline

**A locally-run clinical decision support system, engineered like production software — parallel agent orchestration, semantic retrieval, hybrid LLM verification, and a deliberately narrow use of AI where it actually earns its keep.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.0-green.svg)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-lightgrey.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> ⚠️ **Disclaimer**: This is a **software engineering portfolio project**, not a medical device. All clinical guidelines, drug interactions, and patient records are synthetic, simplified, or drawn from public research datasets for educational purposes. **Never use this system for real clinical decision-making.** The engineering practices below — orchestration, testing, failure isolation, honest evaluation — are real; the clinical judgment is not.

[Report Bug](https://github.com/alhosseinjr/Clinical-Ai-Multi-Agent/issues) · [Request Feature](https://github.com/alhosseinjr/Clinical-Ai-Multi-Agent/issues)

---

## Overview

Clinical decision support tools usually fall into one of two camps: a thin prompt wrapper around a hosted LLM for everything, or a rigid rules engine with no reasoning layer at all. This project deliberately avoids both — it's an 8-agent pipeline where each step uses the *simplest tool that's actually reliable for that job*: deterministic logic where the task is well-bounded, a supervised ML model where the task is prediction, semantic search where the task is retrieval, and a fine-tuned local LLM reserved specifically for the two steps that genuinely require judgment — verifying evidence relevance and synthesizing a final clinical assessment.

Given a patient record, the pipeline extracts clinical entities from free-text notes, predicts cardiovascular risk from structured features, retrieves relevant clinical guidelines by *meaning* rather than keyword, checks for dangerous drug interactions, verifies that retrieved evidence actually supports the case at hand (with a programmatic safety net backing up the LLM's judgment), and synthesizes all of it into a structured, cited clinical report — end to end, in seconds, with zero API calls and zero data leaving the machine.

The interesting part isn't any single agent. It's the architecture: agents that can run concurrently do, agents that depend on each other wait correctly, the LLM is used only where it adds value instead of everywhere by default, and if one agent fails, the pipeline degrades gracefully instead of crashing three steps before the report.

---

## ✨ Key Features

- 🧩 **Deliberately narrow LLM use** — a fine-tuned `Qwen2.5-3B-Instruct` + LoRA handles only Guideline Verification and Clinical Reasoning; simpler, more reliable methods handle everything else
- 🔍 **Deterministic Clinical NER** — symptom/condition/medication/risk-factor extraction via regex + lookup tables, not a model call: instant, reproducible, zero hallucination risk for a well-bounded extraction task
- 🛡️ **Hybrid guideline verification with a programmatic safety net** — if the LLM rejects evidence that the embedding model scored above 0.35 semantic similarity, the pipeline overrides the LLM and trusts the semantic score instead
- 🤖 **8-agent LangGraph pipeline** with real parallel execution — not a linear chain
- 🔎 **Semantic RAG** — `sentence-transformers` (`all-MiniLM-L6-v2`) embeddings + cosine similarity, so "heart attack" retrieves guidelines about "myocardial infarction"
- 📊 **Explainable risk scoring** — continuous probability (not just a bucket), plus the top contributing features per prediction, read directly from the trained model's own feature importances
- 🌲 **Random Forest risk model** — trained on the real UCI Heart Disease (Cleveland) dataset, with results reported honestly rather than cherry-picked
- 💊 **Drug interaction engine** — 28 rule-based interaction pairs with severity levels
- ⚡ **Genuine concurrency** — `intake→{nlp, risk}` and `nlp→{retrieval, drug_safety}` run as parallel branches with correct join semantics
- 🩺 **Failure-isolated execution** — any single agent can fail without taking down the run; the orchestrator times every agent and logs failures instead of crashing
- 🖥️ **Full-stack UI** — FastAPI backend + vanilla JS console with real-time agent tracing
- 🔒 **Privacy-first** — 100% offline, no API keys, no data leaving your machine
- 🧪 **~64 tests across 7 files**, all runnable in mock mode (no GPU, no model download, no network access required)

---

## 💼 Business & Clinical Value

This is a portfolio demo, not a validated product — no clinical trials, no regulatory clearance, no measured deployment outcomes. What it does demonstrate is the *architecture* behind the value drivers a real health-tech buyer or clinical team actually cares about:

- **Cognitive load reduction**: instead of a clinician manually cross-referencing a patient's notes, risk factors, relevant guidelines, and medication list, the pipeline pulls all four into one structured, cited report in seconds. That's the actual workflow bottleneck clinical decision support tools are meant to address.
- **A specific, named answer to the AI liability question**: healthcare buyers' biggest objection to LLM-based tools is hallucination risk. The hybrid guideline-verification safety net — overriding a falsely-strict LLM rejection with a logged semantic-similarity score — is a concrete, inspectable answer to "how do you keep the model from silently dropping or fabricating evidence," not just a claim that the system is "hallucination-resistant."
- **Auditability**: every citation traces back to either an LLM judgment or an explicit programmatic override, and every recommendation is filtered against upstream state before reaching the report. That trace is the kind of artifact a clinical safety or compliance review actually asks for.
- **Data locality by default**: fully local inference means patient data never leaves the machine and there's no per-query API cost — relevant both for HIPAA-style data-residency requirements and for deployment in low-connectivity or resource-constrained settings (rural clinics, field medicine) where "call an external API" isn't an option at all, not just a preference.
- **Low marginal cost to extend scope**: adding a clinical specialty means dropping a `.txt` guideline file or a JSON interaction pair — not retraining the fine-tuned model or restructuring the pipeline. That maps directly to the "how expensive is it to expand coverage" question a product roadmap discussion would raise.
- **Honest evaluation as a trust signal, not a weakness**: reporting a 0.714 recall plainly, alongside the dataset's real limitations, is what a clinical or compliance stakeholder needs to make an informed risk decision — a demo that oversells its numbers is a liability the moment someone checks; one that discloses them up front is the beginning of an actual trust relationship.

---

## 🏗️ Architecture

The pipeline used to be a strict linear chain (Intake → NLP → Risk → Retrieval → Drug Safety → Guideline → Reasoning → Report). It's now wired to match actual data dependencies, which turns an 8-hop critical path into 5:

```
                    ┌─────────────┐
                    │   INTAKE    │
                    └──────┬──────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
           ┌───────────┐       ┌───────────┐
           │    NLP    │       │   RISK    │   ← run in parallel
           │(det. NER) │       │   (ML)    │
           └─────┬─────┘       └─────┬─────┘
                 │                   │
        ┌────────┴────────┐         │
        ▼                 ▼         │
  ┌───────────┐     ┌───────────┐   │
  │ RETRIEVAL │     │   DRUG    │   │  ← retrieval & drug_safety
  │   (RAG)   │     │  SAFETY   │   │    run in parallel
  └─────┬─────┘     └─────┬─────┘   │
        │                 │         │
        └────────┬────────┴─────────┘
                  ▼
         ┌─────────────────┐
         │   GUIDELINE      │  ← waits for BOTH risk + retrieval
         │  VERIFICATION    │    (real join, not two racing edges)
         │    (hybrid)      │
         └────────┬─────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   │
 ┌─────────────┐            │
 │  CLINICAL   │ ◄──────────┘   ← waits for BOTH guideline + drug_safety
 │  REASONING  │       (LLM)
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │   REPORT    │
 └─────────────┘
```

**Why this matters technically:** LangGraph's `add_edge([node_a, node_b], target)` creates a real join/barrier — `target` fires exactly once, after both predecessors complete. Two separate `add_edge` calls to the same target look similar but re-fire it once per predecessor with partial state, which is a subtle, easy-to-ship bug. Making this safe under concurrency also required adding `Annotated[list, operator.add]` reducers to the shared `trace` and `errors` state fields — without them, two parallel branches writing to the same list key in the same step is an ambiguous update LangGraph rejects outright.

### Agent responsibilities

| # | Agent | Type | Job |
|---|---|---|---|
| 1 | Patient Intake | Rule-based | Validates record structure, normalizes demographics/vitals/medications, flags missing fields |
| 2 | Medical NLP | **Deterministic NER** | Regex + lookup-table extraction of symptoms, conditions, medications, risk factors, family history, and negation — no model call |
| 3 | Risk Prediction | ML | Random Forest risk score (0–1) + category + top contributing features, enhanced with NLP-derived risk factors when structured workup data is missing |
| 4 | Evidence Retrieval | RAG (semantic) | `sentence-transformers` embedding search over 10 clinical guideline documents |
| 5 | Drug Safety | Rule-based | Checks 28 known interaction pairs, flags by severity |
| 6 | Guideline Verification | **Hybrid (LLM + safety net)** | LLM judges evidence relevance; a programmatic override trusts high embedding-similarity matches the LLM incorrectly rejected |
| 7 | Clinical Reasoning | LLM | Synthesizes everything into an assessment + prioritized recommendations, with a hallucination filter |
| 8 | Report Generation | Rule-based | Compiles the structured markdown report |

### Why NLP is deterministic, not LLM-based
This is a deliberate design choice, not a shortcut. Symptom/condition/medication extraction from a clinical note is a well-bounded categorization task — the vocabulary is finite and known in advance. Spending a generation pass on a 3B model for that adds latency and a small but real hallucination risk, for a task a regex + lookup-table module handles instantly, deterministically, and with zero risk of inventing a symptom that isn't in the note. The fine-tuned LLM is reserved for the two agents where the input is genuinely open-ended and judgment is actually required: **deciding whether retrieved evidence fits this specific patient**, and **synthesizing everything into a coherent clinical narrative**. Using a large model everywhere is the easy design; using it only where it earns its keep is the harder, more defensible one.

### Hybrid guideline verification, in detail
The verification agent asks the LLM to judge whether retrieved guideline snippets are actually relevant to the patient, given their similarity scores. Small local LLMs are prone to being *overly strict* — they'll sometimes reject a guideline that clearly applies just because the wording doesn't match verbatim (missing that "GERD" and "gastroesophageal reflux disease" are the same thing, for instance). Rather than trust the LLM's judgment blindly, there's a programmatic safety net: if the LLM returns *no* citations but the embedding model scored a snippet above **0.35 cosine similarity**, the pipeline overrides the LLM's rejection and trusts the semantic score instead, auto-verifying that evidence and noting in the report that this happened. This means a false rejection from the small model doesn't silently drop good evidence from the final report.

### Hybrid RAG
The evidence retrieval agent encodes all 10 clinical guideline documents once at startup using `all-MiniLM-L6-v2` (~80MB, downloaded and cached on first run), then finds the top-3 most relevant snippets for a patient's conditions via cosine similarity — understanding that "heart attack" and "myocardial infarction" mean the same thing, which keyword-only matching cannot.

### Fine-tuning approach
The two LLM-backed agents run on `Qwen2.5-3B-Instruct` fine-tuned with a LoRA adapter, trained on ~1,100 weakly-supervised examples derived from the MTSamples clinical corpus (rule-based labeling, not manual annotation — disclosed honestly rather than oversold). The adapter is merged into the base model for faster inference, with automatic fallback to base+adapter or raw base model if a merged checkpoint isn't present.

---

## 🛠️ Technical Stack

| Layer | Technology |
|---|---|
| **LLM** | `Qwen/Qwen2.5-3B-Instruct`, fine-tuned via LoRA (PEFT) — used only for Guideline Verification & Clinical Reasoning |
| **NLP extraction** | Deterministic regex/lookup-table NER (`src/utils/clinical_ner.py`) |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| **ML** | Random Forest (`scikit-learn`), `StandardScaler` pipeline |
| **Orchestration** | LangGraph (parallel state machine, custom reducers) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Vanilla JavaScript (SSE for live agent trace) |
| **Platform** | Apple Silicon (MPS), CPU/CUDA fallback |
| **Testing** | Pytest (~64 tests, 7 files, all runnable in mock mode) |

---

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.11+** (tested on 3.11–3.12)
- **macOS with Apple Silicon** recommended (MPS acceleration); CPU/CUDA fallback works, just slower
- **8GB+ RAM** (16GB recommended)

### Installation

```bash
git clone https://github.com/alhosseinjr/Clinical-Ai-Multi-Agent.git
cd Clinical-Ai-Multi-Agent

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
```

### First-time setup (optional — mock mode needs no trained adapter)

```bash
python finetune/scripts/download_dataset.py   # fetch MTSamples corpus
python finetune/scripts/prepare_dataset.py    # build training data
python finetune/scripts/train_lora.py         # fine-tune the 3B model with LoRA
python merge_3b_lora.py                       # merge adapter into base model
python scripts/train_risk_model.py --force    # train the risk model
```

On first real-mode run, two things download and cache automatically regardless of the above: the `Qwen2.5-3B-Instruct` base model (~6GB) and the `all-MiniLM-L6-v2` embedding model (~80MB).

---

## 💻 Usage

### Web UI (recommended)

```bash
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000**. Select a patient, toggle **Mock Mode** (instant, deterministic, no model needed) or **Real Mode** (full local inference), and run the pipeline. Live agent execution traces stream in via SSE — watch each of the 8 agents complete (or fail and gracefully degrade) in real time, with per-agent timing.

**First-run note:** the first Real Mode request is slower — model load plus one-time guideline embedding encoding. Subsequent requests are fast, since both stay cached in memory.

### Command line

```bash
python main.py                          # run all 12 sample patients
python main.py --patient-id P009         # run a single patient
python main.py --mock                    # mock mode, no model needed
```

Reports are saved to `outputs/report_<PATIENT_ID>.md`.

### API endpoints

```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/patients
curl -X POST http://localhost:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "P009", "mock": false}'
```

---

## 🎯 Why This Matters

- **Right tool for each job, not one tool for every job**: an LLM handles judgment calls; deterministic logic handles well-bounded extraction; a supervised model handles prediction. This keeps the system fast, reproducible, and far less prone to hallucination than an "LLM does everything" design
- **Privacy-first**: patient data never leaves the machine — no API calls to any third party, by design, not just by configuration
- **Explainable, not a black box**: every risk score comes with its top contributing features; every verified citation traces back to either the LLM's judgment or an explicit, logged semantic-score override
- **Honest about scope**: the risk model is trained on a small, decades-old, single-institution dataset, and this document says so plainly rather than burying it

---

## 📦 Project Structure

```
Clinical-Ai-Multi-Agent/
├── api/
│   └── main.py                    # FastAPI backend — REST endpoints + static file serving
├── app.py                         # Streamlit UI entrypoint (prepped for Streamlit Cloud, currently run locally)
├── web/                           # Vanilla JS console
│   ├── index.html
│   ├── styles.css
│   └── app.js                     # SSE client for live agent trace
├── src/
│   ├── agents/                    # 8 agent implementations (one file each)
│   ├── utils/
│   │   ├── llm.py                 # Local LLM load/inference wrapper + caching
│   │   ├── llm_parser.py          # Robust JSON extraction from model output
│   │   ├── clinical_ner.py        # Deterministic NER — the NLP agent's actual engine
│   │   ├── clinical_feature_mapper.py  # Maps NER output into ML model features
│   │   ├── vector_store.py        # Semantic retriever (sentence-transformers)
│   │   └── risk_model.py          # Random Forest risk pipeline
│   ├── graph.py                   # LangGraph wiring — parallel fan-out/fan-in
│   ├── state.py                   # PipelineState schema + concurrency reducers
│   └── orchestrator.py            # Per-agent timing + failure isolation
├── finetune/scripts/               # LoRA data prep + training
├── merge_3b_lora.py                 # Merges the 3B LoRA adapter into the base model
├── save_model.py / test_model.py    # Model caching + sanity-check utilities
├── data/
│   ├── sample_patients.json       # 12 synthetic patient records
│   ├── drug_interactions.json     # 28 drug interaction pairs
│   ├── guidelines/                # 10 clinical guideline .txt files
│   └── risk_model/
│       ├── heart_disease.csv      # UCI Heart Disease dataset (303 patients)
│       └── DATASET_INFO.md        # Column definitions + provenance + label-fix notes
├── scripts/train_risk_model.py     # Trains + evaluates the risk model
├── tests/                          # ~64 tests across 7 files
├── outputs/                        # Generated reports + eval reports (gitignored)
├── main.py                         # CLI entrypoint
├── requirements.txt
├── Procfile                        # Deployment config (Render/Railway)
└── README.md
```

---

## 📊 Performance & Metrics

### Risk model (Random Forest, held-out 20% test split, UCI Heart Disease Cleveland subset, 303 patients)

| Metric | Value |
|---|---|
| Accuracy | 0.803 |
| ROC-AUC | 0.921 |
| Precision | 0.833 |
| Recall | 0.714 |
| F1 | 0.769 |

**Honest read**: on a dataset this small (303 rows), these numbers carry real run-to-run variance. A recall of 0.714 means roughly 1 in 4 positive cases in the test set is missed — worth stating plainly for anything framed as risk stratification, even a demo. ROC-AUC of 0.921 indicates strong overall class separation; the accuracy/recall gap mostly reflects where the classification threshold sits, not the model's underlying discriminative power. Feature importance for a given prediction is read from the trained model's own Gini-importance ranking rather than a fixed, hardcoded list.

### Retrieval quality
Semantic embeddings move retrieval from exact keyword overlap to meaning-based similarity — the practical difference shows up on queries where clinical terminology varies (e.g., a note mentioning "MI" or "myocardial infarction" reliably retrieves the same guideline that keyword matching would only catch on exact term overlap).

### Guideline verification safety net
The 0.35 cosine-similarity threshold for the hybrid override was chosen empirically as the point where the embedding model's confidence reliably indicates true relevance — high enough to avoid false positives from loosely related snippets, low enough to catch the synonym cases (e.g. "GERD" vs. "gastroesophageal reflux disease") that trip up the small LLM's stricter judgment.

### Inference
- Model: `Qwen2.5-3B-Instruct` (~6GB), MPS-accelerated on Apple Silicon, used only for 2 of the 8 agents
- Greedy decoding (no sampling) to keep JSON output stable, with repetition penalty to avoid degenerate loops
- Response caching by prompt hash — identical requests within a session skip regeneration entirely
- Deterministic NER runs in negligible time compared to any LLM call, since it's pure regex/dictionary lookup with no model load

---

## 🌐 Deployment

**Current status: local-only.** The system runs via `uvicorn` for the FastAPI web console. `app.py` is a complete Streamlit entrypoint, written and ready for Streamlit Cloud, but is not currently deployed there — run it locally with `streamlit run app.py` if you want that interface today.

### Render / Railway (PaaS)

```bash
# Procfile (already included)
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Steps: push to GitHub → create a Web Service on Render/Railway → connect the repo → build command `pip install -r requirements.txt` → start command `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.

⚠️ **Memory warning**: `Qwen2.5-3B` needs ~5–6GB RAM; a 512MB–1GB free tier will OOM. Options: run in mock mode (`MOCK_MODE=true`), upgrade to a 4GB+ tier, or quantize the model (GGUF/int8).

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Local network sharing

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
ipconfig getifaddr en0   # find your Mac's IP
# share http://<your-ip>:8000 with teammates on the same network
```

---

## 🔧 Configuration

### Environment variables (`.env`)

```bash
BASE_MODEL_ID=Qwen/Qwen2.5-3B-Instruct
LORA_ADAPTER_PATH=models/clinical-lora-adapter-3b
MERGED_MODEL_PATH=models/clinical-3b-merged
MOCK_MODE=false
LOG_LEVEL=INFO
```

### Code-level customization

Swap the model (`src/utils/llm.py`):
```python
base_model_id = "Qwen/Qwen2.5-7B-Instruct"  # larger, slower
```

Add a guideline — drop a `.txt` file in `data/guidelines/`; the retriever auto-encodes it on next startup, no code change needed.

Add a drug interaction (`data/drug_interactions.json`):
```json
{"drug_a": "ibuprofen", "drug_b": "lisinopril", "severity": "moderate",
 "description": "NSAIDs may reduce ACE inhibitor effectiveness..."}
```

Tune the verification safety net (`src/agents/guideline_agent.py`):
```python
threshold = 0.35  # cosine similarity floor for auto-verifying evidence the LLM rejected
```

---

## 🔮 Future Enhancements

**Near-term**
- Hand-annotate a small eval set to measure NLP extraction precision/recall directly, rather than relying on the deterministic NER's keyword coverage alone
- Persist run history (SQLite instead of flat markdown files)

**Medium-term**
- Deploy the existing `app.py` to Streamlit Cloud for a public hosted demo
- Quantize the model to GGUF/int8 to fit free-tier cloud RAM
- Multi-modal inputs — ECG image classification, lab-result PDF parsing, voice intake via Whisper
- EHR integration (Epic/Cerner FHIR APIs) instead of standalone JSON patient records

**Long-term**
- Replace weak-supervision fine-tuning labels with actual clinical annotation (measure inter-annotator agreement)
- HIPAA-compliant cloud deployment
- Clinician-in-the-loop UI — editable recommendations, confidence scores per extraction, full audit trail
- Additional clinical specialties beyond cardiovascular risk

---

## 📚 Dataset Provenance & Ethics

### MTSamples clinical corpus
- Source: [mtsamples.com](https://www.mtsamples.com) — public medical transcription samples, free for educational/research use
- ~5,000 real, de-identified clinical notes (SOAP format), covering cardiology, pulmonology, gastroenterology, endocrinology, and more
- No PHI: all patient identifiers removed by MTSamples; not synthetic — real clinical documentation, used here for educational fine-tuning
- Labels are weak-supervision (rule-based), not clinician-annotated — estimated ~70–80% accuracy, not formally validated. Use case: portfolio demo, not clinical deployment

### UCI Heart Disease dataset
- Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease), public domain
- 303 patients, Cleveland Clinic, 13 features + binary target, collected 1988
- Limitations: small sample, single center, ~40 years old, treatment guidelines have since changed, skewed demographics (78% male, mostly white)
- Use case: educational demo of a real ML pipeline, not a clinically validated predictor
- **Data quality catch**: the original source mirror had an inverted target label (`1` meant "no disease"). This surfaced by sanity-checking learned feature directions against clinical priors — cholesterol, max heart rate, and several other features were *all* correlating with the label in the clinically wrong direction simultaneously, which pointed at a flipped label rather than seven independent violations of cardiology. Confirmed against an independent mirror of the same cohort, corrected, and retrained. Full details in `data/risk_model/DATASET_INFO.md`.

---

## Contributing

Contributions welcome.

**Reporting bugs** — include:
```bash
python --version
pip list | grep -E "langgraph|transformers|torch|sentence-transformers"
uname -a
```

**Adding features**: fork → `git checkout -b feature/your-feature` → add tests in `tests/` → update this README if relevant → submit a PR.

**Code style**: type hints throughout, Google-style docstrings.

---

## 📄 License & Disclaimer

MIT License — see `LICENSE`. Free to use, modify, and include in your own portfolio; keep the copyright notice and license text with any distribution.

**Medical disclaimer**: this software is for educational and portfolio purposes only. It is not a certified medical device, has not been validated for clinical use, and must not be used to diagnose, treat, or make decisions about real patients. Consult qualified healthcare professionals for actual medical decisions.

---

## 🙏 Acknowledgments

LangChain/LangGraph team · Qwen team (Alibaba) · MTSamples · UCI ML Repository · Hugging Face (`transformers`, `sentence-transformers`)

---

## 👨‍💻 Author

**Al-Hossein Mahmoud** ([@alhosseinjr](https://github.com/alhosseinjr))
- GitHub: [github.com/alhosseinjr](https://github.com/alhosseinjr)
- Portfolio: [alhossein.site](https://alhossein.site)
- Email: al7ossein@gmail.com

---

## 📞 Support

Questions? [Open an issue](https://github.com/alhosseinjr/Clinical-Ai-Multi-Agent/issues). Want to collaborate? Reach out via the contact above.

<div align="center">

⭐ Star this repo if you found it useful

Built with Python • LangGraph • FastAPI • Qwen • sentence-transformers • Apple Silicon

</div>
