# Clinical AI Multi-Agent Pipeline

A LangGraph-orchestrated, multi-agent clinical decision-support **demo**
pipeline, built as a practical internship/grad portfolio project. It shows
end-to-end multi-agent orchestration: a locally fine-tuned LLM, a trained
ML model, a local RAG retriever, rule-based safety checks, and structured
report generation, wired together with LangGraph.

> ⚠️ **This is a software engineering demo, not a medical product.** All
> guideline text and drug interaction data in `data/` is synthetic/simplified
> and clearly labeled as such. Nothing in this repo should ever be used for
> real clinical decisions.

## Architecture

```
Patient Intake Agent        -- normalizes + validates raw patient record
        │
        ▼
Medical NLP Agent           -- extracts symptoms/conditions/meds from notes (local LLM)
        │
        ▼
Risk Prediction Agent (ML)  -- logistic regression trained on the real UCI Heart Disease dataset
        │
        ▼
Evidence Retrieval Agent    -- TF-IDF RAG over 10 local guideline snippets
        │
        ▼
Drug Safety Agent           -- rule-based check against interaction dataset
        │
        ▼
Guideline Verification      -- local LLM checks retrieved evidence actually fits
        │
        ▼
Clinical Reasoning Agent    -- local LLM synthesizes everything into an assessment
        │
        ▼
Report Generation Agent     -- compiles a markdown report
```

Each agent is a plain Python function `run(state) -> dict`, wired together
as nodes in a LangGraph `StateGraph` (`src/graph.py`). State flows through
a single shared `PipelineState` TypedDict (`src/state.py`), so any agent can
be inspected, swapped, or extended independently.

## Local LLM

The 3 LLM-backed agents (NLP extraction, guideline verification, clinical
reasoning) no longer call an external API. They run a small model,
**Qwen2.5-0.5B-Instruct**, fine-tuned locally with **LoRA** (via PEFT) on
Apple Silicon (MPS backend) — no API key, no network calls at inference
time, no per-call cost.

### Dataset & labeling methodology (read this before you present it)

The fine-tuning data is derived from **MTSamples**, a public corpus of
~5,000 real medical transcription samples spanning most clinical
specialties (SOAP-style notes: Subjective, Medications, Assessment, Plan,
etc.). These are real authored clinical documentation used for
transcriptionist training/reference — not synthetic text and not real
patient records (no PHI).

`finetune/scripts/prepare_dataset.py` turns that real text into 3
instruction-tuning datasets. Being transparent about how, since this
matters for how you talk about it:

- **Free-text targets are real, not generated.** The `assessment` and
  `recommendations` fields the model is trained to produce are copied and
  trimmed directly from each note's own ASSESSMENT/IMPRESSION and PLAN
  sections — never authored by a script or an LLM.
- **Known data-quality fix (Jul 2026):** MTSamples encodes multi-item
  sections using `.,` as a pseudo-linebreak (e.g. `1. Finding one.,2.
  Finding two.,PLAN: ,Continue management.`). An early version of
  `prepare_dataset.py` only filtered assessment text with *more than 3*
  numbered items, so shorter comma-joined sections slipped into training
  targets uncleaned, and the fine-tuned model learned to reproduce that
  broken punctuation (including leaking a trailing `PLAN:` fragment into
  the assessment sentence). `prepare_dataset.py` now runs
  `clean_assessment_text()` on every target before it's written, and
  `src/agents/reasoning_agent.py` additionally cleans every real (non-mock)
  generation at inference time as a safety net — so this holds even for
  the already-trained adapter, without requiring a retrain. Recommendation
  lists are also deduped (token-overlap, not exact-match) since the small
  model sometimes restates the same instruction two ways (e.g. "Follow up
  EKG results" / "We will check the EKG"). See
  `tests/test_reasoning_agent.py` for the regression tests.
- **Structured labels (entities, alignment, priority) are rule-based, not
  annotated.** They come from fixed keyword/regex lookups (a curated drug
  list, a curated condition list, a curated symptom list) applied to that
  same real text, with basic negation handling (e.g. "no vomiting" does
  not tag "vomiting" as a symptom).
- **This is weak supervision, not clinical ground truth.** It's enough to
  teach a small model the task *format* and general extraction behavior
  for a portfolio demo. It has not been validated by a clinician and will
  make mistakes a real annotated dataset wouldn't. Say this plainly if
  anyone asks how the model was trained — it's the honest and, frankly,
  more interesting answer than pretending it's a clean labeled dataset.

Yields on the full MTSamples corpus (your numbers will vary slightly if
you regenerate): **400 NLP-extraction examples, ~330 guideline-verification
examples, 400 clinical-reasoning examples** (~1,100 total, 90/10 train/val
split).

### Training on your M4

```bash
# 1. Download the real dataset (~17 MB, one-time)
python finetune/scripts/download_dataset.py

# 2. Build the 3 instruction-tuning datasets from it
python finetune/scripts/prepare_dataset.py

# 3. Fine-tune (LoRA on Qwen2.5-0.5B-Instruct, MPS backend)
python finetune/scripts/train_lora.py
```

Step 3 downloads the ~1GB base model from Hugging Face on first run, then
trains. On an M4 with the default settings (~1,100 examples, 3 epochs),
expect roughly **20–40 minutes**. The adapter (a few MB, not a full model
copy) is saved to `models/clinical-lora-adapter/`.

Optional — merge the adapter into the base model for a slightly faster,
single-folder inference setup:

```bash
python finetune/scripts/merge_lora.py
```

Once either the adapter or the merged model exists, `main.py` and the web
UI pick it up automatically — no config needed. `.env.example` documents
the override variables (`BASE_MODEL_ID`, `LORA_ADAPTER_PATH`,
`MERGED_MODEL_PATH`) if you want to point at a different checkpoint.

Everything still runs in `--mock` mode with zero setup if you just want to
demo the orchestration without training first.

## Risk Prediction Model

The Risk Prediction Agent trains a logistic regression model on the real,
public **UCI Heart Disease dataset** (Cleveland Clinic subset, 303 patients,
13 clinical features) instead of data generated at runtime from a
hand-authored formula. See `data/risk_model/DATASET_INFO.md` for full column
definitions and scope notes.

**Result (held-out 20% test set):** 85.2% accuracy, 0.924 ROC-AUC. Full
metrics (precision, recall, F1, confusion matrix) are in
`outputs/risk_model_eval.md`, regenerated by:

```bash
python scripts/train_risk_model.py --force
```

`requirements.txt` pins `scikit-learn==1.8.0` to match the version
`models/risk_model.joblib` was pickled with — an unpinned `>=1.4.0` let a
fresh `pip install` grab a newer sklearn than the pickle, which throws an
`InconsistentVersionWarning` on every load even though it still works. If
you retrain with a newer sklearn, bump this pin to match.

**A data quality catch worth calling out:** the original source mirror this
dataset came from had its target label inverted (`1` meant "no disease"),
which produced a deceptively plausible ~80% accuracy while every clinical
feature (age, cholesterol, exercise-induced angina, etc.) correlated with
outcome in the wrong direction. This was caught by sanity-checking learned
feature directions against clinical priors rather than trusting the column
name, confirmed against an independent mirror of the same cohort, and
corrected -- see the "Data quality note" in `DATASET_INFO.md` for the full
story. Worth mentioning if asked how the model was validated.

Several features (`thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`) come
from a cardiac stress test / catheterization workup rather than a basic
intake visit -- `sample_patients.json` models this as a `cardiac_workup`
block per patient. A patient record missing this block still runs (defaults
to 0, flagged in the pipeline trace as a possible risk understatement)
rather than crashing.

## Project layout

```
clinical-ai-pipeline/
├── main.py                    # CLI entrypoint
├── requirements.txt
├── Procfile                    # for Render/Railway-style deploys
├── .env.example
├── api/
│   └── main.py                 # FastAPI backend (wraps the pipeline)
├── web/
│   ├── index.html               # console UI
│   ├── styles.css
│   └── app.js
├── data/
│   ├── sample_patients.json   # 12 synthetic sample patients (diverse ages/comorbidities/polypharmacy)
│   ├── drug_interactions.json # 28-pair sample interaction dataset
│   ├── guidelines/            # 10 guideline .txt snippets (RAG corpus)
│   └── risk_model/
│       ├── heart_disease.csv    # real UCI Heart Disease dataset (label-corrected, see DATASET_INFO.md)
│       └── DATASET_INFO.md      # column definitions, provenance, data quality note
├── scripts/
│   └── train_risk_model.py    # trains + evaluates the risk model, writes outputs/risk_model_eval.md
├── finetune/
│   ├── scripts/
│   │   ├── download_dataset.py  # pulls the real MTSamples corpus
│   │   ├── prepare_dataset.py   # derives 3 task datasets from it
│   │   ├── train_lora.py        # LoRA fine-tune on Qwen2.5-0.5B (MPS)
│   │   └── merge_lora.py        # optional: merge adapter into base
│   └── data/                    # train.jsonl / val.jsonl land here
├── models/                      # trained adapter / merged model land here
├── src/
│   ├── state.py                # shared PipelineState schema
│   ├── graph.py                 # LangGraph wiring
│   ├── agents/                  # one file per pipeline stage
│   └── utils/
│       ├── llm.py               # local LLM wrapper (+ mock mode)
│       ├── vector_store.py      # local TF-IDF retriever
│       └── risk_model.py        # synthetic-data-trained sklearn model
├── tests/
│   └── test_pipeline.py
└── outputs/                    # CLI-generated reports land here
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# defaults work out of the box once you've trained the adapter (see above)
```

## Usage

Run the pipeline for all sample patients:

```bash
python main.py
```

Run a single patient:

```bash
python main.py --patient-id P001
```

Run without the local model (canned/mock output, useful for demos or CI):

```bash
python main.py --mock
```

Run against your own patient data:

```bash
python main.py --file path/to/your_patients.json
```

Each run writes one markdown report per patient to `outputs/report_<ID>.md`.

## Testing

```bash
pytest -v
```

21 tests across 5 files, all running in mock mode (no local model, GPU, or
network access needed):

- `test_pipeline.py` -- end-to-end graph execution across all 12 sample
  patients, plus targeted checks (polypharmacy patient flags an interaction,
  healthy baseline patient isn't scored high-risk).
- `test_risk_model.py` -- dataset shape, eval metric sanity thresholds
  (guards against silently regressing to a near-random model), prediction
  behavior.
- `test_intake_agent.py` -- validation behavior on missing/malformed fields
  (missing patient_id, implausible age, invalid sex value, wrong-typed
  medications list) -- confirms these are flagged, not silently ignored or
  allowed to crash the pipeline.
- `test_drug_safety_agent.py` -- interaction flagging, no false positives on
  unrelated medications, no duplicate pairs in the dataset.
- `test_retrieval_agent.py` -- all 10 guideline files present and
  retrievable, retriever distinguishes unrelated conditions correctly.

## Design notes / talking points

- **Why LangGraph:** the pipeline is linear today, but modeling it as a
  graph (rather than a chain of function calls) means branching (e.g. skip
  drug safety if no medications), parallel fan-out (e.g. NLP + risk running
  concurrently), or human-in-the-loop review nodes can be added without
  restructuring agent code.
- **Why a real (if toy) ML model instead of a lookup table:** the Risk
  Prediction Agent trains a logistic regression on synthetic labeled data
  at startup, so `risk_model.py` is a genuine example of an ML component
  inside an agent pipeline, not a hardcoded stand-in.
- **Why fine-tune locally instead of calling an API:** demonstrates the
  full loop — real-data sourcing, weak-supervision labeling, LoRA
  fine-tuning, and local inference — instead of treating the LLM as a black
  box behind an API key. Trade-off worth naming out loud: a 0.5B model
  fine-tuned on ~1,100 weakly-labeled examples will not match a frontier
  model's reasoning quality; the point of this version is the pipeline
  engineering and training loop, not state-of-the-art accuracy.
- **Why TF-IDF instead of embeddings for RAG:** keeps the project fully
  offline/dependency-light (no separate embedding model download required)
  while still demonstrating the retrieve-then-verify RAG pattern. Swapping
  in a real embedding store (e.g. Chroma + sentence-transformers) only
  requires changing `src/utils/vector_store.py`.
- **Mock mode:** every LLM-backed agent accepts `mock_mode` via shared
  state and falls back to deterministic output. This keeps the whole graph
  testable and demoable without a trained model or a GPU.

## Web UI

A FastAPI backend (`api/main.py`) wraps the pipeline and serves a small
static frontend (`web/`) — a console for picking a sample patient, running
the pipeline, watching each agent complete live, and reading the resulting
report as a formatted dashboard instead of raw markdown.

Run it locally:

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Then open **http://127.0.0.1:8000**.

- If no local model is found (no trained adapter, no merged model), the UI
  automatically forces mock mode and shows a status indicator saying so.
- `GET /api/health`, `GET /api/patients`, `POST /api/run` are the underlying
  endpoints — usable directly (e.g. from `curl` or Postman) if you want to
  script against the pipeline instead of using the UI.

## Deploying

The app is a single FastAPI service (frontend is served as static files
from the same process), which makes it a standard Python web service to
deploy. A `Procfile` is included for platforms that use one.

> **Note on hosting cost/size:** now that inference runs locally instead of
> via an API, the deployed instance needs enough RAM to hold the model
> (Qwen2.5-0.5B in fp32 is roughly 2GB of weights, plus overhead). Typical
> free tiers (e.g. Render's 512MB free web service) will not fit this —
> either deploy in `--mock` mode for a UI-only demo, use a paid tier with
> more memory, or quantize the merged model before deploying.

**Render / Railway:**
1. Push the repo to GitHub (see below).
2. Create a new **Web Service** from the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Either ship `models/clinical-llm-merged/` with the deploy (large — consider
   Git LFS or a build-time download step) on a large-enough instance, or
   accept mock-mode-only on a free tier.

**Fly.io / any Docker host:** same start command works inside a standard
`python:3.12-slim` image with `requirements.txt` installed.

## Ideas for taking this further

- Replace the weak-supervision labels with a small hand-annotated
  validation set to actually measure extraction accuracy, not just eyeball
  it.
- Add conditional edges in `graph.py` (e.g. skip Drug Safety when the
  patient has no active medications).
- Try a slightly larger base model (Qwen2.5-1.5B-Instruct) once the data
  pipeline is validated, and compare JSON-validity / task accuracy against
  the 0.5B run.
- Persist run history / reports to a small database instead of flat files.
- Quantize the merged fine-tuned model (e.g. GGUF/int8) so it fits a
  free-tier deployment's RAM budget instead of requiring a paid instance.
- Try a second risk-model dataset (e.g. a diabetes-specific cohort) and
  compare which one the demo patient population is better matched to.
- Add structured logging in place of print statements, and wrap the local
  LLM calls in explicit error handling (missing adapter, OOM) instead of
  letting them surface as raw exceptions.
