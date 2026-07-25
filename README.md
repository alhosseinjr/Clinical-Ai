# 🏥 Clinical AI Multi-Agent Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.0-green.svg)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-lightgrey.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, **locally-hosted clinical decision support system** built with multi-agent orchestration. This end-to-end pipeline demonstrates advanced AI engineering: fine-tuned language models running on Apple Silicon, RAG-based evidence retrieval, rule-based drug safety checks, and ML-powered risk prediction — all orchestrated through LangGraph.

> ⚠️ **IMPORTANT DISCLAIMER**: This is a **software engineering portfolio demo**, not a medical device. All clinical guidelines, drug interactions, and patient data are synthetic or simplified for educational purposes. **Never use this system for real clinical decision-making.**

---

## Key Features

- **🤖 Multi-Agent Architecture**: 8 specialized agents working in concert via LangGraph state machine
- **Local LLM Inference**: Qwen2.5-3B-Instruct running on Apple Silicon (MPS) — zero API calls, zero latency, zero cost
- **📚 RAG-Powered Evidence Retrieval**: TF-IDF vector store over 10 clinical guideline documents
- **💊 Drug Safety Engine**: Rule-based interaction checker with severity levels
- **ML Risk Prediction**: Logistic regression model trained on real UCI Heart Disease dataset (85.2% accuracy)
- **🎯 LoRA Fine-Tuning**: Custom adapter trained on ~1,100 weakly-supervised clinical examples
- **🖥️ Full-Stack UI**: FastAPI backend + modern web console with real-time agent tracing
- **Privacy-First**: Runs 100% offline — no data leaves your machine

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ PATIENT INTAKE AGENT                                                 │
│ • Validates patient record structure                                 │
│ • Normalizes demographics, vitals, medications                       │
│ • Flags missing critical fields                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ MEDICAL NLP AGENT (LLM)                                              │
│ • Extracts symptoms from free-text clinical notes                    │
│ • Identifies mentioned conditions & medications                      │
│ • Detects notable flags (family history, social factors)             │
│ • Model: Qwen2.5-3B-Instruct (local, fine-tuned)                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ RISK PREDICTION AGENT (ML)                                           │
│ • Logistic regression on 13 clinical features                        │
│ • Trained on UCI Heart Disease dataset (Cleveland Clinic)             │
│ • Outputs: risk score (0-1), risk category (low/medium/high)          │
│ • Performance: 85.2% accuracy, 0.924 ROC-AUC                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ EVIDENCE RETRIEVAL AGENT (RAG)                                       │
│ • TF-IDF semantic search over 10 clinical guidelines                 │
│ • Retrieves top-3 relevant guideline snippets                        │
│ • Covers: hypertension, diabetes, COPD, AFib, CKD, etc.               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ DRUG SAFETY AGENT                                                    │
│ • Rule-based interaction checker                                     │
│ • 28 drug pairs with severity levels (low/moderate/high)              │
│ • Flags: warfarin+allopurinol, lisinopril+furosemide, etc.            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ GUIDELINE VERIFICATION (LLM)                                         │
│ • Validates retrieved evidence matches patient conditions             │
│ • Prevents hallucinated or irrelevant citations                      │
│ • Outputs: aligned=True/False, citation list                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ CLINICAL REASONING AGENT (LLM)                                       │
│ • Synthesizes all agent outputs into assessment                      │
│ • Generates prioritized recommendations                              │
│ • Priority classification: routine/urgent/emergent                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ REPORT GENERATION AGENT                                              │
│ • Compiles structured markdown report                                │
│ • Includes: patient profile, extracted entities, risk factors,       │
│   drug safety alerts, evidence citations, recommendations             │
│ • Exports to outputs/report_<PATIENT_ID>.md                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Project Structure

```
clinical-ai-pipeline/
├── api/                          # FastAPI backend
│   └── main.py                   # REST API endpoints + static file serving
├── web/                          # Frontend console UI
│   ├── index.html                # Main dashboard
│   ├── styles.css                # Modern, responsive styling
│   └── app.js                    # Client-side logic + SSE for live updates
├── src/                          # Core pipeline logic
│   ├── agents/                   # Individual agent implementations
│   │   ├── intake_agent.py       # Patient data validation
│   │   ├── nlp_agent.py          # Symptom/condition extraction
│   │   ├── risk_agent.py         # ML risk prediction
│   │   ├── retrieval_agent.py    # RAG evidence search
│   │   ├── drug_safety_agent.py  # Interaction checking
│   │   ├── guideline_agent.py    # Evidence verification
│   │   ├── reasoning_agent.py    # Clinical assessment
│   │   └── report_agent.py       # Markdown report generation
│   ├── utils/                    # Shared utilities
│   │   ├── llm.py                # Local LLM wrapper + caching
│   │   ├── llm_parser.py         # Robust JSON extraction from LLM output
│   │   ├── vector_store.py       # TF-IDF retriever
│   │   └── risk_model.py         # Sklearn logistic regression wrapper
│   ├── graph.py                  # LangGraph StateGraph definition
│   └── state.py                  # PipelineState TypedDict schema
├── finetune/                     # LoRA fine-tuning pipeline
│   ├── scripts/
│   │   ├── download_dataset.py   # Fetches MTSamples corpus (17MB)
│   │   ├── prepare_dataset.py    # Creates 3 task-specific datasets
│   │   ├── train_lora.py         # LoRA training on Apple Silicon
│   │   └── merge_lora.py         # Merges adapter into base model
│   └── data/                     # Generated train/val JSONL files
├── data/                         # Datasets & knowledge bases
│   ├── sample_patients.json      # 12 synthetic patient records
│   ├── drug_interactions.json    # 28 drug interaction pairs
│   ├── guidelines/               # 10 clinical guideline .txt files
│   └── risk_model/
│       ├── heart_disease.csv     # UCI Heart Disease dataset (303 patients)
│       └── DATASET_INFO.md       # Column definitions + provenance
├── models/                       # Trained models (auto-populated)
│   ├── clinical-lora-adapter/    # LoRA adapter (5MB)
│   ├── Qwen2.5-3B-Instruct/      # Base model (~6GB, downloaded on first run)
│   └── risk_model.joblib         # Sklearn model + vectorizer
├── scripts/                      # Utility scripts
│   └── train_risk_model.py       # Trains + evaluates risk model
├── tests/                        # Pytest test suite
│   ├── test_pipeline.py          # End-to-end integration tests
│   ├── test_llm_parser.py        # JSON parsing robustness tests
│   ├── test_intake_agent.py      # Input validation tests
│   ├── test_drug_safety_agent.py # Interaction detection tests
│   └── test_retrieval_agent.py   # RAG accuracy tests
├── outputs/                      # Generated reports (gitignored)
├── main.py                       # CLI entrypoint
├── requirements.txt              # Python dependencies
├── Procfile                      # Deployment config (Render/Railway)
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (tested on 3.11-3.12)
- **macOS with Apple Silicon** (M1/M2/M3/M4) for MPS acceleration
  - *Intel Macs work too but will be slower (CPU fallback)*
- **8GB+ RAM** (16GB recommended for smooth inference)
- **Git** (for cloning the repo)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/alhosseinjr/Clinical-Ai.git
cd clinical-ai-pipeline

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment template (optional, defaults work out-of-the-box)
cp .env.example .env
```

### First-Time Setup (Optional but Recommended)

```bash
# Download the fine-tuning dataset (MTSamples, ~17MB)
python finetune/scripts/download_dataset.py

# Prepare training data (~1,100 examples across 3 tasks)
python finetune/scripts/prepare_dataset.py

# Fine-tune Qwen2.5-0.5B with LoRA (~20-40 min on M4)
python finetune/scripts/train_lora.py

# (Optional) Merge adapter into base model for faster inference
python finetune/scripts/merge_lora.py

# Train the risk prediction model
python scripts/train_risk_model.py --force
```

---

## 💻 Usage

### Option 1: Web UI (Recommended for Demos)

```bash
# Start the FastAPI server
uvicorn api.main:app --reload --port 8000

# Open your browser to:
http://localhost:8000
```

Features:
- Interactive patient selection dropdown
- Toggle between Mock Mode (instant, no model) and Real Mode (local LLM)
- Live agent execution trace (watch each step complete in real-time)
- Beautiful markdown report rendering
- Drug safety alerts highlighted in red/orange/green
- Evidence citations with relevance scores

### Option 2: Command Line Interface

```bash
# Run all 12 sample patients
python main.py

# Run a single patient
python main.py --patient-id P009

# Run with mock mode (no local model needed)
python main.py --mock

# Run with custom patient file
python main.py --file path/to/my_patient.json
```

Reports are saved to `outputs/report_<PATIENT_ID>.md`.

### Option 3: API Endpoints

```bash
# Health check
curl http://localhost:8000/api/health

# List available patients
curl http://localhost:8000/api/patients

# Run pipeline for patient P009
curl -X POST http://localhost:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "P009", "mock": false}'
```

---

## 🧠 Local LLM Details

### Model Selection

**Current Model**: `Qwen/Qwen2.5-3B-Instruct`

Why this model?
- **Size**: 3B parameters (~6GB on disk, ~4-5GB RAM usage)
- **Speed**: ~50-100 tokens/sec on M4 MPS backend
- **Quality**: Dramatically better JSON formatting and instruction following vs 0.5B
- **Cost**: Free, open-source, no API keys required

Alternative Models (swap in `src/utils/llm.py`):
- `Qwen/Qwen2.5-1.5B-Instruct` — Faster (100-150 tok/sec), lower RAM (2-3GB)
- `microsoft/Phi-3-mini-4k-instruct` — 3.8B, excellent instruction following
- `Qwen/Qwen2.5-0.5B-Instruct` — Original fine-tuned model (requires LoRA adapter)

### Fine-Tuning Process

**Dataset**: MTSamples clinical transcription corpus (~5,000 real medical notes)

**Methodology**:
- Weak Supervision: Rule-based labeling (no manual annotation)
  - Symptoms: keyword matching with negation handling
  - Conditions: curated medical condition list
  - Medications: curated drug list
  - Assessments: extracted directly from note ASSESSMENT sections
  - Recommendations: extracted from note PLAN sections

**Data Quality Fix (Jul 2026)**:
- MTSamples uses `.,` as pseudo-linebreaks in multi-item sections
- Early versions leaked `PLAN:` fragments into assessments
- Fixed via `clean_assessment_text()` in `prepare_dataset.py`
- Additional safety net in `reasoning_agent.py` at inference time

**Training Config**:
- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Method: LoRA (Low-Rank Adaptation) via PEFT
- Backend: Apple Silicon MPS (Metal Performance Shaders)
- Epochs: 3
- Train/val split: 90/10
- Total examples: ~1,100 (400 NLP + 330 verification + 400 reasoning)
- Training time: ~20-40 min on M4
- Output: `models/clinical-lora-adapter/` (~5MB adapter weights)

### Inference Optimization

```python
# In src/utils/llm.py

# Cached model loading (loads once, reuses across requests)
_model = None

def _load_model():
    global _model
    if _model is not None:
        return _model

    # Load with MPS acceleration
    _model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        torch_dtype=torch.bfloat16,  # MPS-optimized dtype
        device_map="auto"            # Automatic GPU/CPU placement
    )
    return _model

# Response caching (identical prompts return cached result)
_response_cache: dict = {}

def call_llm(system_prompt, user_prompt, max_tokens=500):
    cache_key = hashlib.sha256(
        f"{system_prompt}{user_prompt}{max_tokens}".encode()
    ).hexdigest()

    if cache_key in _response_cache:
        return _response_cache[cache_key]

    # ... generate response ...
    _response_cache[cache_key] = response
    return response
```

---

## 📊 Risk Prediction Model

**Dataset**: UCI Heart Disease (Cleveland Clinic)
- Size: 303 patients
- Features: 13 clinical variables
  - Demographics: age, sex
  - Vitals: resting BP, cholesterol
  - ECG: resting ECG results, max heart rate, exercise angina
  - Stress test: oldpeak (ST depression), slope (ST slope)
  - Catheterization: major vessels colored, thalassemia type
- Target: Presence/absence of coronary artery disease (0-4 scale, binarized)

### Model Performance

Test Set (20% holdout):
- Accuracy: 85.2%
- ROC-AUC: 0.924
- Precision: 0.83
- Recall: 0.81
- F1 Score: 0.82

Feature Importance (top 5):
1. `cp` (chest pain type) — strongest predictor
2. `thalach` (max heart rate) — inverse correlation
3. `oldpeak` (ST depression) — positive correlation
4. `ca` (major vessels colored) — positive correlation
5. `exang` (exercise-induced angina) — positive correlation

### Data Quality Note

**Critical Bug Discovered**: Original dataset mirror had inverted labels (1 = "no disease"). This caused:
- Deceptively high accuracy (~80%)
- All clinical features correlating in wrong direction
- Example: Higher cholesterol → lower disease risk (clinically impossible)

**Detection Method**: Sanity-checking learned feature directions against clinical priors (e.g., "cholesterol should positively correlate with heart disease risk").

**Resolution**: Confirmed against independent dataset mirror, corrected labels, retrained model.

---

## 🧪 Testing

```bash
# Run full test suite
pytest -v

# Run specific test file
pytest tests/test_llm_parser.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Coverage

21 tests across 5 files:

- `test_pipeline.py` (8 tests)
  - End-to-end execution for all 12 patients
  - Polypharmacy patient flags drug interactions
  - Healthy baseline patient not scored high-risk
  - Missing cardiac workup handled gracefully
- `test_llm_parser.py` (7 tests)
  - Valid JSON parsing
  - Markdown fence extraction
  - Smart quote normalization
  - Trailing comma repair
  - Nested object/array handling
  - Python dict → JSON conversion (`ast.literal_eval`)
  - Key normalization (fixes model output errors)
- `test_intake_agent.py` (3 tests)
  - Missing patient_id → error flag
  - Implausible age (e.g., 200 years) → warning
  - Invalid sex value → normalization
- `test_drug_safety_agent.py` (2 tests)
  - Known interaction detected (warfarin + allopurinol)
  - Unrelated drugs → no false positive
- `test_retrieval_agent.py` (1 test)
  - All 10 guidelines retrievable
  - Unrelated condition query returns low relevance

All tests run in mock mode — no GPU, no model download, no network access required.

---

## 🌐 Deployment

### Option 1: Render / Railway (PaaS)

```bash
# Procfile (already included)
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Steps:
1. Push repo to GitHub
2. Create new Web Service on Render/Railway
3. Connect GitHub repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

⚠️ **Memory Warning**:
- Qwen2.5-3B requires ~5-6GB RAM
- Free tier (512MB-1GB) will crash with OOM

Solutions:
- Deploy in mock mode (set `MOCK_MODE=true` in env vars)
- Upgrade to paid tier (4GB+ RAM)
- Quantize model to GGUF/int8 (advanced)

### Option 2: Docker (Fly.io, AWS, GCP)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download model at build time (optional, speeds up startup)
RUN python -c "from transformers import AutoModelForCausalLM; \
               AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build & Run:

```bash
# Make server accessible on local network
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Find your Mac's IP address
ipconfig getifaddr en0

# Share with teammates: http://192.168.1.XXX:8000
```

### Option 3: Local Network Sharing

```bash
# Make server accessible on local network
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Find your Mac's IP address
ipconfig getifaddr en0

# Share with teammates: http://192.168.1.XXX:8000
```

---

## 🔧 Configuration

### Environment Variables (`.env`)

```bash
# Model configuration
BASE_MODEL_ID=Qwen/Qwen2.5-3B-Instruct
LORA_ADAPTER_PATH=models/clinical-lora-adapter
MERGED_MODEL_PATH=models/clinical-llm-merged

# API keys (optional, for external services)
# ANTHROPIC_API_KEY=sk-...  # If you want to test Claude instead
# OPENAI_API_KEY=sk-...      # If you want to test GPT-4

# Mock mode override (set to "true" to force mock responses)
MOCK_MODE=false

# Logging level
LOG_LEVEL=INFO
```

### Code-Level Customization

Change model:

```python
# src/utils/llm.py
base_model_id = "Qwen/Qwen2.5-7B-Instruct"  # Larger, slower
```

Add new guideline:

```bash
# 1. Create new file in data/guidelines/
echo "Topic: Asthma - General Overview..." > data/guidelines/asthma.txt

# 2. Retriever auto-detects it (no code change needed)
```

Add drug interaction:

```json
// data/drug_interactions.json
{
  "drug1": "ibuprofen",
  "drug2": "lisinopril",
  "severity": "moderate",
  "description": "NSAIDs may reduce ACE inhibitor effectiveness..."
}
```

---

## 🎯 Key Design Decisions

### Why LangGraph?

**Problem**: Traditional function chains are rigid and hard to extend.

**Solution**: LangGraph's state machine provides:
- Conditional branching: Skip drug safety if no medications
- Parallel execution: Run NLP + risk prediction concurrently
- Human-in-the-loop: Pause for clinician review before recommendations
- Observability: Every agent's input/output logged in state

Example:

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(PipelineState)

# Add nodes
graph.add_node("intake", intake_agent)
graph.add_node("nlp", nlp_agent)
graph.add_node("risk", risk_agent)

# Add edges
graph.add_edge("intake", "nlp")
graph.add_conditional_edges(
    "nlp",
    lambda state: "risk" if state["conditions"] else "skip_risk"
)
graph.add_edge("risk", END)

app = graph.compile()
```

### Why TF-IDF Instead of Embeddings?

Trade-offs:

**TF-IDF**:
- ✅ Zero dependencies (no embedding model download)
- ✅ Fast for small corpus (10 guidelines)
- ✅ Interpretable (term frequency visible)
- ❌ Doesn't capture semantic similarity

**Embeddings** (e.g., sentence-transformers):
- ✅ Captures semantic meaning ("heart attack" ≈ "myocardial infarction")
- ❌ Requires 100MB+ embedding model
- ❌ Slower inference on CPU

**Decision**: TF-IDF keeps project fully offline and dependency-light. Swapping to embeddings requires changing only `src/utils/vector_store.py`.

### Why Fine-Tune Instead of Prompt Engineering?

Prompt Engineering:

```python
prompt = f"""Extract symptoms from this note:
Note: {patient_note}
Symptoms: """
```

- ✅ Zero training data needed
- ❌ Inconsistent output format
- ❌ Struggles with medical jargon
- ❌ Requires large model (7B+)

Fine-Tuning:

```json
// Training example
{
  "instruction": "Extract symptoms from clinical note",
  "input": "Patient reports chest pain and shortness of breath",
  "output": "{\"symptoms\": [\"chest pain\", \"shortness of breath\"]}"
}
```

- ✅ Consistent JSON output
- ✅ Learns medical terminology
- ✅ Works with small model (0.5B-3B)
- ❌ Requires ~1,000 training examples

**Decision**: Fine-tuning demonstrates full ML pipeline (data sourcing → labeling → training → inference), which is more impressive for portfolio than just prompt engineering.

---

## Future Improvements

### Short-Term (1-2 weeks)

Add Conditional Edges

```python
# Skip drug safety if no medications
def should_check_drugs(state):
    return "drug_safety" if state["medications"] else "skip_drugs"

graph.add_conditional_edges("nlp", should_check_drugs)
```

Add Evaluation Metrics
- Hand-annotate 50 examples
- Measure NLP extraction accuracy (precision/recall/F1)
- Compare 0.5B vs 1.5B vs 3B model performance

Persist Run History

```python
# SQLite database instead of flat files
import sqlite3
conn = sqlite3.connect("pipeline_runs.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE runs (
        id INTEGER PRIMARY KEY,
        patient_id TEXT,
        timestamp DATETIME,
        report TEXT
    )
""")
```

### Medium-Term (1-2 months)

Quantize Model for Deployment

```bash
# Convert to GGUF format (4-bit quantization)
python convert.py --outtype q4_0 models/Qwen2.5-3B-Instruct

# Reduces size from 6GB → 2GB, fits free-tier RAM
```

Add Multi-Modal Inputs
- Accept ECG images (CNN classifier)
- Process lab result PDFs (OCR + NLP)
- Voice-to-text intake (Whisper ASR)

Parallel Agent Execution

```python
# Run NLP + Risk + Drug Safety concurrently
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as executor:
    nlp_future = executor.submit(nlp_agent, state)
    risk_future = executor.submit(risk_agent, state)
    drug_future = executor.submit(drug_safety_agent, state)

    nlp_result = nlp_future.result()
    risk_result = risk_future.result()
    drug_result = drug_future.result()
```

### Long-Term (3-6 months)

Replace Weak Supervision with Clinical Annotation
- Hire medical students to label 500 examples
- Measure inter-annotator agreement (Cohen's kappa)
- Retrain model with high-quality labels

Add Federated Learning
- Train across multiple hospitals without sharing patient data
- Privacy-preserving model updates

Build Clinician UI
- Editable recommendations (clinician can modify AI output)
- Confidence scores for each extraction
- Audit trail (who changed what, when)

---

## 📚 Dataset Provenance & Ethics

### MTSamples Clinical Corpus

- Source: https://www.mtsamples.com (public medical transcription samples)
- License: Free for educational/research use
- Content:
  - ~5,000 real clinical notes
  - Authored by medical transcriptionists
  - Covers: cardiology, pulmonology, gastroenterology, endocrinology, etc.
  - Format: SOAP notes (Subjective, Objective, Assessment, Plan)
- Privacy:
  - ✅ No PHI: All patient identifiers removed by MTSamples
  - ✅ Not synthetic: Real clinical documentation
  - ✅ Educational use: Approved for NLP research
- Weak Supervision Labels:
  - Not clinician-annotated: Labels generated via rule-based heuristics
  - Accuracy: ~70-80% (estimated, not formally validated)
  - Use case: Portfolio demo, not clinical deployment

### UCI Heart Disease Dataset

- Source: https://archive.ics.uci.edu/ml/datasets/heart+disease
- License: Public domain (UCI Machine Learning Repository)
- Content:
  - 303 patients from Cleveland Clinic
  - 13 clinical features + binary target
  - Collected 1988 (historical dataset)
- Limitations:
  - Small sample size (303 patients)
  - Single center (Cleveland Clinic only)
  - Outdated (1988 data, treatment guidelines changed)
  - Demographics: 78% male, mostly white
- Use Case: Educational demo of ML pipeline, not clinical prediction

---

## Contributing

Contributions welcome! Here's how to help:

### Reporting Bugs

```bash
# Include this info in your issue
python --version
pip list | grep -E "langgraph|transformers|torch"
uname -a  # macOS version
```

### Adding Features

1. Fork the repo
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Add tests: `pytest tests/test_new_feature.py`
4. Update README if needed
5. Submit PR

### Code Style

```python
# We use type hints everywhere
def process_patient(patient_id: str, mock: bool = False) -> dict[str, Any]:
    """Process patient and return report."""
    ...

# We use docstrings (Google style)
def extract_symptoms(note: str) -> list[str]:
    """Extract symptoms from clinical note.

    Args:
        note: Free-text clinical note

    Returns:
        List of extracted symptom strings

    Example:
        >>> extract_symptoms("Patient has chest pain")
        ['chest pain']
    """
    ...
```

---

## 📄 License

MIT License — see LICENSE file for details.

You are free to:
- ✅ Use this code in commercial projects
- ✅ Modify and distribute it
- ✅ Use it in your portfolio

You must:
- Include original copyright notice
- Provide license text with distributions

---

## 👨‍💻 Author

**Al-Hossein Mahmoud** (alhosseinjr)
- GitHub: https://github.com/alhosseinjr
- LinkedIn: https://linkedin.com/in/alhosseinjr
- Portfolio: https://alhossein.site

Built with ❤️ on Apple Silicon

---

## 🙏 Acknowledgments

- LangChain/LangGraph team for excellent orchestration framework
- Qwen team (Alibaba) for open-source models
- MTSamples for public clinical corpus
- UCI ML Repository for Heart Disease dataset
- Hugging Face for transformers library

---

## 📞 Support

- Questions? Open an issue on GitHub.
- Urgent? Email: al7ossein@gmail.com
- Want to collaborate? Let's connect on LinkedIn!

<div align="center">

⭐ Star this repo if you found it useful!

Built with Python • LangGraph • FastAPI • Qwen • Apple Silicon

Report Bug · Request Feature · View Demo

</div>