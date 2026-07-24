#!/usr/bin/env python3
"""
Builds the local LLM's fine-tuning data from a REAL corpus: MTSamples, a
public collection of ~5,000 de-identified medical transcription samples
across specialties (SOAP-style notes: Subjective, Medications, Assessment,
Plan, etc.). This is real clinical-style text written by transcriptionists
for training/reference purposes -- not text authored by this script.

What this script does NOT do: it does not invent patients, notes, or
free-text output. Every "assistant" field below is either:
  (a) copied/trimmed directly from that row's own Assessment/Plan/Impression
      section, or
  (b) a label computed with a fixed, documented rule (regex / keyword
      lookup) applied to that row's own text or metadata (specialty,
      keywords).

This is weak/rule-based supervision over real data, not hand-written
synthetic examples. It is good enough to teach a small model the *format*
and *task* (structured JSON extraction / verification / synthesis), but it
is NOT clinically validated ground truth -- see README limitations section.

Usage:
    python finetune/scripts/prepare_dataset.py \
        --input finetune/data/mtsamples_raw.csv \
        --out-dir finetune/data \
        --max-per-task 400
"""

import argparse
import json
import os
import random
import re

random.seed(13)

SECTION_RE = re.compile(r"([A-Z][A-Z0-9 /&\-]{2,40}):,\s*")

# --- curated lookup vocabularies (compiled by hand, not inferred from the
#     data) -- used to turn free text into structured labels -------------

DRUG_VOCAB = [
    "metformin", "warfarin", "aspirin", "ibuprofen", "albuterol", "metoprolol",
    "lisinopril", "atorvastatin", "amlodipine", "hydrochlorothiazide", "insulin",
    "losartan", "omeprazole", "levothyroxine", "gabapentin", "prednisone",
    "furosemide", "clopidogrel", "simvastatin", "amoxicillin", "azithromycin",
    "hydrocodone", "tramadol", "sertraline", "citalopram", "montelukast",
    "fluticasone", "digoxin", "carvedilol", "enoxaparin", "apixaban",
]

CONDITION_VOCAB = {
    "type_2_diabetes": ["type 2 diabetes", "diabetes mellitus", "diabetic"],
    "hypertension": ["hypertension", "high blood pressure"],
    "asthma": ["asthma"],
    "atrial_fibrillation": ["atrial fibrillation", "afib", "a-fib"],
    "copd": ["copd", "chronic obstructive pulmonary"],
    "coronary_artery_disease": ["coronary artery disease", "cad "],
    "hyperlipidemia": ["hyperlipidemia", "high cholesterol"],
    "gerd": ["gerd", "reflux disease"],
}

SYMPTOM_VOCAB = [
    "shortness of breath", "chest pain", "chest tightness", "headache",
    "fatigue", "dizziness", "cough", "fever", "swelling", "palpitations",
    "weight loss", "weight gain", "nausea", "vomiting", "abdominal pain",
    "back pain", "joint pain", "blurred vision", "increased thirst",
    "frequent urination", "wheezing", "nosebleeds",
]

GUIDELINE_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "guidelines")

# Populated by main() before build_reasoning_example() is called.
GUIDELINES_CACHE = {}
GUIDELINE_KEYS_CACHE = set()


# --- shared parsing helpers ---------------------------------------------

def parse_sections(transcription: str) -> dict:
    """Splits an MTSamples transcription into {SECTION_NAME: text} using
    the corpus's own 'HEADER:,' convention."""
    if not isinstance(transcription, str):
        return {}
    parts = SECTION_RE.split(transcription)
    sections = {}
    # split() with a capturing group returns [pre, header, body, header, body, ...]
    for i in range(1, len(parts) - 1, 2):
        header = parts[i].strip().upper()
        body = parts[i + 1].strip()
        sections.setdefault(header, body)
    return sections


def first_matching(sections: dict, candidates: list) -> str:
    for c in candidates:
        if c in sections:
            return sections[c]
    return ""


def find_drugs(text: str) -> list:
    text_l = text.lower()
    return sorted({d for d in DRUG_VOCAB if re.search(rf"\b{re.escape(d)}\b", text_l)})


def find_conditions(text: str) -> list:
    text_l = text.lower()
    found = []
    for canon, synonyms in CONDITION_VOCAB.items():
        for syn in synonyms:
            if syn in text_l and not _is_negated(text_l, syn):
                found.append(canon.replace("_", " "))
                break
    return found


NEGATION_WORDS = ["no ", "denies", "without", "negative for", "not experiencing", "absence of"]


def _is_negated(text_l: str, phrase: str, window: int = 25) -> bool:
    """Cheap negation check: looks for a negation cue in the ~25 chars
    immediately before the matched phrase (e.g. 'No vomiting.')."""
    idx = text_l.find(phrase)
    if idx == -1:
        return False
    start = max(0, idx - window)
    preceding = text_l[start:idx]
    return any(neg in preceding for neg in NEGATION_WORDS)


def find_symptoms(text: str) -> list:
    text_l = text.lower()
    return [s for s in SYMPTOM_VOCAB if s in text_l and not _is_negated(text_l, s)]


def find_age_sex(text: str):
    age_m = re.search(r"(\d{1,3})[\s-]*year[\s-]*old", text, re.IGNORECASE)
    age = int(age_m.group(1)) if age_m else None
    sex = "female" if re.search(r"\bfemale\b", text, re.IGNORECASE) else (
        "male" if re.search(r"\bmale\b", text, re.IGNORECASE) else None)
    return age, sex


def is_smoker(text: str):
    text_l = text.lower()
    if "non-smoker" in text_l or "does not smoke" in text_l or "denies tobacco" in text_l:
        return False
    if "smoker" in text_l or "pack-year" in text_l or "tobacco use" in text_l:
        return True
    return False


def clip(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


NUMBERED_ITEM_RE = re.compile(r"^\s*\d+\.\s*")


def split_plan_items(plan_text: str, max_items: int = 3) -> list:
    # Plan sections are usually comma-joined pseudo-sentences; split on
    # '.,' boundaries (period immediately followed by the corpus's comma
    # separator) and fall back to '. ' otherwise. Also strip MTSamples'
    # own numbering ("1. ", "2. ") since each chunk becomes its own list
    # item -- keeping the numbers would double them up with the report's
    # own bullet formatting.
    chunks = re.split(r"\.,\s*|\.\s+", plan_text)
    items = []
    for c in chunks:
        c = NUMBERED_ITEM_RE.sub("", c.strip())
        if len(c) > 8:
            items.append(clip(c, 140))
    return items[:max_items] if items else []


def clean_assessment_text(assessment_text: str) -> str:
    """MTSamples' own ',' convention turns 'HEADER:,' into a pseudo-
    newline and also joins numbered sub-items with '.,' (e.g.
    '1. Stable ejection fraction.,2. Normal EF.,PLAN: ,Continue...').
    Training directly on that taught the fine-tuned model to reproduce
    the same broken punctuation. This normalizes it into real prose:
    strip numbering, turn '.,' into '. ', and cut off anything from a
    leaked trailing header (e.g. an un-matched 'PLAN:') since that
    content belongs in recommendations, not the assessment sentence.
    """
    text = assessment_text
    # Drop a leaked header fragment and everything after it (handled
    # separately as its own PLAN/RECOMMENDATION section upstream).
    text = re.split(r"[.,]?\s*(?:PLAN|RECOMMENDATIONS?)\s*:\s*,?", text, maxsplit=1)[0]
    # '.,' -> '. ' (real sentence boundary instead of the corpus's comma)
    text = re.sub(r"\.,\s*", ". ", text)
    # Strip embedded numbering ("1. ", "2. ") now that items read as
    # consecutive sentences rather than a list.
    text = re.sub(r"(?<=[.\s])\d+\.\s+", "", text)
    text = re.sub(r"^\d+\.\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


# --- system prompts (must match src/agents/*.py exactly) -----------------

NLP_SYSTEM_PROMPT = """You are a clinical NLP extraction agent. Given a free-text
patient intake note, extract structured entities. Respond ONLY with valid
JSON (no markdown fences, no commentary) in exactly this shape:

{
  "symptoms": ["..."],
  "mentioned_conditions": ["..."],
  "mentioned_medications": ["..."],
  "notable_flags": ["..."]
}

"notable_flags" should capture anything clinically noteworthy in the phrasing
(e.g. "new OTC medication mentioned", "chest symptoms reported",
"family history mentioned"). Keep lists short and only include what is
actually supported by the text."""

GUIDELINE_SYSTEM_PROMPT = """You are a guideline verification agent in a clinical
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

REASONING_SYSTEM_PROMPT = """You are the clinical reasoning agent in a multi-agent
decision-support pipeline (for a software demo, not real patient care).
You receive structured output from upstream agents: patient profile,
extracted NLP entities, an ML risk prediction, retrieved evidence
snippets, drug safety flags, and a guideline verification note.

Synthesize these into a single assessment. Respond ONLY with valid JSON in
exactly this shape:

{
  "assessment": "2-4 sentence synthesis of the patient's situation",
  "recommendations": ["short actionable next step", "..."],
  "priority": "routine" | "elevated" | "urgent"
}

Ground every statement in the provided data -- do not invent findings."""


# --- per-task builders ---------------------------------------------------

def build_nlp_example(row) -> dict:
    sections = parse_sections(row["transcription"])
    note = first_matching(sections, ["HISTORY OF PRESENT ILLNESS", "SUBJECTIVE", "HISTORY"])
    if not note:
        note = clip(str(row["transcription"]), 900)
    note = clip(note, 900)

    meds_section = first_matching(sections, ["MEDICATIONS", "CURRENT MEDICATIONS"])
    meds = find_drugs(meds_section or note)
    conditions = find_conditions(note + " " + str(row.get("keywords", "")))
    symptoms = find_symptoms(note)

    flags = []
    if "family history" in note.lower():
        flags.append("family history mentioned")
    if any(p in note.lower() for p in ["chest pain", "chest tightness", "shortness of breath"]):
        flags.append("chest/respiratory symptoms reported")
    if is_smoker(note):
        flags.append("smoking history noted")

    if not (meds or conditions or symptoms or flags):
        return None  # not enough signal, skip

    target = {
        "symptoms": symptoms,
        "mentioned_conditions": conditions,
        "mentioned_medications": meds,
        "notable_flags": flags,
    }
    return {
        "task": "nlp_extraction",
        "system": NLP_SYSTEM_PROMPT,
        "user": f"Intake note:\n\n{note}",
        "assistant": json.dumps(target),
    }


def load_guidelines(guideline_dir: str) -> dict:
    out = {}
    for fname in os.listdir(guideline_dir):
        if fname.endswith(".txt"):
            key = fname.replace(".txt", "")
            with open(os.path.join(guideline_dir, fname), "r", encoding="utf-8") as f:
                out[key] = {"source": fname, "snippet": clip(f.read(), 500)}
    return out


def build_guideline_examples(row, guidelines: dict, all_guideline_keys: list) -> list:
    """Mirrors src/agents/guideline_agent.py's exact prompt format:
    a structured '- Age/Comorbidities/Smoker/Predicted risk category'
    profile block, plus ALL retrieved snippets joined together (the real
    agent always sees up to top_k=3 snippets at once, not one at a time --
    training on single-snippet examples taught the model the wrong task
    shape, so we replicate multi-snippet retrieval here too)."""
    sections = parse_sections(row["transcription"])
    note = first_matching(sections, ["HISTORY OF PRESENT ILLNESS", "SUBJECTIVE", "ASSESSMENT", "IMPRESSION"])
    if not note:
        return []
    note = clip(note, 500)
    age, sex = find_age_sex(str(row.get("description", "")) + " " + note)
    conditions_found = find_conditions(note + " " + str(row.get("keywords", "")))
    matched_keys = [k for k in guidelines if k.replace("_", " ") in conditions_found]
    if not matched_keys:
        return []

    smoker = is_smoker(note)
    risk_category = random.choice(["low", "moderate", "high"])  # verifier shouldn't depend on this field's exact value
    profile_block = (
        f"- Age: {age or 'unknown'}, Sex: {sex or 'unknown'}\n"
        f"- Comorbidities: {conditions_found}\n"
        f"- Smoker: {smoker}\n"
        f"- Predicted risk category: {risk_category} (score 0.5)"
    )

    # Build a realistic top_k=3 retrieval mix: the true matching guideline(s)
    # plus enough mismatched ones to fill out 3, mirroring what TF-IDF
    # retrieval actually returns (it doesn't know relevance in advance).
    other_keys = [k for k in all_guideline_keys if k not in matched_keys]
    random.shuffle(other_keys)
    n_distractors = max(0, min(len(other_keys), 3 - len(matched_keys)))
    snippet_keys = matched_keys[:3] + other_keys[:n_distractors]
    random.shuffle(snippet_keys)

    evidence_block = "\n\n".join(
        f"Source: {guidelines[k]['source']}\nSnippet: {guidelines[k]['snippet']}" for k in snippet_keys
    )
    relevant_sources = [guidelines[k]["source"] for k in snippet_keys if k in matched_keys]

    example = {
        "task": "guideline_verification",
        "system": GUIDELINE_SYSTEM_PROMPT,
        "user": f"Patient summary:\n{profile_block}\n\nRetrieved evidence:\n{evidence_block}",
        "assistant": json.dumps({
            "aligned": True,
            "notes": f"The retrieved evidence for {', '.join(k.replace('_', ' ') for k in matched_keys)} is consistent with the patient's documented condition(s); other retrieved snippets are not relevant.",
            "citations": relevant_sources,
        }),
    }

    # One negative example: same patient, but ONLY mismatched snippets retrieved
    # (simulates a weak/irrelevant retrieval -- aligned should be false).
    neg_examples = []
    if len(other_keys) >= 1:
        neg_keys = other_keys[:min(3, len(other_keys))]
        neg_evidence_block = "\n\n".join(
            f"Source: {guidelines[k]['source']}\nSnippet: {guidelines[k]['snippet']}" for k in neg_keys
        )
        neg_examples.append({
            "task": "guideline_verification",
            "system": GUIDELINE_SYSTEM_PROMPT,
            "user": f"Patient summary:\n{profile_block}\n\nRetrieved evidence:\n{neg_evidence_block}",
            "assistant": json.dumps({
                "aligned": False,
                "notes": "None of the retrieved guideline snippets match this patient's documented condition(s).",
                "citations": [],
            }),
        })

    return [example] + neg_examples


LIST_ITEM_RE = re.compile(r"\d+\.\s")


def is_list_heavy(text: str, max_items: int = 3) -> bool:
    """MTSamples ASSESSMENT sections are often raw numbered problem lists
    (e.g. '1. Hypertension. 2. Obesity. ... 14. Hepatitis.'). The reasoning
    agent's system prompt asks for a short prose synthesis, not a long
    list -- training on list-heavy targets teaches the model to always
    dump a generic numbered list regardless of the actual input. Skip
    those as training targets."""
    return len(LIST_ITEM_RE.findall(text)) > max_items


def build_reasoning_example(row) -> dict:
    sections = parse_sections(row["transcription"])
    assessment_src = first_matching(sections, ["ASSESSMENT", "IMPRESSION"])
    plan_src = first_matching(sections, ["PLAN", "RECOMMENDATION", "RECOMMENDATIONS"])
    if not assessment_src or is_list_heavy(assessment_src):
        return None

    note_all = str(row["transcription"])
    age, sex = find_age_sex(str(row.get("description", "")) + " " + note_all[:400])
    conditions = find_conditions(note_all + " " + str(row.get("keywords", "")))
    symptoms = find_symptoms(note_all[:1200])
    meds = find_drugs(note_all)
    smoker = is_smoker(note_all)

    flag_count = 0
    known_pairs = [("warfarin", "ibuprofen"), ("warfarin", "aspirin"),
                   ("albuterol", "metoprolol"), ("aspirin", "ibuprofen")]
    for a, b in known_pairs:
        if a in meds and b in meds:
            flag_count += 1

    specialty = str(row.get("medical_specialty", "")).strip()
    urgent_specialties = {"Emergency Room Reports"}
    risk_category = "high" if (flag_count > 0 or specialty in urgent_specialties) else (
        "moderate" if conditions else "low")

    # Mirror retrieved_evidence's real shape (source/snippet/score) using
    # this row's own matched conditions as a stand-in for what TF-IDF
    # retrieval would have surfaced.
    matched_condition_keys = [c.replace(" ", "_") for c in conditions if c.replace(" ", "_") in GUIDELINE_KEYS_CACHE]
    retrieved_evidence = [
        {"source": GUIDELINES_CACHE[k]["source"], "snippet": GUIDELINES_CACHE[k]["snippet"], "score": 0.5}
        for k in matched_condition_keys[:2]
    ]

    payload = {
        "patient_profile": {
            "age": age,
            "sex": sex,
            "comorbidities": conditions,
            "smoker": smoker,
        },
        "extracted_entities": {
            "symptoms": symptoms,
            "mentioned_conditions": conditions,
            "mentioned_medications": meds,
            "notable_flags": [],
        },
        "risk_result": {"risk_category": risk_category},
        "retrieved_evidence": retrieved_evidence,
        "drug_safety_result": {"flag_count": flag_count},
        "guideline_verification": {"aligned": bool(retrieved_evidence)},
    }

    assessment_text = clip(clean_assessment_text(assessment_src), 220)
    if not assessment_text:
        return None  # cleaning left nothing usable -- skip rather than train on an empty target

    recommendations = split_plan_items(plan_src) if plan_src else []
    # Drop near-duplicate items (MTSamples plans sometimes restate the same
    # instruction two ways, e.g. "Follow up EKG results" / "We will check
    # the EKG") -- dedupe on a normalized form so the model doesn't learn
    # to pad recommendations with restatements.
    seen, deduped = set(), []
    for item in recommendations:
        key = re.sub(r"[^a-z0-9 ]", "", item.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    recommendations = deduped
    if not recommendations:
        recommendations = ["Continue current management and reassess at next visit."]

    if specialty in urgent_specialties or flag_count > 0:
        priority = "urgent" if specialty in urgent_specialties else "elevated"
    elif risk_category == "high":
        priority = "elevated"
    else:
        priority = "routine"

    target = {
        "assessment": assessment_text,
        "recommendations": recommendations,
        "priority": priority,
    }
    return {
        "task": "clinical_reasoning",
        "system": REASONING_SYSTEM_PROMPT,
        "user": json.dumps(payload, indent=2),
        "assistant": json.dumps(target),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(os.path.dirname(__file__), "..", "data", "mtsamples_raw.csv"))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--guidelines-dir", default=GUIDELINE_DIR_DEFAULT)
    parser.add_argument("--max-per-task", type=int, default=400)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.input)
    df = df.dropna(subset=["transcription"])
    guidelines = load_guidelines(args.guidelines_dir)
    all_guideline_keys = list(guidelines.keys())

    global GUIDELINES_CACHE, GUIDELINE_KEYS_CACHE
    GUIDELINES_CACHE = guidelines
    GUIDELINE_KEYS_CACHE = set(all_guideline_keys)

    nlp_examples, guideline_examples, reasoning_examples = [], [], []

    for _, row in df.iterrows():
        if len(nlp_examples) < args.max_per_task:
            ex = build_nlp_example(row)
            if ex:
                nlp_examples.append(ex)
        if len(guideline_examples) < args.max_per_task:
            guideline_examples.extend(build_guideline_examples(row, guidelines, all_guideline_keys))
        if len(reasoning_examples) < args.max_per_task:
            ex = build_reasoning_example(row)
            if ex:
                reasoning_examples.append(ex)
        if (len(nlp_examples) >= args.max_per_task and
                len(guideline_examples) >= args.max_per_task and
                len(reasoning_examples) >= args.max_per_task):
            break

    guideline_examples = guideline_examples[:args.max_per_task]

    all_examples = nlp_examples + guideline_examples + reasoning_examples
    random.shuffle(all_examples)

    n_val = max(1, int(len(all_examples) * args.val_fraction))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path = os.path.join(args.out_dir, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"nlp_extraction examples:        {len(nlp_examples)}")
    print(f"guideline_verification examples: {len(guideline_examples)}")
    print(f"clinical_reasoning examples:     {len(reasoning_examples)}")
    print(f"Total: {len(all_examples)}  (train={len(train_examples)}, val={len(val_examples)})")
    print(f"Wrote {train_path}")
    print(f"Wrote {val_path}")


if __name__ == "__main__":
    main()
