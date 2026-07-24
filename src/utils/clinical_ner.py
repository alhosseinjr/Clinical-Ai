"""
Deterministic Clinical NER module.

Extracts:
- symptoms
- medical conditions
- medications
- risk factors
- family history

Used before LLM reasoning to provide reliable structured clinical data.
"""

import re


# -----------------------------
# Medical vocabulary
# -----------------------------

SYMPTOMS = {
    "chest pain": "chest pain",
    "chest tightness": "chest tightness",
    "shortness of breath": "shortness of breath",
    "sob": "shortness of breath",
    "fatigue": "fatigue",
    "tiredness": "fatigue",
    "headache": "headache",
    "headaches": "headache",
    "dizziness": "dizziness",
    "nausea": "nausea",
    "vomiting": "vomiting",
    "cough": "cough",
    "fever": "fever",
    "palpitations": "palpitations",
}


CONDITIONS = {
    "type 2 diabetes": "type 2 diabetes",
    "diabetes": "type 2 diabetes",
    "t2dm": "type 2 diabetes",

    "hypertension": "hypertension",
    "high blood pressure": "hypertension",
    "htn": "hypertension",

    "coronary artery disease": "coronary artery disease",
    "cad": "coronary artery disease",

    "heart failure": "heart failure",
    "myocardial infarction": "myocardial infarction",
    "heart attack": "myocardial infarction",
    "mi": "myocardial infarction",

    "obesity": "obesity",
    "copd": "copd",
    "asthma": "asthma",

    "chronic kidney disease": "chronic kidney disease",
    "ckd": "chronic kidney disease",
}


MEDICATIONS = {
    "metformin": "metformin",
    "aspirin": "aspirin",
    "atorvastatin": "atorvastatin",
    "statin": "statin",
    "lisinopril": "lisinopril",
    "insulin": "insulin",
}


RISK_FACTORS = {
    "smoker": "active smoker",
    "smoking": "active smoker",
    "tobacco": "tobacco use",

    "family history": "family history mentioned",
    "father": "family history mentioned",
    "mother": "family history mentioned",

    "sedentary": "sedentary lifestyle",

    "obese": "obesity risk factor",
}


# -----------------------------
# Helpers
# -----------------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_terms(text: str, dictionary: dict):
    found = []

    for keyword, normalized in dictionary.items():
        if keyword in text:
            found.append(normalized)

    return sorted(list(set(found)))


def detect_negations(text: str):
    """
    Detect simple clinical negation.
    """

    negations = []

    patterns = [
        r"no (\w+)",
        r"denies (\w+)",
        r"without (\w+)",
        r"negative for (\w+)"
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)

        for item in matches:
            negations.append(item)

    return negations


def extract_family_history(text: str):
    results = []

    patterns = [
        "father.*heart",
        "mother.*heart",
        "father.*mi",
        "mother.*diabetes",
        "family history of"
    ]

    for p in patterns:
        if re.search(p, text):
            results.append("family cardiovascular history")

    return list(set(results))


# -----------------------------
# Main extraction function
# -----------------------------

def extract_clinical_terms(note: str):

    if not note:
        return {
            "symptoms": [],
            "mentioned_conditions": [],
            "mentioned_medications": [],
            "risk_factors": [],
            "family_history": [],
            "negations": []
        }


    text = normalize_text(note)


    result = {
        "symptoms": extract_terms(
            text,
            SYMPTOMS
        ),

        "mentioned_conditions": extract_terms(
            text,
            CONDITIONS
        ),

        "mentioned_medications": extract_terms(
            text,
            MEDICATIONS
        ),

        "risk_factors": extract_terms(
            text,
            RISK_FACTORS
        ),

        "family_history": extract_family_history(
            text
        ),

        "negations": detect_negations(
            text
        )
    }


    return result