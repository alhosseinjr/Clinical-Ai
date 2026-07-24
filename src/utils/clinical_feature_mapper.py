"""
Maps NLP extracted clinical entities
into ML model features.
"""


def map_clinical_features(entities, patient_profile):

    features = {}

    # Demographics
    features["age"] = patient_profile.get("age", 0)

    features["sex"] = patient_profile.get(
        "sex",
        "unknown"
    )


    # Conditions

    conditions = [
        c.lower()
        for c in entities.get(
            "mentioned_conditions",
            []
        )
    ]

    features["diabetes"] = int(
        "type 2 diabetes" in conditions
    )

    features["hypertension"] = int(
        "hypertension" in conditions
    )

    features["coronary_artery_disease"] = int(
        "coronary artery disease" in conditions
    )


    # Symptoms

    symptoms = [
        s.lower()
        for s in entities.get(
            "symptoms",
            []
        )
    ]


    features["chest_symptom"] = int(
        "chest pain" in symptoms
        or
        "chest tightness" in symptoms
    )


    features["shortness_of_breath"] = int(
        "shortness of breath" in symptoms
    )


    # Lifestyle

    risk = [
        r.lower()
        for r in entities.get(
            "risk_factors",
            []
        )
    ]


    features["smoker"] = int(
        "active smoker" in risk
    )


    features["family_history"] = int(
        len(
            entities.get(
                "family_history",
                []
            )
        ) > 0
    )


    return features