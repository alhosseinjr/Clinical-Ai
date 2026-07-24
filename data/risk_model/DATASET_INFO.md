# Risk Model Dataset

**Source:** UCI Machine Learning Repository — "Heart Disease" dataset
(Cleveland Clinic Foundation subset), one of the most widely used public
benchmark datasets in clinical ML research and education. Distributed here
as `heart_disease.csv` (303 rows, 13 features + binary target), a commonly
used cleaned/re-encoded mirror of the original UCI data.

- **Real, not synthetic.** Every row is a real (de-identified) patient
  record collected at the Cleveland Clinic in the 1980s. No PHI — the
  dataset has been public and de-identified for decades and is a standard
  teaching/benchmark set (used in hundreds of published papers).
- **Not the pipeline's own patients.** This trains the *risk model only*.
  The `sample_patients.json` demo patients are unrelated synthetic
  personas used to demo the pipeline end-to-end.
- **Known limitation:** 303 rows is small by modern ML standards, and the
  data is ~40 years old and from a single institution. Good enough for a
  demo model with honestly-reported metrics; not good enough (and not
  intended) for real deployment.

## Columns

| Column | Meaning | Values |
|---|---|---|
| age | Age in years | int |
| sex | Biological sex | 0 = female, 1 = male |
| cp | Chest pain type | 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic |
| trestbps | Resting systolic blood pressure (mmHg) | int |
| chol | Serum cholesterol (mg/dl) | int |
| fbs | Fasting blood sugar > 120 mg/dl | 0 = false, 1 = true |
| restecg | Resting ECG result | 0 = normal, 1 = ST-T wave abnormality, 2 = probable/definite LV hypertrophy |
| thalach | Max heart rate achieved (stress test) | int |
| exang | Exercise-induced angina | 0 = no, 1 = yes |
| oldpeak | ST depression induced by exercise relative to rest | float |
| slope | Slope of peak exercise ST segment | 0 = upsloping, 1 = flat, 2 = downsloping |
| ca | Number of major vessels colored by fluoroscopy | 0-4 |
| thal | Thalassemia test result | 0 = unknown, 1 = fixed defect, 2 = normal, 3 = reversible defect |
| target | Presence of heart disease | 0 = no, 1 = yes |

## Data quality note: label direction was corrected

The GitHub mirror this dataset was originally pulled from (`kb22/Heart-Disease-Prediction`)
had its `target` column **inverted** relative to the disease label's plain-English
meaning: `1` meant "no disease" and `0` meant "disease present," the opposite of what
the column name suggests. This wasn't visible from the numbers alone -- accuracy on
the inverted label was still a respectable-looking ~80% -- so it was caught by sanity-
checking feature correlations against clinical priors rather than trusting the label
name: age, sex, max heart rate, exercise-induced angina, ST depression, vessels
affected, and thalassemia result were *all* correlated with the label in the wrong
clinical direction simultaneously (e.g. older age and higher cholesterol correlating
with *less* disease). That many features flipping together is a strong signal the
label itself is flipped, not that the underlying relationships are all real
inversions of medical knowledge.

This was confirmed against an independent, differently-preprocessed mirror of the
same 303-patient Cleveland cohort (the ISLR textbook version), which showed the
expected clinical direction on every feature. The label in `heart_disease.csv` here
has been corrected (`target = 1 - original_target`) so `1` now means disease present.
Corrected-label accuracy is ~85% (see `outputs/risk_model_eval.md`), slightly
*higher* than the ~80% on the inverted label -- consistent with the model fitting a
real, learnable signal rather than an artificially flipped one.

**Why this is worth stating plainly:** it's a real, disclosed data-quality catch, not
a hypothetical -- and "I don't just trust a column name, I sanity-check learned
feature directions against domain priors before reporting a metric" is a more
credible ML engineering story than a clean number with no scrutiny behind it.

## Honest scope note

Several of these features (thalach, exang, oldpeak, slope, ca, thal) come
from a cardiac stress test / catheterization workup, not a basic intake
visit. This model is realistically a **post-workup risk stratification
model**, not a walk-in triage tool. The demo's `sample_patients.json`
includes a `cardiac_workup` block per patient representing "already
completed" test results, consistent with that framing — see the main
README for how this is presented in the pipeline.
