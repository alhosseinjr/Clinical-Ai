#!/usr/bin/env python3
"""
Downloads the real MTSamples dataset (~5,000 public medical transcription
samples, used widely in academic/portfolio clinical-NLP work) to
finetune/data/mtsamples_raw.csv.

Source note: MTSamples (mtsamples.com) publishes these transcription
samples for transcriptionist training/reference. They are real authored
clinical documentation (not real patient records -- no PHI), which is why
they're commonly reused as a public NLP research/teaching corpus. This
script pulls a plain-CSV mirror of that public dataset from GitHub.

If this mirror ever goes down, get the same file from Kaggle instead:
https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
(download mtsamples.csv and place it at finetune/data/mtsamples_raw.csv).
"""

import os
import urllib.request

URL = "https://raw.githubusercontent.com/socd06/medical-nlp/master/data/mtsamples.csv"
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mtsamples_raw.csv")


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print(f"Downloading MTSamples dataset from:\n  {URL}")
    try:
        urllib.request.urlretrieve(URL, OUT_PATH)
    except Exception as e:
        print(f"Download failed ({e}).")
        print("Manual fallback: download mtsamples.csv from")
        print("https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions")
        print(f"and save it to: {OUT_PATH}")
        raise SystemExit(1)

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"Saved {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
