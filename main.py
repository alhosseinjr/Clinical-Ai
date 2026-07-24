#!/usr/bin/env python3
"""
CLI entrypoint for the Clinical AI Pipeline demo project.

Usage:
    python main.py                          # run all sample patients, real local-LLM calls
    python main.py --patient-id P001         # run a single sample patient
    python main.py --mock                    # run without a local model (canned LLM output)
    python main.py --file path/to/patient.json --patient-id P099
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from src.graph import build_graph

DEFAULT_PATIENTS_FILE = os.path.join(os.path.dirname(__file__), "data", "sample_patients.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def load_patients(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_for_patient(app, patient_raw: dict, mock: bool) -> str:
    initial_state = {
        "patient_raw": patient_raw,
        "mock_mode": mock,
        "trace": [],
        "errors": [],
    }
    final_state = app.invoke(initial_state)
    return final_state["final_report"]


def main():
    parser = argparse.ArgumentParser(description="Run the clinical multi-agent pipeline.")
    parser.add_argument("--file", default=DEFAULT_PATIENTS_FILE, help="Path to a JSON file of patient records.")
    parser.add_argument("--patient-id", default=None, help="Run only the patient with this ID (default: run all).")
    parser.add_argument("--mock", action="store_true", help="Run without the local fine-tuned model.")
    args = parser.parse_args()

    load_dotenv()

    merged_path = os.environ.get("MERGED_MODEL_PATH", "models/clinical-llm-merged")
    adapter_path = os.environ.get("LORA_ADAPTER_PATH", "models/clinical-lora-adapter")
    if not args.mock and not (os.path.isdir(merged_path) or os.path.isdir(adapter_path)):
        print(f"No local model found at '{merged_path}' or '{adapter_path}' -- running in --mock mode.\n"
              f"(Run finetune/scripts/train_lora.py to train the local adapter -- see README.)\n")
        args.mock = True

    patients = load_patients(args.file)
    if args.patient_id:
        patients = [p for p in patients if p.get("patient_id") == args.patient_id]
        if not patients:
            print(f"No patient found with ID '{args.patient_id}' in {args.file}")
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app = build_graph()

    for patient in patients:
        print(f"Running pipeline for {patient.get('patient_id')} - {patient.get('name')}...")
        report = run_for_patient(app, patient, mock=args.mock)

        out_path = os.path.join(OUTPUT_DIR, f"report_{patient.get('patient_id')}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  -> report written to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
