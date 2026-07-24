#!/usr/bin/env python3
"""
Optional: merges the trained LoRA adapter into the base model weights and
saves a single standalone model. This avoids loading base + adapter
separately at inference time (slightly faster load, one folder to ship).

Usage:
    python finetune/scripts/merge_lora.py \
        --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --adapter-dir models/clinical-lora-adapter \
        --out-dir models/clinical-llm-merged
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "models", "clinical-lora-adapter"))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "models", "clinical-llm-merged"))
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)

    print(f"Loading adapter from: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base, args.adapter_dir)

    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()

    os.makedirs(args.out_dir, exist_ok=True)
    merged.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved merged model to: {args.out_dir}")
    print("Set MERGED_MODEL_PATH in .env to use it directly (see README).")


if __name__ == "__main__":
    main()
