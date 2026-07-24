#!/usr/bin/env python3
"""
LoRA fine-tunes Qwen2.5-0.5B-Instruct on the 3 pipeline tasks
(nlp_extraction, guideline_verification, clinical_reasoning), replacing
the Anthropic API calls in src/utils/llm.py with a local model.

Targets Apple Silicon (MPS backend) by default -- tested for an M-series
Mac. Falls back to CPU automatically if MPS isn't available; will also use
CUDA if present.

Usage:
    python finetune/scripts/train_lora.py \
        --train finetune/data/train.jsonl \
        --val finetune/data/val.jsonl \
        --out-dir models/clinical-lora-adapter \
        --epochs 3

Runtime on an M4 (16GB+ unified memory): roughly 20-40 minutes for
~1,000 examples / 3 epochs with the default settings below. Reduce
--max-steps or --epochs if you want a faster first pass.

If training is dramatically slower than that on your machine, MPS bf16
kernel support is a common culprit -- try `--precision fp32` or
`--precision fp16` to compare, and `--max-steps 10` for a quick smoke
test before committing to a full run. See README for more.
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE_MODEL_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LEN = 1024


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_precision(device: str, requested: str) -> str:
    """Resolves the --precision flag to an actual dtype choice.

    `requested="auto"` (the default) preserves the original behavior:
    bf16 on mps/cuda, fp32 on cpu. Passing --precision explicitly lets you
    A/B test against that default -- e.g. bf16 kernel support on MPS is
    known to be immature/slow in some torch versions, so `fp32` or `fp16`
    can end up faster there despite using more memory/bandwidth per op.
    """
    if requested != "auto":
        if requested in ("bf16", "fp16") and device == "cpu":
            print(f"Warning: --precision {requested} requested on CPU; most CPU kernels don't "
                  f"have a fast low-precision path, this may not help (or may even be slower).")
        return requested
    return "bf16" if device in ("mps", "cuda") else "fp32"


def load_jsonl(path: str) -> list:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_chat_text(tokenizer, example: dict) -> tuple:
    """Returns (full_text, prompt_text). prompt_text is the part we mask
    out of the loss (system+user turns + generation prompt); full_text
    additionally includes the assistant's target JSON + eos."""
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + example["assistant"] + tokenizer.eos_token
    return full_text, prompt_text


def tokenize_example(tokenizer, example: dict) -> dict:
    full_text, prompt_text = build_chat_text(tokenizer, example)

    full_ids = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]

    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    for i in range(prompt_len):
        labels[i] = -100  # mask prompt tokens out of the loss

    return {"input_ids": full_ids, "labels": labels}


def make_data_collator(tokenizer):
    """DataCollatorForLanguageModeling doesn't know how to pad a custom
    `labels` field (it only pads input_ids/attention_mask and expects to
    derive labels itself). Since we build our own -100-masked labels to
    only train on the assistant's completion, we need our own padding
    logic here instead."""
    pad_id = tokenizer.pad_token_id

    def collate(features):
        import torch
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = f["input_ids"]
            lab = f["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(lab + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    parser.add_argument("--train", default=os.path.join(os.path.dirname(__file__), "..", "data", "train.jsonl"))
    parser.add_argument("--val", default=os.path.join(os.path.dirname(__file__), "..", "data", "val.jsonl"))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "models", "clinical-lora-adapter"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
        help="Compute dtype for the base model + training. 'auto' (default) matches the "
             "original behavior: bf16 on mps/cuda, fp32 on cpu. Override to A/B test -- "
             "MPS bf16 kernels are immature in some torch versions and can be *slower* "
             "than fp32 there; try 'fp32' or 'fp16' if training is unexpectedly slow.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Stop after this many optimizer steps regardless of --epochs. Use a small "
             "value (e.g. 10) for a quick smoke test that the data/model/save path all "
             "work end-to-end before committing to a full multi-hour run.",
    )
    args = parser.parse_args()

    device = pick_device()
    precision = pick_precision(device, args.precision)
    print(f"Using device: {device}, precision: {precision}")
    if device == "cpu":
        print("Warning: no GPU/MPS backend detected -- training will be slow.")

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[precision]
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch_dtype)
    model.to(device)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_raw = load_jsonl(args.train)
    val_raw = load_jsonl(args.val)
    print(f"Train examples: {len(train_raw)} | Val examples: {len(val_raw)}")

    train_tokenized = [tokenize_example(tokenizer, ex) for ex in train_raw]
    val_tokenized = [tokenize_example(tokenizer, ex) for ex in val_raw]

    train_dataset = Dataset.from_list(train_tokenized)
    val_dataset = Dataset.from_list(val_tokenized)

    collator = make_data_collator(tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,  # -1 = disabled, use num_train_epochs
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch" if args.max_steps is None else "no",  # epoch-based eval doesn't
        # make sense when max_steps stops mid-epoch during a smoke test
        save_strategy="epoch" if args.max_steps is None else "no",
        save_total_limit=2,
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        report_to=[],
        use_mps_device=(device == "mps"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    trainer.train()

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"\nSaved LoRA adapter + tokenizer to: {args.out_dir}")
    print("Set LOCAL_MODEL_ID / LORA_ADAPTER_PATH in .env to point the pipeline at it (see README).")


if __name__ == "__main__":
    main()
