"""
Fine-tune Qwen2.5-3B-Instruct using LoRA on Apple Silicon (MPS).
Optimized for memory efficiency to prevent OOM on consumer Macs.
"""

import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct" 
OUTPUT_DIR = "models/clinical-lora-adapter-3b"
DATA_DIR = "finetune/data"

# Memory optimization for Mac M-series
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "mps" else torch.float32

def main():
    print(f"Loading tokenizer and model ({MODEL_ID}) on {DEVICE} with {DTYPE}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        trust_remote_code=True,
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    # LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load Dataset
    train_data_path = os.path.join(DATA_DIR, "train.jsonl")
    if not os.path.exists(train_data_path):
        train_data_path = os.path.join(DATA_DIR, "clinical_train.jsonl")
        
    print(f"Loading dataset from {train_data_path}...")
    dataset = load_dataset("json", data_files={"train": train_data_path})

    # --- FIX: Preprocess dataset to create the 'text' column ---
    def format_example(example):
        # If 'text' already exists, do nothing
        if "text" in example:
            return example
            
        # Combine instruction and input for the user message
        user_content = f"{example.get('instruction', '')}\n{example.get('input', '')}".strip()
        
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example.get("output", "")}
        ]
        
        # Use the tokenizer's chat template to format the text
        example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
        return example

    print("Formatting dataset with chat template...")
    dataset = dataset.map(format_example)
    # -------------------------------------------------------------

    # Training Arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        optim="adamw_torch",
        max_grad_norm=0.3,
        report_to="none",
        packing=False,
    )

    print("Starting training... (This will take 1-2 hours on M4)")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        # "text" is the default column name, so no need to specify it!
    )

    trainer.train()

    print(f"Training complete! Saving adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()