"""
Merges the newly trained 3B LoRA adapter into the base model.
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "models/clinical-lora-adapter-3b"
OUTPUT_PATH = "models/clinical-3b-merged"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu", # Merge on CPU to save MPS memory
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging...")
model = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_PATH}...")
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
print("Merge complete!")