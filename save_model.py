from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model from cache... (This will be fast since it's already downloaded)")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

print("Saving to local project folder...")
model.save_pretrained("models/Qwen2.5-3B-Instruct")
tokenizer.save_pretrained("models/Qwen2.5-3B-Instruct")

print("Done! Model saved to models/Qwen2.5-3B-Instruct")