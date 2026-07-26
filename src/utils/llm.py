"""
Local LLM wrapper used by every LLM-backed agent.
"""

import hashlib
import os
import torch
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

_model = None
_tokenizer = None
_device = None
_dtype = None
_model_unavailable_reason = None
_response_cache: dict = {}


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _pick_dtype(device: str):
    if device == "cpu":
        return torch.float32
    if device == "mps":
        return torch.bfloat16
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _load_model():
    """Lazily loads the local model."""
    global _model, _tokenizer, _device, _dtype, _model_unavailable_reason

    if _model is not None or _model_unavailable_reason is not None:
        return _model

    merged_path = os.environ.get("MERGED_MODEL_PATH", "models/clinical-3b-merged")
    adapter_path = os.environ.get("LORA_ADAPTER_PATH", "models/clinical-lora-adapter")
    base_model_id = os.environ.get("BASE_MODEL_ID", "models/Qwen2.5-3B-Instruct")

    try:
        _device = _pick_device()
        _dtype = _pick_dtype(_device)

        print(f"Loading model on {_device} with dtype {_dtype}...")

        # 1. Try Merged Model
        if os.path.isdir(merged_path) and os.path.exists(os.path.join(merged_path, "config.json")):
            print(f"Loading merged model from {merged_path}...")
            _tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
            _model = AutoModelForCausalLM.from_pretrained(merged_path, torch_dtype=_dtype, device_map="auto")
            print("Merged model loaded.")

        # 2. Try Base + Adapter
        elif os.path.isdir(adapter_path):
            print(f"Loading LoRA adapter from {adapter_path}...")
            print(f"Loading base model: {base_model_id}...")
            _tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=_dtype, device_map="auto")
            from peft import PeftModel
            _model = PeftModel.from_pretrained(base, adapter_path)
            print("Base model + LoRA adapter loaded.")
            
        # 3. FALLBACK: Just load the Base Model
        else:
            print(f"No adapter found. Loading raw base model: {base_model_id}...")
            # CRITICAL: Do NOT use fix_mistral_regex=True here. It breaks Qwen tokenizers.
            _tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
            _model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=_dtype, device_map="auto")
            print("Base model loaded successfully!")

        _model.eval()
        print("✅ Model ready for inference!")
        return _model

    except Exception as e:
        _model_unavailable_reason = f"Local model failed to load: {str(e)}"
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def _cache_key(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    h = hashlib.sha256()
    h.update(system_prompt.encode("utf-8"))
    h.update(b"\x00")
    h.update(user_prompt.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(max_tokens).encode("utf-8"))
    return h.hexdigest()


def call_llm(
    system_prompt: str,
    user_prompt: str,
    mock: bool = False,
    mock_response: Optional[str] = None,
    max_tokens: int = 500,
) -> str:
    """Calls the local LLM with the given prompts."""
    if mock:
        return mock_response or "[MOCK MODE]"
    
    # Check cache
    key = _cache_key(system_prompt, user_prompt, max_tokens)
    if key in _response_cache:
        return _response_cache[key]

    model = _load_model()
    if model is None:
        return f"[MOCK MODE] {_model_unavailable_reason}"
    
    # Use Qwen's built-in chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    input_text = _tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = _tokenizer(input_text, return_tensors="pt").to(_device)
    
    # GREEDY DECODING - strictly no sampling to prevent garbage output
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,      # CRITICAL: Must be False for JSON
            repetition_penalty=1.1, # Helps prevent repeating the same token (like !!!)
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.pad_token_id if _tokenizer.pad_token_id is not None else _tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = _tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    _response_cache[key] = response
    return response