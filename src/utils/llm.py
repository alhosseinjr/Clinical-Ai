"""
Local LLM wrapper used by every LLM-backed agent.

Replaces the previous Anthropic API client with a locally fine-tuned model
(Qwen2.5-0.5B-Instruct + a LoRA adapter trained on this project's clinical
tasks -- see finetune/). No network calls, no API key.

Loads (in priority order):
  1. MERGED_MODEL_PATH, if set and it exists  -- a merged standalone model
     produced by finetune/scripts/merge_lora.py
  2. BASE_MODEL_ID + LORA_ADAPTER_PATH, if the adapter directory exists --
     base model with the LoRA adapter applied on top
  3. Otherwise: no model available -- every call falls back to mock mode
     automatically (with a clear message), so the pipeline is always
     runnable even before you've trained the adapter.

Model loading now also picks a device-appropriate dtype: fp32 on every
device works but wastes ~2x the memory bandwidth and compute on GPU/MPS,
which is where this actually matters for latency. CPU stays fp32 since
most CPU kernels don't have a fast low-precision path. Generation with an
identical (system_prompt, user_prompt, max_tokens) triple is memoized
in-process, since repeated demo/dev runs against the same patient
otherwise regenerate byte-identical output.
"""

import hashlib
import os
from typing import Optional

_model = None
_tokenizer = None
_device = None
_dtype = None
_model_unavailable_reason = None
_response_cache: dict = {}


def _pick_device() -> str:
    # import torch
    # if torch.backends.mps.is_available():
    #     return "mps"
    # if torch.cuda.is_available():
    #     return "cuda"
    return "mps"


def _pick_dtype(device: str):
    """fp32 on CPU (safe default, no real speed win from fp16 there for
    most kernels); bf16 on CUDA when the GPU supports it (numerically more
    stable than fp16, no loss-scaling needed); fp16 otherwise on CUDA/MPS,
    since bf16 support on MPS is inconsistent across torch/macOS versions."""
    import torch
    if device == "cpu":
        return torch.float32
    if device == "mps":
        return torch.bfloat16
    if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16


def _load_model():
    """Lazily loads the local model on first real (non-mock) call."""
    global _model, _tokenizer, _device, _dtype, _model_unavailable_reason

    if _model is not None or _model_unavailable_reason is not None:
        return

    merged_path = os.environ.get("MERGED_MODEL_PATH", "models/clinical-llm-merged")
    adapter_path = os.environ.get("LORA_ADAPTER_PATH", "models/clinical-lora-adapter")
    base_model_id = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

    try:
        print("Importing torch...")
        import torch
        print("Torch imported.")

        print("Importing transformers...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Transformers imported.")

        _device = _pick_device()
        _dtype = _pick_dtype(_device)

        print(f"Loading model on {_device}...")

        if os.path.isdir(merged_path):
            ...

        elif os.path.isdir(adapter_path):

            print("Importing PEFT...")
            from peft import PeftModel
            print("PEFT imported.")

            print("Loading tokenizer...")
            _tokenizer = AutoTokenizer.from_pretrained(adapter_path)

            print("Tokenizer loaded.")

            print("Loading base model...")
            base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=_dtype,
            )

            print("Base model loaded.")

            print("Loading LoRA adapter...")
            _model = PeftModel.from_pretrained(base, adapter_path)

            print("Model loaded.")
        else:
            _model_unavailable_reason = (
                f"No local model found (checked '{merged_path}' and '{adapter_path}'). "
                f"Run finetune/scripts/train_lora.py first, or pass mock=True."
            )
            return

        _model.to(_device)
        _model.eval()

    except Exception as e:  # missing deps, OOM, corrupt checkpoint, etc.
        _model_unavailable_reason = f"Local model failed to load ({e})."


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
    """
    Runs the local fine-tuned model on a system + user prompt and returns
    the generated text.

    If mock=True, or no local model is available, no inference is run --
    `mock_response` (or a generic placeholder) is returned instead. Each
    agent supplies its own sensible mock_response so the pipeline stays
    meaningfully testable without a trained model.

    Real (non-mock) calls are memoized in-process by (system_prompt,
    user_prompt, max_tokens): decoding is deterministic (do_sample=False),
    so an identical prompt always produces identical output, and re-running
    the same patient (dev loop, retries, demos) skips the generation pass
    entirely on a cache hit.
    """
    if mock:
        return mock_response or "[MOCK MODE] Local model call skipped -- returning placeholder output."

    _load_model()

    if _model is None:
        return mock_response or f"[MOCK MODE] {_model_unavailable_reason}"

    cache_key = _cache_key(system_prompt, user_prompt, max_tokens)
    cached = _response_cache.get(cache_key)
    if cached is not None:
        return cached

    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(prompt_text, return_tensors="pt").to(_device)

    with torch.no_grad():
     output_ids = _model.generate(
        **inputs,
        max_new_tokens=max_tokens,

        # Deterministic decoding
        do_sample=False,
        num_beams=1,

        # Better stability
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,

        # Faster inference
        use_cache=True,

        eos_token_id=_tokenizer.eos_token_id,
        pad_token_id=_tokenizer.eos_token_id,
         )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = _tokenizer.decode(generated, skip_special_tokens=True)
    text = text.strip()

    # Remove markdown fences if the model emits them
    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    _response_cache[cache_key] = text
    return text
