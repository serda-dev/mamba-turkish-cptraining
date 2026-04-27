#!/usr/bin/env python3
"""Inference script for local or Hugging Face Jamba2 checkpoints."""

import argparse
import sys
from pathlib import Path

import torch

from src.model.load import load_model_and_tokenizer


DEFAULT_MODEL_NAME = "serda-dev/Jamba2-3B-Turkish"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.92
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 1.1
DEFAULT_PROMPTS = [
    "Türkiye'de yazılım mühendisliği öğrencisi olmak",
    "Yapay zeka ve büyük dil modelleri hakkında kısa bir açıklama",
    "Bir orta çağ fantastik evreninde geçen kısa bir hikaye",
]
DEFAULT_SYSTEM_PROMPT = (
    "Sen akıcı ve doğal Türkçe yazan bir yardımcı asistansın. "
    "Yanıtların açık, tutarlı ve doğrudan olsun."
)


def get_runtime_device(requested_device: str | None, device_map: str | None) -> str | None:
    """Resolve the preferred runtime placement for model loading."""
    if device_map is not None:
        print(f"[INFO] Using device_map={device_map}")
        return None

    if requested_device:
        return requested_device

    if torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return "cuda"

    print("[WARNING] CUDA not available, using CPU")
    return "cpu"


def get_model_input_device(model: torch.nn.Module) -> torch.device:
    """Find the device where input tensors should be placed."""
    if hasattr(model, "device") and getattr(model, "device").type != "meta":
        return model.device
    return next(model.parameters()).device


def build_prompt(tokenizer, prompt: str, system_prompt: str | None) -> str:
    """Mirror the original Jamba2 chat-style call when a chat template is available."""
    if getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt


def load_inference_artifacts(
    model_name_or_path: str,
    tokenizer_name_or_path: str | None,
    torch_dtype: str,
    requested_device: str | None,
    device_map: str | None,
    attn_implementation: str,
) -> tuple[torch.nn.Module, object]:
    runtime_device = get_runtime_device(requested_device, device_map)
    print(f"[INFO] Loading model from: {model_name_or_path}")
    if tokenizer_name_or_path:
        print(f"[INFO] Loading tokenizer from: {tokenizer_name_or_path}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        torch_dtype=torch_dtype,
        device=runtime_device,
        device_map=device_map,
        attn_implementation=attn_implementation,
        use_cache=True,
    )
    model.eval()

    input_device = get_model_input_device(model)
    print(f"[INFO] Model input device: {input_device}")
    print(f"[INFO] Tokenizer size: {len(tokenizer)}")
    if getattr(tokenizer, "chat_template", None):
        print("[INFO] Chat template detected; prompts will use chat formatting")
    else:
        print("[INFO] No chat template found; prompts will be used as plain text")

    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str | None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    do_sample: bool = True,
) -> str:
    rendered_prompt = build_prompt(tokenizer, prompt, system_prompt)
    input_device = get_model_input_device(model)

    inputs = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=2048,
    )
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        })

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)

    prompt_length = inputs["input_ids"].shape[1]
    completion_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()


def run_inference(
    model_name_or_path: str,
    tokenizer_name_or_path: str | None = None,
    prompts: list[str] | None = None,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    do_sample: bool = True,
    torch_dtype: str = "bfloat16",
    device: str | None = None,
    device_map: str | None = None,
    attn_implementation: str = "auto",
) -> list[dict]:
    prompts = prompts or DEFAULT_PROMPTS
    model, tokenizer = load_inference_artifacts(
        model_name_or_path=model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        torch_dtype=torch_dtype,
        requested_device=device,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )

    results = []
    for prompt in prompts:
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )
        results.append({"prompt": prompt, "generated": generated})
        print(f"PROMPT: {prompt}\n")
        print(f"GENERATED:\n{generated}\n")
        print("=" * 70)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a local checkpoint or Hugging Face Jamba model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Backward-compatible alias for --model_name_or_path",
    )
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default="auto")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--repetition_penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    parser.add_argument("--no_sample", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    model_name_or_path = args.checkpoint_dir or args.model_name_or_path
    if model_name_or_path is None:
        print("[ERROR] No model source was provided")
        sys.exit(1)

    prompts = None
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        prompts_path = Path(args.prompts_file)
        if not prompts_path.exists():
            print(f"[ERROR] Prompts file not found: {prompts_path}")
            sys.exit(1)
        prompts = [line.strip() for line in prompts_path.read_text().splitlines() if line.strip()]

    try:
        run_inference(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            prompts=prompts,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            do_sample=not args.no_sample,
            torch_dtype=args.torch_dtype,
            device=args.device,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
        )
    except Exception as exc:
        print(f"[ERROR] Inference failed: {exc}")
        raise


if __name__ == "__main__":
    main()
