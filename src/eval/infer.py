#!/usr/bin/env python3
"""Inference script for Jamba2 CPT checkpoints."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device

    print("[WARNING] CUDA not available, using CPU")
    return torch.device("cpu")


def load_tokenizer(checkpoint_dir: str) -> AutoTokenizer:
    print(f"[INFO] Loading tokenizer from: {checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(checkpoint_dir: str, device: torch.device) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    print(f"[INFO] Loading model from: {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    )
    model = model.to(device)
    model.eval()
    return model


def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    do_sample: bool = True,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

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
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def run_inference(
    checkpoint_dir: str,
    prompts: list[str] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    do_sample: bool = True,
) -> list[dict]:
    prompts = prompts or DEFAULT_PROMPTS
    device = get_device()
    tokenizer = load_tokenizer(checkpoint_dir)
    model = load_model(checkpoint_dir, device)

    results = []
    for prompt in prompts:
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        results.append({"prompt": prompt, "generated": generated})
        print(f"PROMPT: {prompt}\n")
        print(f"GENERATED:\n{generated}\n")
        print("=" * 70)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a continued-pretrained Jamba model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint_dir", type=str, default="./output/checkpoints/final")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--no_sample", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

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
            checkpoint_dir=args.checkpoint_dir,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=not args.no_sample,
        )
    except Exception as exc:
        print(f"[ERROR] Inference failed: {exc}")
        raise


if __name__ == "__main__":
    main()
