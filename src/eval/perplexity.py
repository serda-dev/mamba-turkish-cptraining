#!/usr/bin/env python3
"""Minimal perplexity evaluation for Jamba CPT checkpoints."""

import argparse
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_EVAL_TEXTS = [
    "Yapay zeka, makinelerin insan benzeri zeka sergilemesini sağlayan bir bilgisayar bilimi dalıdır.",
    "İstanbul, Türkiye'nin en kalabalık şehri ve ülkenin ekonomik merkezlerinden biridir.",
    "Yazılım mühendisliği, tasarım, geliştirme, test ve bakım süreçlerini kapsar.",
]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(checkpoint_dir: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(checkpoint_dir: str, device: torch.device) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    )
    model = model.to(device)
    model.eval()
    return model


def compute_perplexity(model, tokenizer, text: str, device: torch.device, max_length: int = 512) -> tuple[float, int]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    seq_len = inputs["input_ids"].size(1)
    if seq_len < 2:
        return float("inf"), 0

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"], use_cache=False)
    return math.exp(outputs.loss.item()), seq_len


def evaluate_perplexity(checkpoint_dir: str, texts: list[str] | None = None, max_length: int = 512) -> dict:
    texts = texts or DEFAULT_EVAL_TEXTS
    device = get_device()
    tokenizer = load_tokenizer(checkpoint_dir)
    model = load_model(checkpoint_dir, device)

    results = []
    total_tokens = 0
    for i, text in enumerate(texts, 1):
        ppl, num_tokens = compute_perplexity(model, tokenizer, text, device, max_length=max_length)
        results.append({"text_id": i, "perplexity": ppl, "num_tokens": num_tokens})
        total_tokens += num_tokens
        print(f"[{i}/{len(texts)}] PPL: {ppl:8.2f} | Tokens: {num_tokens:4d}")

    avg_ppl = sum(r["perplexity"] for r in results) / len(results)
    return {
        "individual_results": results,
        "average_perplexity": avg_ppl,
        "total_tokens": total_tokens,
        "num_texts": len(texts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple perplexity evaluation for Jamba CPT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint_dir", type=str, default="./output/checkpoints/final")
    parser.add_argument("--texts_file", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    texts = None
    if args.texts_file:
        texts_path = Path(args.texts_file)
        if not texts_path.exists():
            print(f"[ERROR] Texts file not found: {texts_path}")
            sys.exit(1)
        texts = [line.strip() for line in texts_path.read_text().splitlines() if line.strip()]

    results = evaluate_perplexity(
        checkpoint_dir=args.checkpoint_dir,
        texts=texts,
        max_length=args.max_length,
    )
    print(f"[RESULT] Average Perplexity: {results['average_perplexity']:.2f}")


if __name__ == "__main__":
    main()
