#!/usr/bin/env python3
"""Minimal perplexity evaluation for local or Hub-hosted Jamba checkpoints."""

import argparse
import math
import sys
from pathlib import Path

import torch

from src.model.load import load_model_and_tokenizer


DEFAULT_EVAL_TEXTS = [
    "Yapay zeka, makinelerin insan benzeri zeka sergilemesini saglayan bir bilgisayar bilimi dalidir.",
    "Istanbul, Turkiye'nin en kalabalik sehri ve ulkenin ekonomik merkezlerinden biridir.",
    "Yazilim muhendisligi, tasarim, gelistirme, test ve bakim sureclerini kapsar.",
]


def get_runtime_device(requested_device: str | None) -> str:
    """Resolve the evaluation device."""
    if requested_device:
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def evaluate_perplexity(
    model_name_or_path: str,
    tokenizer_name_or_path: str | None = None,
    texts: list[str] | None = None,
    max_length: int = 512,
    torch_dtype: str = "bfloat16",
    device: str | None = None,
    attn_implementation: str = "auto",
) -> dict:
    texts = texts or DEFAULT_EVAL_TEXTS
    runtime_device = get_runtime_device(device)
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        torch_dtype=torch_dtype,
        device=runtime_device,
        attn_implementation=attn_implementation,
        use_cache=False,
    )
    model.eval()
    eval_device = next(model.parameters()).device

    results = []
    total_tokens = 0
    for i, text in enumerate(texts, 1):
        ppl, num_tokens = compute_perplexity(model, tokenizer, text, eval_device, max_length=max_length)
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
        description="Simple perplexity evaluation for a Jamba model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name_or_path", type=str, default="./output/checkpoints/final")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Backward-compatible alias for --model_name_or_path",
    )
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--texts_file", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default="auto")
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
        model_name_or_path=args.checkpoint_dir or args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        texts=texts,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )
    print(f"[RESULT] Average Perplexity: {results['average_perplexity']:.2f}")


if __name__ == "__main__":
    main()
