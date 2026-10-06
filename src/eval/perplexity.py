#!/usr/bin/env python3
"""Token-weighted perplexity on explicit text or a verified heldout token cache."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from src.model.load import load_model_and_tokenizer


DEFAULT_EVAL_TEXTS = [
    "Yapay zeka, makinelerin insan benzeri zeka sergilemesini sağlayan bir bilgisayar bilimi dalıdır.",
    "İstanbul, Türkiye'nin en kalabalık şehri ve ülkenin ekonomik merkezlerinden biridir.",
    "Yazılım mühendisliği, tasarım, geliştirme, test ve bakım süreçlerini kapsar.",
]


def get_runtime_device(requested_device: str | None) -> str:
    return requested_device or ("cuda" if torch.cuda.is_available() else "cpu")


def _exp_nll(mean_nll: float) -> float:
    """Very bad checkpoints can overflow exp; preserve that result as infinity."""
    try:
        return math.exp(mean_nll)
    except OverflowError:
        return float("inf")


def compute_token_nll(model, inputs: dict, device: torch.device) -> tuple[float, int]:
    """Return summed causal NLL and the number of *predicted* nonpadding tokens."""
    inputs = {k: v.to(device) for k, v in inputs.items()
              if k in ("input_ids", "attention_mask", "labels")}
    labels = inputs.get("labels", inputs["input_ids"].clone())
    if "attention_mask" in inputs:
        labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)
    inputs["labels"] = labels
    count = int((labels[..., 1:] != -100).sum().item())
    if not count:
        return 0.0, 0
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)
    mean_nll = float(outputs.loss.item())
    if not math.isfinite(mean_nll):
        raise ValueError(f"Nonfinite evaluation loss: {mean_nll}")
    return mean_nll * count, count


def compute_perplexity(model, tokenizer, text: str, device: torch.device, max_length: int = 512) -> tuple[float, int]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_length, padding=False)
    nll, count = compute_token_nll(model, inputs, device)
    return (_exp_nll(nll / count), count) if count else (float("inf"), 0)


def evaluate_batches(model, batches, device, evaluation_kind: str) -> dict:
    results, total_nll, total_tokens = [], 0.0, 0
    for index, inputs in enumerate(batches, 1):
        nll, count = compute_token_nll(model, inputs, device)
        ppl = _exp_nll(nll / count) if count else float("inf")
        results.append({"text_id": index, "perplexity": ppl, "num_tokens": count})
        total_nll += nll
        total_tokens += count
    if not total_tokens:
        raise ValueError("Evaluation input contains no predictable tokens")
    mean_nll = total_nll / total_tokens
    return {"individual_results": results, "average_perplexity": _exp_nll(mean_nll),
            "mean_nll": mean_nll, "total_tokens": total_tokens,
            "num_texts": len(results), "evaluation_kind": evaluation_kind}


def _heldout_batches(cache_manifest: str, max_chunks: int | None, tokenizer=None):
    from src.data.dataloader import ShardedMemmapPackedDataset
    manifest_path = Path(cache_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest.get("cache_identity", {})
    split = identity.get("split") if isinstance(identity, dict) else None
    if split not in {"validation", "val", "test", "heldout"}:
        raise ValueError("Heldout cache must declare cache_identity.split=validation/val/test/heldout")
    if tokenizer is not None:
        from src.data.tokenize_pack import tokenization_fingerprint
        if manifest.get("tokenizer_fingerprint") != tokenization_fingerprint(tokenizer):
            raise ValueError("Heldout cache tokenizer fingerprint differs from evaluation tokenizer")
    dataset = ShardedMemmapPackedDataset(str(manifest_path))
    limit = len(dataset) if max_chunks is None else min(max_chunks, len(dataset))
    for index in range(limit):
        item = dataset[index]
        yield {key: value.unsqueeze(0) for key, value in item.items()
               if key in ("input_ids", "labels", "attention_mask")}


def evaluate_perplexity(
    model_name_or_path: str,
    tokenizer_name_or_path: str | None = None,
    texts: list[str] | None = None,
    max_length: int = 512,
    torch_dtype: str = "bfloat16",
    device: str | None = None,
    attn_implementation: str = "auto",
    revision: str | None = None,
    smoke_test: bool = False,
    cache_manifest: str | None = None,
    max_chunks: int | None = None,
) -> dict:
    if max_length < 2 or (max_chunks is not None and max_chunks < 1):
        raise ValueError("max_length must be >= 2 and max_chunks must be positive")
    if sum([texts is not None, smoke_test, cache_manifest is not None]) != 1:
        raise ValueError("Choose exactly one explicit texts input, heldout cache, or smoke_test")
    if smoke_test:
        texts = DEFAULT_EVAL_TEXTS
    if texts is not None and not texts:
        raise ValueError("Evaluation texts must not be empty")
    runtime_device = get_runtime_device(device)
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name_or_path, tokenizer_name_or_path=tokenizer_name_or_path,
        torch_dtype=torch_dtype, device=runtime_device,
        attn_implementation=attn_implementation, use_cache=False, revision=revision,
    )
    model.eval()
    eval_device = next(model.parameters()).device
    if cache_manifest:
        batches = _heldout_batches(cache_manifest, max_chunks, tokenizer=tokenizer)
        kind = "heldout_cache"
    else:
        batches = (tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_length, padding=False) for text in texts)
        kind = "smoke_test" if smoke_test else "explicit_texts"
    return evaluate_batches(model, batches, eval_device, kind)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name_or_path", default="./output/checkpoints/final")
    parser.add_argument("--checkpoint_dir", default=None, help="Alias for --model_name_or_path")
    parser.add_argument("--tokenizer_name_or_path", default=None)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--texts_file", help="UTF-8 file with one evaluation document per line")
    source.add_argument("--cache_manifest", help="Complete sharded cache manifest with a heldout split identity")
    source.add_argument("--smoke_test", action="store_true", help="Tiny plumbing check; not a heldout model score")
    parser.add_argument("--max_chunks", type=int, default=None, help="Optional heldout cache chunk limit")
    parser.add_argument("--max_length", type=int, default=512, help="Text truncation length; caches use their stored length")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn_implementation", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    texts = None
    if args.texts_file:
        texts_path = Path(args.texts_file)
        if not texts_path.exists():
            print(f"[ERROR] Texts file not found: {texts_path}")
            sys.exit(1)
        texts = [line.strip() for line in texts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = evaluate_perplexity(
        model_name_or_path=args.checkpoint_dir or args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path, texts=texts,
        max_length=args.max_length, torch_dtype=args.torch_dtype, device=args.device,
        attn_implementation=args.attn_implementation, revision=args.revision,
        smoke_test=args.smoke_test, cache_manifest=args.cache_manifest, max_chunks=args.max_chunks,
    )
    print(f"[RESULT] {results['evaluation_kind']}: token-weighted PPL={results['average_perplexity']:.2f}; "
          f"predicted tokens={results['total_tokens']}")


if __name__ == "__main__":
    main()
