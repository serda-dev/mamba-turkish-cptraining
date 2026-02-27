#!/usr/bin/env python3
"""
Simple perplexity evaluation script for Mamba CPT model.

This is a MINIMAL sanity-check tool, NOT a benchmark-grade evaluation.
It computes perplexity on a small set of texts to verify the model
is working correctly after continued pretraining.

Usage:
    python -m src.eval.perplexity --checkpoint_dir ./output/checkpoints/final
    python -m src.eval.perplexity --checkpoint_dir ./output/checkpoints/final --texts_file ./eval_texts.txt
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, MambaForCausalLM


# =============================================================================
# Configuration Constants
# =============================================================================

BASE_MODEL_ID = "state-spaces/mamba-370m-hf"

# Default evaluation texts (Turkish, covering different domains)
DEFAULT_EVAL_TEXTS = [
    "Yapay zeka, makinelerin insan benzeri zeka sergilemesini sağlayan bir bilgisayar bilimi dalıdır. Günümüzde derin öğrenme ve büyük dil modelleri bu alanda önemli gelişmeler kaydetmiştir.",
    "İstanbul, Türkiye'nin en kalabalık şehri ve ülkenin ekonomik, kültürel ve tarihi merkezidir. Boğaziçi ile Avrupa ve Asya kıtalarını birbirine bağlar.",
    "Yazılım mühendisliği, yazılım geliştirme süreçlerinin sistematik bir şekilde yönetilmesini içerir. Bu alan, tasarım, kodlama, test ve bakım aşamalarını kapsar.",
    "Türk mutfağı, Orta Asya'dan Anadolu'ya uzanan zengin bir tarihi yansıtır. Kebaplar, mezeler ve tatlılar bu mutfağın en bilinen öğeleri arasındadır.",
    "Makine öğrenmesi algoritmaları, verilerden örüntüler çıkararak tahminlerde bulunabilir. Denetimli ve denetimsiz öğrenme bu alanın temel yaklaşımlarıdır.",
]


# =============================================================================
# Core Functions
# =============================================================================

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[WARNING] CUDA not available, using CPU")
    return device


def load_tokenizer(base_model_id: str = BASE_MODEL_ID) -> AutoTokenizer:
    """Load tokenizer from base model."""
    print(f"[INFO] Loading tokenizer from: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(checkpoint_dir: str, device: torch.device) -> MambaForCausalLM:
    """Load model from checkpoint."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"[INFO] Loading model from: {checkpoint_path}")
    
    model = MambaForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    )
    
    model = model.to(device)
    model.eval()
    
    print(f"[INFO] Model loaded successfully")
    return model


def compute_perplexity(
    model: MambaForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
    max_length: int = 512,
) -> tuple[float, int]:
    """
    Compute perplexity for a single text.
    
    This is a simple implementation that:
    1. Tokenizes the text
    2. Computes cross-entropy loss
    3. Converts to perplexity
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        text: Input text
        device: Target device
        max_length: Maximum sequence length
    
    Returns:
        Tuple of (perplexity, num_tokens)
    """
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    
    input_ids = inputs["input_ids"].to(device)
    seq_len = input_ids.size(1)
    
    if seq_len < 2:
        return float("inf"), 0
    
    # Compute loss
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
    
    # Convert to perplexity
    perplexity = math.exp(loss.item())
    
    return perplexity, seq_len


def evaluate_perplexity(
    checkpoint_dir: str,
    texts: list[str] | None = None,
    max_length: int = 512,
) -> dict:
    """
    Evaluate perplexity on a list of texts.
    
    Args:
        checkpoint_dir: Path to checkpoint
        texts: List of texts (uses defaults if None)
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with evaluation results
    """
    if texts is None:
        texts = DEFAULT_EVAL_TEXTS
    
    device = get_device()
    tokenizer = load_tokenizer()
    model = load_model(checkpoint_dir, device)
    
    print("\n" + "=" * 70)
    print("PERPLEXITY EVALUATION")
    print("=" * 70 + "\n")
    
    results = []
    total_tokens = 0
    
    for i, text in enumerate(texts, 1):
        ppl, num_tokens = compute_perplexity(
            model=model,
            tokenizer=tokenizer,
            text=text,
            device=device,
            max_length=max_length,
        )
        
        results.append({
            "text_id": i,
            "text_preview": text[:60] + "..." if len(text) > 60 else text,
            "perplexity": ppl,
            "num_tokens": num_tokens,
        })
        
        total_tokens += num_tokens
        
        print(f"[{i}/{len(texts)}] PPL: {ppl:8.2f} | Tokens: {num_tokens:4d} | \"{text[:40]}...\"")
    
    # Compute average perplexity
    avg_ppl = sum(r["perplexity"] for r in results) / len(results)
    
    print("\n" + "-" * 70)
    print(f"SUMMARY:")
    print(f"  Total texts:      {len(texts)}")
    print(f"  Total tokens:     {total_tokens}")
    print(f"  Average PPL:      {avg_ppl:.2f}")
    print("-" * 70 + "\n")
    
    return {
        "individual_results": results,
        "average_perplexity": avg_ppl,
        "total_tokens": total_tokens,
        "num_texts": len(texts),
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simple perplexity evaluation for Mamba CPT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./output/checkpoints/final",
        help="Path to checkpoint directory",
    )
    
    parser.add_argument(
        "--texts_file",
        type=str,
        default=None,
        help="Path to text file with evaluation texts (one per line)",
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for evaluation",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("MAMBA CPT PERPLEXITY EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Max length: {args.max_length}")
    print("=" * 70 + "\n")
    
    # Load texts
    texts = None
    if args.texts_file:
        texts_path = Path(args.texts_file)
        if not texts_path.exists():
            print(f"[ERROR] Texts file not found: {texts_path}")
            sys.exit(1)
        texts = [line.strip() for line in texts_path.read_text().splitlines() if line.strip()]
        print(f"[INFO] Loaded {len(texts)} texts from: {texts_path}")
    else:
        print(f"[INFO] Using {len(DEFAULT_EVAL_TEXTS)} default evaluation texts")
    
    # Run evaluation
    try:
        results = evaluate_perplexity(
            checkpoint_dir=args.checkpoint_dir,
            texts=texts,
            max_length=args.max_length,
        )
        print(f"[SUCCESS] Evaluation complete")
        print(f"[RESULT] Average Perplexity: {results['average_perplexity']:.2f}")
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
