#!/usr/bin/env python3
"""
Inference script for Mamba CPT model.

This script loads a continued-pretrained Mamba model and generates text
from given prompts. It supports both single and multiple prompts.

Usage:
    python -m src.eval.infer --checkpoint_dir ./output/checkpoints/final
    python -m src.eval.infer --checkpoint_dir ./output/checkpoints/final --prompt "Merhaba dünya"
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, MambaForCausalLM


# =============================================================================
# Configuration Constants
# =============================================================================

# Base model for tokenizer (CPT did not modify the tokenizer)
BASE_MODEL_ID = "state-spaces/mamba-370m-hf"

# Default generation parameters (tuned for creative Turkish text generation)
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.92
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 1.1

# Default Turkish prompts for sanity checking
DEFAULT_PROMPTS = [
    "Türkiye'de yazılım mühendisliği öğrencisi olmak",
    "Yapay zeka ve büyük dil modelleri hakkında kısa bir açıklama",
    "Bir orta çağ fantastik evreninde geçen kısa bir hikaye",
]


# =============================================================================
# Core Functions
# =============================================================================

def get_device() -> torch.device:
    """Get the best available device (CUDA preferred)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("[WARNING] CUDA not available, using CPU (this will be slow!)")
    return device


def load_tokenizer(base_model_id: str = BASE_MODEL_ID) -> AutoTokenizer:
    """
    Load tokenizer from the base model.
    
    The tokenizer was NOT modified during CPT, so we load it from the
    original base model to ensure compatibility.
    """
    print(f"[INFO] Loading tokenizer from: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
    )
    
    # Ensure pad token is set (Mamba uses eos_token as pad_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[INFO] Set pad_token = eos_token")
    
    print(f"[INFO] Tokenizer vocab size: {len(tokenizer)}")
    return tokenizer


def load_model(checkpoint_dir: str, device: torch.device) -> MambaForCausalLM:
    """
    Load the CPT model from a local checkpoint directory.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory containing model.safetensors
        device: Target device (cuda or cpu)
    
    Returns:
        Loaded model in eval mode on the target device
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    
    if not (checkpoint_path / "model.safetensors").exists():
        raise FileNotFoundError(f"model.safetensors not found in: {checkpoint_path}")
    
    print(f"[INFO] Loading model from: {checkpoint_path}")
    
    # Load model - use bfloat16 to match training dtype and save memory
    model = MambaForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=None,  # We'll move to device manually
        trust_remote_code=True,
    )
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model loaded successfully")
    print(f"[INFO] Model parameters: {num_params / 1e6:.2f}M")
    print(f"[INFO] Model dtype: {next(model.parameters()).dtype}")
    print(f"[INFO] Model device: {next(model.parameters()).device}")
    
    return model


def generate_text(
    model: MambaForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    do_sample: bool = True,
) -> str:
    """
    Generate text continuation for a given prompt.
    
    Args:
        model: Loaded Mamba model
        tokenizer: Loaded tokenizer
        prompt: Input text prompt
        device: Target device
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (higher = more creative)
        top_p: Nucleus sampling probability
        top_k: Top-k sampling
        repetition_penalty: Penalty for repeating tokens
        do_sample: Whether to use sampling (vs greedy)
    
    Returns:
        Generated text (prompt + continuation)
    """
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512,  # Leave room for generation
    )
    input_ids = inputs["input_ids"].to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def run_inference(
    checkpoint_dir: str,
    prompts: list[str] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    do_sample: bool = True,
) -> list[dict]:
    """
    Run inference on one or more prompts.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        prompts: List of prompts (uses defaults if None)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
        do_sample: Whether to sample
    
    Returns:
        List of dicts with 'prompt' and 'generated' keys
    """
    if prompts is None:
        prompts = DEFAULT_PROMPTS
    
    # Setup
    device = get_device()
    tokenizer = load_tokenizer()
    model = load_model(checkpoint_dir, device)
    
    print("\n" + "=" * 70)
    print("STARTING INFERENCE")
    print("=" * 70 + "\n")
    
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Generating for prompt: \"{prompt[:50]}...\"")
        print("-" * 50)
        
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
        
        results.append({
            "prompt": prompt,
            "generated": generated,
        })
        
        # Print result
        print(f"PROMPT: {prompt}")
        print(f"\nGENERATED:\n{generated}")
        print("\n" + "=" * 70 + "\n")
    
    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with a continued-pretrained Mamba model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./output/checkpoints/final",
        help="Path to the checkpoint directory containing model.safetensors",
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate from (uses defaults if not provided)",
    )
    
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to a text file with prompts (one per line)",
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of new tokens to generate",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature (higher = more creative)",
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Nucleus sampling probability",
    )
    
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("MAMBA CPT INFERENCE SCRIPT")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Sampling: {not args.no_sample}")
    print("=" * 70 + "\n")
    
    # Determine prompts
    prompts = None
    
    if args.prompt:
        prompts = [args.prompt]
        print(f"[INFO] Using single prompt from CLI")
    elif args.prompts_file:
        prompts_path = Path(args.prompts_file)
        if not prompts_path.exists():
            print(f"[ERROR] Prompts file not found: {prompts_path}")
            sys.exit(1)
        prompts = [line.strip() for line in prompts_path.read_text().splitlines() if line.strip()]
        print(f"[INFO] Loaded {len(prompts)} prompts from: {prompts_path}")
    else:
        print(f"[INFO] Using {len(DEFAULT_PROMPTS)} default Turkish prompts")
    
    # Run inference
    try:
        results = run_inference(
            checkpoint_dir=args.checkpoint_dir,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=not args.no_sample,
        )
        print(f"[SUCCESS] Generated {len(results)} responses")
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
