"""Load Mamba model and tokenizer from HuggingFace."""

import logging
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, MambaForCausalLM

logger = logging.getLogger(__name__)


def load_tokenizer(
    model_name: str = "state-spaces/mamba-130m-hf",
) -> AutoTokenizer:
    """
    Load tokenizer for Mamba model.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Configured tokenizer
    """
    logger.info(f"Loading tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: '{tokenizer.pad_token}'")
    
    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    return tokenizer


def load_model(
    model_name: str = "state-spaces/mamba-130m-hf",
    torch_dtype: str = "float16",
    device: Optional[str] = None,
) -> MambaForCausalLM:
    """
    Load Mamba model for causal language modeling.
    
    Args:
        model_name: HuggingFace model identifier
        torch_dtype: "float16", "bfloat16", or "float32"
        device: Target device (auto-detected if None)
        
    Returns:
        Loaded model on specified device
    """
    logger.info(f"Loading model: {model_name}")
    
    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype.lower(), torch.float16)
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using dtype={dtype}, device={device}")
    
    # Load model
    model = MambaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )
    
    # Move to device
    model = model.to(device)
    
    # Enable gradient checkpointing for memory efficiency (optional)
    # model.gradient_checkpointing_enable()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(
        f"Model loaded: {total_params:,} total params, "
        f"{trainable_params:,} trainable ({trainable_params/total_params*100:.1f}%)"
    )
    
    return model


def load_model_and_tokenizer(
    model_name: str = "state-spaces/mamba-130m-hf",
    torch_dtype: str = "float16",
    device: Optional[str] = None,
) -> Tuple[MambaForCausalLM, AutoTokenizer]:
    """Convenience function to load both model and tokenizer."""
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, torch_dtype, device)
    return model, tokenizer
