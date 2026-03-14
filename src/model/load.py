"""Model and tokenizer loading for Jamba CPT."""

import logging
from typing import Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def parse_torch_dtype(torch_dtype: str) -> torch.dtype:
    """Map a user-facing dtype string to a torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(torch_dtype.lower(), torch.bfloat16)


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load tokenizer for a pretrained causal LM checkpoint."""
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Tokenizer had no pad token, reusing eos token as pad")
        else:
            raise ValueError("Tokenizer has neither pad_token nor eos_token")

    logger.info(
        "Tokenizer loaded: vocab_size=%s, pad_token_id=%s, eos_token_id=%s",
        tokenizer.vocab_size,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
    )
    return tokenizer


def load_model(
    model_name: str,
    torch_dtype: str = "bfloat16",
    device: Optional[str] = None,
    attn_implementation: str = "sdpa",
    use_mamba_kernels: bool = True,
    use_cache: bool = False,
) -> torch.nn.Module:
    """Load a causal LM with Jamba-specific config overrides."""
    logger.info(f"Loading model: {model_name}")

    dtype = parse_torch_dtype(torch_dtype)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, "use_mamba_kernels"):
        config.use_mamba_kernels = use_mamba_kernels
    if hasattr(config, "use_cache"):
        config.use_cache = use_cache
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = attn_implementation

    logger.info(
        "Model config overrides: dtype=%s, device=%s, attn_implementation=%s, "
        "use_mamba_kernels=%s, use_cache=%s",
        dtype,
        device,
        attn_implementation,
        use_mamba_kernels,
        use_cache,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model loaded: %s total params, %s trainable (%.1f%%)",
        f"{total_params:,}",
        f"{trainable_params:,}",
        trainable_params / total_params * 100,
    )
    return model


def load_model_and_tokenizer(
    model_name: str,
    torch_dtype: str = "bfloat16",
    device: Optional[str] = None,
    attn_implementation: str = "sdpa",
    use_mamba_kernels: bool = True,
    use_cache: bool = False,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Convenience function to load both model and tokenizer."""
    tokenizer = load_tokenizer(model_name)
    model = load_model(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device=device,
        attn_implementation=attn_implementation,
        use_mamba_kernels=use_mamba_kernels,
        use_cache=use_cache,
    )
    return model, tokenizer
