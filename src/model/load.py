"""Model and tokenizer loading for Jamba CPT."""

import importlib.util
import logging
from typing import Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


FALLBACK_JAMBA_CHAT_TEMPLATE = """{% if bos_token is defined and bos_token is not none %}{{ bos_token }}{% endif %}{% for message in messages %}{% if message.role in ['system', 'user', 'assistant'] %}{{ '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"""


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


def resolve_attn_implementation(attn_implementation: str, device: Optional[str] = None) -> str:
    """Choose a practical attention backend for the current runtime."""
    if attn_implementation != "auto":
        return attn_implementation

    if device == "cpu" or (device is None and not torch.cuda.is_available()):
        return "sdpa"

    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"

    return "sdpa"


def load_tokenizer(tokenizer_name_or_path: str) -> AutoTokenizer:
    """Load tokenizer for a pretrained causal LM checkpoint."""
    logger.info(f"Loading tokenizer: {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Tokenizer had no pad token, reusing eos token as pad")
        else:
            raise ValueError("Tokenizer has neither pad_token nor eos_token")

    added_vocab = tokenizer.get_added_vocab()
    has_jamba_chat_tokens = "<|im_start|>" in added_vocab and "<|im_end|>" in added_vocab
    if not getattr(tokenizer, "chat_template", None) and has_jamba_chat_tokens:
        tokenizer.chat_template = FALLBACK_JAMBA_CHAT_TEMPLATE
        logger.info("Tokenizer had no chat template, injecting Jamba-compatible fallback template")

    logger.info(
        "Tokenizer loaded: vocab_size=%s, pad_token_id=%s, eos_token_id=%s, chat_template=%s",
        tokenizer.vocab_size,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        "yes" if getattr(tokenizer, "chat_template", None) else "no",
    )
    return tokenizer


def load_model(
    model_name: str,
    tokenizer=None,
    torch_dtype: str = "bfloat16",
    device: Optional[str] = None,
    attn_implementation: str = "auto",
    use_mamba_kernels: bool = True,
    use_cache: bool = False,
    device_map: Optional[str] = None,
    low_cpu_mem_usage: bool = True,
) -> torch.nn.Module:
    """Load a causal LM with Jamba-specific config overrides."""
    logger.info(f"Loading model: {model_name}")

    dtype = parse_torch_dtype(torch_dtype)
    if device is None and device_map is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_attn_implementation = resolve_attn_implementation(attn_implementation, device)

    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, "use_mamba_kernels"):
        config.use_mamba_kernels = use_mamba_kernels
    if hasattr(config, "use_cache"):
        config.use_cache = use_cache
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = resolved_attn_implementation

    logger.info(
        "Model config overrides: dtype=%s, device=%s, device_map=%s, "
        "attn_implementation=%s, use_mamba_kernels=%s, use_cache=%s",
        dtype,
        device,
        device_map,
        resolved_attn_implementation,
        use_mamba_kernels,
        use_cache,
    )

    model_load_kwargs = {
        "config": config,
        "dtype": dtype,
        "attn_implementation": resolved_attn_implementation,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    if device_map is not None:
        model_load_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_load_kwargs,
    )

    if tokenizer is not None and len(tokenizer) != model.get_input_embeddings().num_embeddings:
        old_size = model.get_input_embeddings().num_embeddings
        new_size = len(tokenizer)
        logger.info(
            "Resizing token embeddings to match custom tokenizer: %s -> %s",
            old_size,
            new_size,
        )
        model.resize_token_embeddings(new_size)

    if device_map is None and device is not None:
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
    tokenizer_name_or_path: Optional[str] = None,
    torch_dtype: str = "bfloat16",
    device: Optional[str] = None,
    attn_implementation: str = "auto",
    use_mamba_kernels: bool = True,
    use_cache: bool = False,
    device_map: Optional[str] = None,
    low_cpu_mem_usage: bool = True,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Convenience function to load both model and tokenizer."""
    tokenizer = load_tokenizer(tokenizer_name_or_path or model_name)
    model = load_model(
        model_name=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        device=device,
        attn_implementation=attn_implementation,
        use_mamba_kernels=use_mamba_kernels,
        use_cache=use_cache,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    return model, tokenizer
