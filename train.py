"""Jamba2 3B continued pretraining orchestration."""

import argparse
import glob
import logging
import os
import random
import sys
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import (
    read_jsonl_files,
    preprocess_texts,
    pack_and_tokenize_to_memmap,
    MemmapPackedDataset,
    create_dataloader,
)
from src.model import load_model, load_tokenizer
from src.train import Trainer
from src.utils import check_environment, setup_logging

logger = logging.getLogger(__name__)


# --- Memory instrumentation (Stage 0) ---
def log_mem(label: str):
    """Log current process RSS memory usage."""
    try:
        import psutil
        rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
        logger.info(f"[MEM] {label}: {rss_gb:.2f} GB RSS")
    except ImportError:
        pass


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """Override config values with CLI arguments."""
    # Override training params
    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = args.max_steps
    if args.time_budget is not None:
        config.setdefault("training", {})["time_budget_seconds"] = args.time_budget
    if args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = args.lr
    
    # Override logging
    if args.log_every_steps is not None:
        config.setdefault("logging", {})["log_every_steps"] = args.log_every_steps
    
    # Override checkpointing
    if args.checkpoint_every_steps is not None:
        config.setdefault("checkpointing", {})["checkpoint_every_steps"] = args.checkpoint_every_steps
    
    return config


def set_seed(seed: int, deterministic: bool = False):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def save_resolved_config(config: dict, output_dir: str):
    """Save the final resolved config."""
    output_path = Path(output_dir) / "resolved_config.yaml"
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Resolved config saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Jamba2 3B Continued Pretraining for Turkish"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/train.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max training steps"
    )
    parser.add_argument(
        "--time_budget",
        type=int,
        default=None,
        help="Override time budget in seconds"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--log_every_steps",
        type=int,
        default=None,
        help="Override logging frequency"
    )
    parser.add_argument(
        "--checkpoint_every_steps",
        type=int,
        default=None,
        help="Override checkpoint frequency"
    )
    parser.add_argument(
        "--skip_env_check",
        action="store_true",
        help="Skip environment check"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from (e.g., output/checkpoints/step_005000)"
    )
    
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    config = merge_config_with_args(config, args)
    
    # Setup output directory
    output_dir = config.get("checkpointing", {}).get("output_dir", "./output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"
    setup_logging(log_level="INFO", log_file=str(log_file))
    
    logger.info("=" * 60)
    logger.info("Jamba2 CPT Pipeline - Turkish")
    logger.info("=" * 60)
    
    # Environment check
    if not args.skip_env_check:
        env_info = check_environment(verbose=True)
        if not env_info.get("cuda_available"):
            logger.warning("CUDA not available - training will be slow!")
    
    # Set seed
    seed = config.get("seed", 42)
    deterministic = config.get("deterministic", False)
    set_seed(seed, deterministic)
    logger.info(f"Random seed: {seed}")

    # RTX 6000 Ada benefits from TF32 for matmuls while keeping bf16 activations.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("TF32 matmul/cuDNN enabled")
    
    # Save resolved config
    save_resolved_config(config, output_dir)
    
    # === Data Preparation ===
    logger.info("Preparing data...")
    log_mem("before data preparation")
    
    data_cfg = config.get("data", {})
    dataset_dir = data_cfg.get("dataset_dir", "./dataset")
    file_pattern = data_cfg.get("file_pattern", "*.jsonl")
    text_field = data_cfg.get("text_field", "text")
    min_text_length = data_cfg.get("min_text_length", 50)
    seq_len = data_cfg.get("seq_len", 1024)
    
    # Find data files
    data_files = glob.glob(os.path.join(dataset_dir, file_pattern))
    if not data_files:
        logger.error(f"No data files found in {dataset_dir}/{file_pattern}")
        logger.error("Please add JSONL files to the dataset directory.")
        sys.exit(1)
    
    logger.info(f"Found {len(data_files)} data files")
    
    # Load tokenizer first (needed for packing)
    model_name = config.get("model", {}).get("name", "ai21labs/AI21-Jamba2-3B")
    tokenizer = load_tokenizer(model_name)
    
    # Read, preprocess, tokenize, and pack to memmap (memory-efficient)
    log_mem("before tokenization")
    texts = read_jsonl_files(data_files, text_field=text_field)
    texts = preprocess_texts(texts, min_length=min_text_length)
    
    cache_dir = str(Path(output_dir) / "token_cache")
    memmap_path, num_chunks = pack_and_tokenize_to_memmap(
        texts, tokenizer, seq_len=seq_len,
        cache_dir=cache_dir,
    )
    log_mem("after tokenization + memmap write")
    
    if num_chunks == 0:
        logger.error("No valid chunks created from data!")
        sys.exit(1)
    
    # Create dataset from memmap (near-zero RAM)
    dataset = MemmapPackedDataset(memmap_path, num_chunks, seq_len)
    log_mem("after MemmapPackedDataset created")
    
    train_cfg = config.get("training", {})
    dataloader = create_dataloader(
        dataset,
        batch_size=train_cfg.get("micro_batch_size", 1),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
    )
    
    logger.info(f"Dataset: {len(dataset)} chunks of {seq_len} tokens")
    
    # === Model Loading ===
    logger.info("Loading model...")
    
    model_cfg = config.get("model", {})
    
    # Determine model source: checkpoint (if resuming) or base model
    if args.resume:
        model_source = args.resume
        logger.info(f"Resuming from checkpoint: {model_source}")
    else:
        model_source = model_cfg.get("name", "ai21labs/AI21-Jamba2-3B")

    model = load_model(
        model_name=model_source,
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        use_mamba_kernels=model_cfg.get("use_mamba_kernels", True),
        use_cache=model_cfg.get("use_cache", False),
    )
    log_mem("after model loaded")
    
    # === Training ===
    logger.info("Initializing trainer...")
    
    trainer = Trainer(
        model=model,
        train_loader=dataloader,
        config=config,
        output_dir=output_dir,
        resume_from_checkpoint=args.resume,
        tokenizer=tokenizer,
    )
    
    logger.info("Starting training...")
    summary = trainer.train()
    
    logger.info("Training complete!")
    logger.info(f"Summary: {summary}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
