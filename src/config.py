"""Config loading and normalization for single-phase and curriculum CPT."""

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return apply_env_overrides(config)


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply Docker/server-friendly path and resume overrides."""
    config = copy.deepcopy(config)
    paths = config.setdefault("paths", {})
    env_to_path = {
        "DATA_DIR": "data_dir",
        "CACHE_DIR": "cache_dir",
        "CHECKPOINT_DIR": "checkpoint_dir",
        "LOG_DIR": "log_dir",
        "MANIFEST_DIR": "manifest_dir",
    }
    for env_name, key in env_to_path.items():
        value = os.getenv(env_name)
        if value:
            paths[key] = value

    resume = os.getenv("CPT_RESUME")
    if resume:
        config.setdefault("training", {})["resume"] = resume
    return config


def save_yaml_config(config: Dict[str, Any], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def get_training_mode(config: Dict[str, Any]) -> str:
    training = config.get("training", {})
    return training.get("mode") or config.get("training_mode") or "single_phase"


def get_seq_len(config: Dict[str, Any]) -> int:
    return int(
        config.get("training", {}).get(
            "seq_len",
            config.get("data", {}).get("seq_len", 1024),
        )
    )


def get_output_dir(config: Dict[str, Any]) -> str:
    paths = config.get("paths", {})
    if paths.get("checkpoint_dir"):
        return str(Path(paths["checkpoint_dir"]).parent)
    return config.get("checkpointing", {}).get("output_dir", "./output")


def get_checkpoint_dir(config: Dict[str, Any]) -> str:
    paths = config.get("paths", {})
    return paths.get("checkpoint_dir") or str(Path(get_output_dir(config)) / "checkpoints")


def get_log_dir(config: Dict[str, Any]) -> str:
    paths = config.get("paths", {})
    if paths.get("log_dir"):
        return paths["log_dir"]
    return str(Path(get_output_dir(config)) / "logs")


def get_manifest_dir(config: Dict[str, Any]) -> str:
    return config.get("paths", {}).get("manifest_dir", "artifacts/dataset_manifests")


def phase_by_id(config: Dict[str, Any], phase_id: int) -> Optional[Dict[str, Any]]:
    for phase in config.get("phases", []):
        if int(phase.get("id")) == int(phase_id):
            return phase
    return None


def phase_training_config(config: Dict[str, Any], phase: Dict[str, Any]) -> Dict[str, Any]:
    """Return a training config with phase optimizer/scheduler overrides applied."""
    resolved = copy.deepcopy(config)
    train_cfg = resolved.setdefault("training", {})
    phase_optimizer = phase.get("optimizer", {})
    if "learning_rate" in phase_optimizer:
        train_cfg["learning_rate"] = phase_optimizer["learning_rate"]
    if "weight_decay" in phase_optimizer:
        train_cfg["weight_decay"] = phase_optimizer["weight_decay"]
    if "warmup_ratio" in phase_optimizer:
        train_cfg["warmup_ratio"] = phase_optimizer["warmup_ratio"]
    if "warmup_steps" in phase_optimizer:
        train_cfg["warmup_steps"] = phase_optimizer["warmup_steps"]
    train_cfg["scheduler"] = phase.get("scheduler", train_cfg.get("scheduler", "cosine"))
    train_cfg["phase_id"] = int(phase["id"])
    train_cfg["phase_name"] = phase.get("name", f"phase_{phase['id']}")
    return resolved


def calculate_global_batch(
    seq_len: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int = 1,
) -> Dict[str, int]:
    samples = micro_batch_size * gradient_accumulation_steps * world_size
    return {
        "seq_len": seq_len,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "world_size": world_size,
        "global_batch_samples": samples,
        "global_batch_tokens": seq_len * samples,
    }
