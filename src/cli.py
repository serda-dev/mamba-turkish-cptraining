"""Command-line entry point for single-phase and four-phase CPT workflows."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .config import (
    calculate_global_batch,
    get_checkpoint_dir,
    get_log_dir,
    get_manifest_dir,
    get_seq_len,
    get_training_mode,
    load_yaml_config,
    phase_training_config,
    save_yaml_config,
)
from .data import pack_and_tokenize_to_sharded_cache, token_cache_is_complete
from .data.curriculum import (
    iter_english_texts,
    iter_turkish_texts,
    prepare_manifests,
    read_json,
)
from .data.mixing import weighted_mix_texts
from .data.preprocess import clean_text, strip_legacy_end_markers
from .train.checkpoint import find_latest_checkpoint, read_latest_metadata
from .utils import check_environment, setup_logging

logger = logging.getLogger(__name__)


FALLBACK_JAMBA_CHAT_TEMPLATE = """{% if bos_token is defined and bos_token is not none %}{{ bos_token }}{% endif %}{% for message in messages %}{% if message.role in ['system', 'user', 'assistant'] %}{{ '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"""


def load_tokenizer_for_cli(tokenizer_name_or_path: str):
    """Load tokenizer without importing torch, so VPS preprocessing stays CPU-only."""
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer: %s", tokenizer_name_or_path)
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


def set_seed(seed: int, deterministic: bool = False) -> None:
    import random
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def setup_from_config(config: Dict[str, Any]) -> None:
    log_dir = Path(get_log_dir(config))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        probe = log_dir / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except PermissionError:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        config.setdefault("paths", {})["log_dir"] = str(log_dir)
        config.setdefault("logging", {})["log_dir"] = str(log_dir)
    setup_logging("INFO", str(log_dir / "training.log"))


def preprocess_with_sources(
    items: Iterable[Tuple[str, str]],
    min_length: int,
    strip_legacy_markers: bool,
) -> Iterator[Tuple[str, str]]:
    total = 0
    kept = 0
    for text, source in items:
        total += 1
        if strip_legacy_markers:
            text = strip_legacy_end_markers(text)
        text = clean_text(text)
        if len(text) < min_length:
            continue
        kept += 1
        yield text, source
    logger.info("Mixed preprocessing complete: kept %s/%s texts", kept, total)


def load_phase_manifest(manifest_dir: str, phase_id: int) -> Dict[str, Any]:
    return read_json(str(Path(manifest_dir) / f"phase_{phase_id}_manifest.json"))


def phase_ids(config: Dict[str, Any], only_phase: Optional[int] = None) -> List[int]:
    ids = [int(phase["id"]) for phase in config.get("phases", [])]
    if only_phase is not None:
        ids = [phase_id for phase_id in ids if phase_id == int(only_phase)]
    return ids


def get_phase(config: Dict[str, Any], phase_id: int) -> Dict[str, Any]:
    for phase in config.get("phases", []):
        if int(phase["id"]) == int(phase_id):
            return phase
    raise ValueError(f"Unknown phase id: {phase_id}")


def estimate_phase_tokens(phase: Dict[str, Any], chars_per_token: float = 4.0) -> int:
    raw_gb = float(phase.get("turkish_target_gb", 0)) + float(phase.get("english_target_gb", 0))
    return int(raw_gb * 1024**3 / chars_per_token)


def phase_token_cache_dir(config: Dict[str, Any], phase_id: int) -> Path:
    return Path(config.get("paths", {}).get("cache_dir", "./cache")) / "token_cache" / f"phase_{int(phase_id)}"


def ensure_manifests(config: Dict[str, Any]) -> str:
    manifest_dir = get_manifest_dir(config)
    turkish_manifest = Path(manifest_dir) / "turkish_partitions.json"
    missing_phase = any(
        not (Path(manifest_dir) / f"phase_{int(phase['id'])}_manifest.json").exists()
        for phase in config.get("phases", [])
    )
    if not turkish_manifest.exists() or missing_phase:
        logger.info("Preparing dataset manifests in %s", manifest_dir)
        prepare_manifests(config, manifest_dir)
    return manifest_dir


def build_phase_text_stream(
    config: Dict[str, Any],
    phase: Dict[str, Any],
    phase_manifest: Dict[str, Any],
) -> Iterator[Tuple[str, str]]:
    data_cfg = config.get("datasets", {}).get("turkish", config.get("data", {}))
    min_text_length = int(data_cfg.get("min_text_length", config.get("data", {}).get("min_text_length", 50)))
    strip_legacy = bool(data_cfg.get("strip_legacy_endoftext", True))
    tr_texts = iter_turkish_texts(phase_manifest, config)
    en_texts = iter_english_texts(phase, config)
    mix_cfg = phase.get("mix", {})
    seed = int(config.get("project", {}).get("seed", config.get("seed", 42))) + int(phase["id"])
    mixed = weighted_mix_texts(
        tr_texts,
        en_texts,
        turkish_ratio=float(mix_cfg.get("turkish_ratio", 0.75)),
        seed=seed,
    )
    return preprocess_with_sources(mixed, min_length=min_text_length, strip_legacy_markers=strip_legacy)


def prepare_phase_token_cache(
    config: Dict[str, Any],
    phase: Dict[str, Any],
    phase_manifest: Dict[str, Any],
    tokenizer,
    force_rebuild: bool = False,
    batch_size_override: Optional[int] = None,
    chunks_per_shard_override: Optional[int] = None,
) -> Dict[str, Any]:
    cache_dir = phase_token_cache_dir(config, int(phase["id"]))
    seq_len = get_seq_len(config)
    token_cache_cfg = config.get("token_cache", {})
    batch_size = int(batch_size_override or token_cache_cfg.get("batch_size", 1024))
    chunks_per_shard = int(chunks_per_shard_override or token_cache_cfg.get("chunks_per_shard", 8192))
    texts = build_phase_text_stream(config, phase, phase_manifest)
    return pack_and_tokenize_to_sharded_cache(
        texts,
        tokenizer,
        seq_len=seq_len,
        cache_dir=str(cache_dir),
        batch_size=batch_size,
        chunks_per_shard=chunks_per_shard,
        resume=True,
        force_rebuild=force_rebuild,
    )


def build_phase_dataloader(
    config: Dict[str, Any],
    phase: Dict[str, Any],
    phase_manifest: Dict[str, Any],
    tokenizer,
) -> tuple[Any, int]:
    from .data import ShardedMemmapPackedDataset, create_dataloader

    cache_dir = phase_token_cache_dir(config, int(phase["id"]))
    seq_len = get_seq_len(config)

    if not token_cache_is_complete(str(cache_dir), seq_len=seq_len):
        logger.info("No complete token cache for phase %s; preparing it now", phase["id"])
        prepare_phase_token_cache(config, phase, phase_manifest, tokenizer)

    dataset = ShardedMemmapPackedDataset(str(cache_dir / "manifest.json"))
    num_chunks = len(dataset)
    if num_chunks == 0:
        raise RuntimeError(f"Phase {phase['id']} produced no token chunks")

    train_cfg = config.get("training", {})
    data_cfg = config.get("datasets", {}).get("turkish", config.get("data", {}))
    loader = create_dataloader(
        dataset,
        batch_size=int(train_cfg.get("micro_batch_size", 1)),
        shuffle=True,
        num_workers=int(train_cfg.get("dataloader_num_workers", data_cfg.get("num_workers", 2))),
        prefetch_factor=int(train_cfg.get("prefetch_factor", data_cfg.get("prefetch_factor", 2))),
    )
    return loader, num_chunks


def apply_runtime_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint_dir = get_checkpoint_dir(config)
    log_dir = get_log_dir(config)
    output_dir = str(Path(checkpoint_dir).parent)
    config.setdefault("checkpointing", {})["checkpoint_dir"] = checkpoint_dir
    config.setdefault("checkpointing", {})["output_dir"] = output_dir
    config.setdefault("logging", {})["log_dir"] = log_dir
    config.setdefault("data", {})["seq_len"] = get_seq_len(config)
    return config


def command_prepare_manifests(args: argparse.Namespace) -> int:
    config = load_yaml_config(args.config)
    setup_from_config(config)
    result = prepare_manifests(config, get_manifest_dir(config))
    logger.info("Wrote manifests: %s", result)
    return 0


def command_validate_datasets(args: argparse.Namespace) -> int:
    config = load_yaml_config(args.config)
    setup_from_config(config)
    result = prepare_manifests(config, get_manifest_dir(config))
    turkish = read_json(result["turkish_manifest"])
    logger.info("Turkish shards: %s", turkish["total_files"])
    if turkish["total_files"] == 0:
        logger.error("No Turkish shards matched the configured pattern")
        return 1
    for path in result["phase_manifests"]:
        manifest = read_json(path)
        logger.info(
            "phase=%s turkish_files=%s english=%s",
            manifest["phase"],
            len(manifest["turkish_files"]),
            manifest["english_dataset"],
        )
    return 0


def print_dry_run(config: Dict[str, Any]) -> None:
    manifest_dir = ensure_manifests(config)
    turkish_manifest = read_json(str(Path(manifest_dir) / "turkish_partitions.json"))
    train_cfg = config.get("training", {})
    seq_len = get_seq_len(config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    batch = calculate_global_batch(
        seq_len=seq_len,
        micro_batch_size=int(train_cfg.get("micro_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        world_size=world_size,
    )
    checkpoint_dir = get_checkpoint_dir(config)
    latest = read_latest_metadata(checkpoint_dir)

    print("CPT dry run")
    print(f"  mode: {get_training_mode(config)}")
    print(f"  model: {config.get('model', {}).get('base_model') or config.get('model', {}).get('name')}")
    print(f"  tokenizer: {config.get('model', {}).get('tokenizer') or config.get('tokenizer', {}).get('path')}")
    print(f"  checkpoint_dir: {checkpoint_dir}")
    print(f"  resume_metadata_exists: {bool(latest)}")
    print(f"  turkish_repo: {turkish_manifest.get('turkish_dataset_repo')}")
    print(f"  turkish_shards: {turkish_manifest.get('total_files')}")
    print(f"  global_batch_tokens: {batch['global_batch_tokens']:,}")
    print(f"  global_batch_samples: {batch['global_batch_samples']:,}")
    for phase in config.get("phases", []):
        phase_id = int(phase["id"])
        manifest = read_json(str(Path(manifest_dir) / f"phase_{phase_id}_manifest.json"))
        estimated_tokens = estimate_phase_tokens(phase)
        estimated_steps = max(1, estimated_tokens // max(1, batch["global_batch_tokens"]))
        optimizer = phase.get("optimizer", {})
        print(
            "  phase={phase} name={name} tr_files={tr_files} english={english} "
            "lr={lr} warmup={warmup} est_tokens={tokens:,} est_steps={steps:,}".format(
                phase=phase_id,
                name=phase.get("name"),
                tr_files=len(manifest.get("turkish_files", [])),
                english=phase.get("english_dataset"),
                lr=optimizer.get("learning_rate"),
                warmup=optimizer.get("warmup_ratio"),
                tokens=estimated_tokens,
                steps=estimated_steps,
            )
        )


def command_dry_run(args: argparse.Namespace) -> int:
    config = load_yaml_config(args.config)
    setup_from_config(config)
    print_dry_run(config)
    return 0


def command_prepare_token_cache(args: argparse.Namespace) -> int:
    config = apply_runtime_paths(load_yaml_config(args.config))
    setup_from_config(config)
    if args.tokenizers_parallelism is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = args.tokenizers_parallelism
    else:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    logger.info(
        "Preparing token cache; TOKENIZERS_PARALLELISM=%s RAYON_NUM_THREADS=%s",
        os.environ.get("TOKENIZERS_PARALLELISM"),
        os.environ.get("RAYON_NUM_THREADS"),
    )

    manifest_dir = ensure_manifests(config)
    model_cfg = config.get("model", {})
    tokenizer_path = (
        model_cfg.get("tokenizer")
        or config.get("tokenizer", {}).get("path")
        or model_cfg.get("base_model")
        or model_cfg.get("name")
    )
    tokenizer = load_tokenizer_for_cli(tokenizer_path)

    selected_phase_ids = phase_ids(config, args.phase)
    if not selected_phase_ids:
        raise ValueError("No phases selected")

    for phase_id in selected_phase_ids:
        phase = get_phase(config, phase_id)
        phase_manifest = load_phase_manifest(manifest_dir, phase_id)
        cache_dir = phase_token_cache_dir(config, phase_id)
        if token_cache_is_complete(str(cache_dir), seq_len=get_seq_len(config)) and not args.force:
            logger.info("Phase %s token cache already complete: %s", phase_id, cache_dir)
            continue
        manifest = prepare_phase_token_cache(
            config,
            phase,
            phase_manifest,
            tokenizer,
            force_rebuild=args.force,
            batch_size_override=args.batch_size,
            chunks_per_shard_override=args.chunks_per_shard,
        )
        logger.info(
            "Phase %s token cache ready: chunks=%s shards=%s path=%s",
            phase_id,
            manifest.get("total_chunks"),
            len(manifest.get("shards", [])),
            cache_dir,
        )
    return 0


def resolve_resume_checkpoint(config: Dict[str, Any], resume: Optional[str], phase_id: Optional[int]) -> Optional[str]:
    checkpoint_dir = get_checkpoint_dir(config)
    resume = resume if resume is not None else config.get("training", {}).get("resume")
    if not resume or resume in (False, "false", "none", "off"):
        return None
    if resume == "auto" or resume is True:
        return find_latest_checkpoint(checkpoint_dir, phase_id=phase_id)
    return str(resume)


def command_train(args: argparse.Namespace) -> int:
    import torch
    from .model import load_model
    from .train import Trainer

    config = apply_runtime_paths(load_yaml_config(args.config))
    setup_from_config(config)
    logger.info("Training mode: %s", get_training_mode(config))
    check_environment(verbose=True)
    set_seed(int(config.get("project", {}).get("seed", config.get("seed", 42))), config.get("deterministic", False))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if get_training_mode(config) != "four_phase_curriculum":
        logger.error("src.cli train currently expects training.mode=four_phase_curriculum. Use train.py for legacy single-phase runs.")
        return 2

    manifest_dir = ensure_manifests(config)
    Path(get_checkpoint_dir(config)).mkdir(parents=True, exist_ok=True)
    Path(get_log_dir(config)).mkdir(parents=True, exist_ok=True)
    save_yaml_config(config, str(Path(get_checkpoint_dir(config)).parent / "resolved_config.yaml"))

    model_cfg = config.get("model", {})
    tokenizer_path = model_cfg.get("tokenizer") or config.get("tokenizer", {}).get("path") or model_cfg.get("base_model") or model_cfg.get("name")
    tokenizer = load_tokenizer_for_cli(tokenizer_path)

    selected_phase_ids = phase_ids(config, args.phase)
    if not selected_phase_ids:
        raise ValueError("No phases selected")

    resume_checkpoint = resolve_resume_checkpoint(config, args.resume, args.phase)
    latest = read_latest_metadata(get_checkpoint_dir(config)) if resume_checkpoint else None
    if args.phase is None and latest and latest.get("phase"):
        start_phase = int(latest["phase"])
        if str(latest.get("latest_checkpoint", "")).rstrip("/").endswith("/final"):
            start_phase += 1
        selected_phase_ids = [phase_id for phase_id in selected_phase_ids if phase_id >= start_phase]

    model = None
    for phase_id in selected_phase_ids:
        phase = get_phase(config, phase_id)
        phase_manifest = load_phase_manifest(manifest_dir, phase_id)
        phase_config = apply_runtime_paths(phase_training_config(config, phase))
        train_cfg = phase_config.setdefault("training", {})
        if args.max_steps is not None:
            train_cfg["max_steps"] = args.max_steps
        if args.checkpoint_every_steps is not None:
            phase_config.setdefault("checkpointing", {})["checkpoint_every_steps"] = args.checkpoint_every_steps

        logger.info("Preparing phase %s data", phase_id)
        dataloader, num_chunks = build_phase_dataloader(phase_config, phase, phase_manifest, tokenizer)
        logger.info("Phase %s dataset: %s chunks of %s tokens", phase_id, num_chunks, get_seq_len(phase_config))

        if model is None:
            model_source = resume_checkpoint or model_cfg.get("base_model") or model_cfg.get("name")
            logger.info("Loading model source: %s", model_source)
            model = load_model(
                model_name=model_source,
                tokenizer=tokenizer,
                torch_dtype=model_cfg.get("dtype", model_cfg.get("torch_dtype", "bfloat16")),
                attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
                use_mamba_kernels=model_cfg.get("use_mamba_kernels", True),
                use_cache=model_cfg.get("use_cache", False),
            )

        phase_resume = resume_checkpoint if latest and int(latest.get("phase", phase_id)) == phase_id else None
        trainer = Trainer(
            model=model,
            train_loader=dataloader,
            config=phase_config,
            output_dir=phase_config["checkpointing"]["output_dir"],
            resume_from_checkpoint=phase_resume,
            tokenizer=tokenizer,
        )
        summary = trainer.train()
        logger.info("Phase %s complete: %s", phase_id, summary)
        resume_checkpoint = None
        latest = None

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jamba2 Turkish CPT curriculum CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_arg(subparser):
        subparser.add_argument("--config", "-c", default="configs/cpt_4phase.yaml")

    prepare = subparsers.add_parser("prepare-manifests")
    add_config_arg(prepare)
    prepare.set_defaults(func=command_prepare_manifests)

    validate = subparsers.add_parser("validate-datasets")
    add_config_arg(validate)
    validate.set_defaults(func=command_validate_datasets)

    dry_run = subparsers.add_parser("dry-run")
    add_config_arg(dry_run)
    dry_run.set_defaults(func=command_dry_run)

    token_cache = subparsers.add_parser("prepare-token-cache")
    add_config_arg(token_cache)
    token_cache.add_argument("--phase", type=int, default=None)
    token_cache.add_argument("--batch_size", type=int, default=None)
    token_cache.add_argument("--chunks_per_shard", type=int, default=None)
    token_cache.add_argument("--force", action="store_true")
    token_cache.add_argument(
        "--tokenizers_parallelism",
        choices=["true", "false"],
        default=None,
        help="Override TOKENIZERS_PARALLELISM for this command",
    )
    token_cache.set_defaults(func=command_prepare_token_cache)

    train = subparsers.add_parser("train")
    add_config_arg(train)
    train.add_argument("--phase", type=int, default=None)
    train.add_argument("--resume", default=None)
    train.add_argument("--max_steps", type=int, default=None)
    train.add_argument("--checkpoint_every_steps", type=int, default=None)
    train.set_defaults(func=command_train)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
