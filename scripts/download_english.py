#!/usr/bin/env python3
"""
download_english.py — Download English datasets for each CPT phase to local JSONL files.

Reads configs/cpt_4phase.yaml and for each phase:
  - Streams the specified HuggingFace dataset
  - Applies score filtering if configured (e.g. fineweb_edu score >= 4.5)
  - Writes texts to local JSONL shards (500MB per shard)
  - Stops when english_target_gb is reached
  - Supports resume: skips phases that already have enough data

Output structure:
    cache/english_data/
      phase_1/
        shard_000000.jsonl
        shard_000001.jsonl
        progress.json
      phase_2/
        ...

Usage:
    python3 scripts/download_english.py --config configs/cpt_4phase.yaml
    python3 scripts/download_english.py --config configs/cpt_4phase.yaml --phase 2
    python3 scripts/download_english.py --config configs/cpt_4phase.yaml --phase 2 --output-root ./cache/english_data
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_english")

# Each shard is ~500 MB of raw text
SHARD_SIZE_BYTES = 500 * 1024 * 1024


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_progress(progress_path: Path, progress: dict) -> None:
    tmp = progress_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    tmp.replace(progress_path)


def read_progress(progress_path: Path) -> Optional[dict]:
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def stream_hf_dataset(
    dataset_name: str,
    subset: Optional[str],
    split: str,
    data_dir: Optional[str],
    text_column: str,
    score_column: Optional[str],
    score_gte: Optional[float],
) -> Iterator[str]:
    """Stream texts from a HuggingFace dataset with optional score filtering."""
    from datasets import load_dataset

    load_kwargs: Dict[str, Any] = {"split": split, "streaming": True}
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        load_kwargs["token"] = token
    if data_dir:
        load_kwargs["data_dir"] = data_dir

    logger.info(
        "Loading HF dataset: name=%s subset=%s split=%s data_dir=%s",
        dataset_name, subset, split, data_dir,
    )
    logger.info("This may take a few minutes for large repos (Resolving data files)...")

    ds = load_dataset(dataset_name, subset, **load_kwargs)

    accepted = 0
    rejected = 0

    for row in ds:
        # Apply score filter if configured
        if score_gte is not None and score_column:
            try:
                score = float(row.get(score_column, 0.0))
            except (TypeError, ValueError):
                score = 0.0
            if score < score_gte:
                rejected += 1
                if rejected % 100_000 == 0:
                    logger.info(
                        "Score filter progress: accepted=%d rejected=%d (score_gte=%.1f)",
                        accepted, rejected, score_gte,
                    )
                continue

        text = row.get(text_column)
        if not isinstance(text, str) or not text.strip():
            continue

        accepted += 1
        yield text

    logger.info("Stream ended: accepted=%d rejected=%d", accepted, rejected)


def download_phase_english(
    phase: dict,
    english_datasets: dict,
    output_dir: Path,
    force: bool = False,
) -> dict:
    """Download English data for a single phase."""
    phase_id = int(phase["id"])
    english_key = phase.get("english_dataset")
    if not english_key:
        logger.info("Phase %d has no English dataset, skipping", phase_id)
        return {"phase": phase_id, "status": "no_english"}

    en_cfg = english_datasets.get(english_key, {})
    if not en_cfg:
        logger.error("Phase %d references unknown English dataset '%s'", phase_id, english_key)
        return {"phase": phase_id, "status": "error", "error": f"Unknown dataset {english_key}"}

    target_gb = float(phase.get("english_target_gb", 0))
    target_bytes = int(target_gb * 1024 ** 3)
    if target_bytes <= 0:
        logger.info("Phase %d has english_target_gb=0, skipping", phase_id)
        return {"phase": phase_id, "status": "zero_target"}

    phase_dir = output_dir / f"phase_{phase_id}"
    progress_path = phase_dir / "progress.json"

    # Check if already complete
    progress = read_progress(progress_path)
    if progress and progress.get("complete") and not force:
        logger.info(
            "Phase %d English data already complete: %.2f GB in %d shards",
            phase_id,
            progress.get("total_bytes", 0) / 1024 ** 3,
            progress.get("shards_written", 0),
        )
        return {"phase": phase_id, "status": "already_complete", **progress}

    phase_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: count existing bytes
    existing_bytes = 0
    existing_shards = 0
    existing_texts = 0
    if progress and not force:
        existing_bytes = progress.get("total_bytes", 0)
        existing_shards = progress.get("shards_written", 0)
        existing_texts = progress.get("texts_written", 0)
        if existing_bytes >= target_bytes:
            progress["complete"] = True
            write_progress(progress_path, progress)
            logger.info("Phase %d already has enough data (%.2f GB)", phase_id, existing_bytes / 1024**3)
            return {"phase": phase_id, "status": "already_complete", **progress}
        logger.info(
            "Phase %d resuming: %.2f/%.2f GB (%d shards so far)",
            phase_id, existing_bytes / 1024**3, target_gb, existing_shards,
        )

    # Dataset config
    dataset_name = en_cfg.get("repo") or en_cfg.get("name")
    subset = en_cfg.get("subset")
    split = en_cfg.get("split", "train")
    data_dir = en_cfg.get("data_dir")
    text_column = en_cfg.get("text_column", "text")
    score_column = en_cfg.get("score_column")
    score_gte = phase.get("english_filter", {}).get("score_gte")

    logger.info("=" * 60)
    logger.info("Phase %d: Downloading '%s'", phase_id, english_key)
    logger.info("  Dataset: %s", dataset_name)
    logger.info("  Score filter: %s >= %s", score_column, score_gte)
    logger.info("  Target: %.1f GB", target_gb)
    logger.info("  Output: %s", phase_dir)
    logger.info("=" * 60)

    # Stream and write
    shard_idx = existing_shards
    total_bytes = existing_bytes
    total_texts = existing_texts
    shard_buffer = []
    shard_bytes = 0
    t0 = time.time()
    last_log = t0

    def flush_shard():
        nonlocal shard_buffer, shard_bytes, shard_idx
        if not shard_buffer:
            return
        shard_path = phase_dir / f"shard_{shard_idx:06d}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as f:
            for line in shard_buffer:
                f.write(line)
                f.write("\n")
        logger.info(
            "Wrote shard %s: %.1f MB (%d texts) | Total: %.2f/%.2f GB",
            shard_path.name,
            shard_bytes / 1024**2,
            len(shard_buffer),
            total_bytes / 1024**3,
            target_gb,
        )
        shard_buffer = []
        shard_bytes = 0
        shard_idx += 1

    texts_to_skip = existing_texts if existing_bytes > 0 else 0
    skipped = 0

    try:
        for text in stream_hf_dataset(
            dataset_name=dataset_name,
            subset=subset,
            split=split,
            data_dir=data_dir,
            text_column=text_column,
            score_column=score_column,
            score_gte=score_gte,
        ):
            # Skip already-downloaded texts for resume
            if skipped < texts_to_skip:
                skipped += 1
                if skipped % 100_000 == 0:
                    logger.info("Resuming: skipped %d/%d texts...", skipped, texts_to_skip)
                continue

            text_bytes = len(text.encode("utf-8"))

            # Check if we've reached the target
            if total_bytes + text_bytes > target_bytes:
                break

            # Write as JSON line (just {"text": "..."} format)
            json_line = json.dumps({text_column: text}, ensure_ascii=False)
            shard_buffer.append(json_line)
            shard_bytes += len(json_line.encode("utf-8"))
            total_bytes += text_bytes
            total_texts += 1

            if shard_bytes >= SHARD_SIZE_BYTES:
                flush_shard()
                # Save progress after each shard
                write_progress(progress_path, {
                    "complete": False,
                    "phase": phase_id,
                    "dataset": english_key,
                    "target_gb": target_gb,
                    "total_bytes": total_bytes,
                    "texts_written": total_texts,
                    "shards_written": shard_idx,
                })

            # Periodic log
            now = time.time()
            if now - last_log > 30:
                elapsed = now - t0
                speed_mbps = (total_bytes - existing_bytes) / elapsed / 1024**2
                logger.info(
                    "Progress: %.2f/%.2f GB | %d texts | %.1f MB/s",
                    total_bytes / 1024**3, target_gb, total_texts, speed_mbps,
                )
                last_log = now

    except KeyboardInterrupt:
        logger.warning("Interrupted! Saving progress...")

    # Flush remaining
    flush_shard()

    elapsed = time.time() - t0
    is_complete = total_bytes >= target_bytes * 0.95  # 95% is close enough

    final_progress = {
        "complete": is_complete,
        "phase": phase_id,
        "dataset": english_key,
        "dataset_repo": dataset_name,
        "score_filter": score_gte,
        "text_column": text_column,
        "target_gb": target_gb,
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / 1024**3, 3),
        "texts_written": total_texts,
        "shards_written": shard_idx,
        "elapsed_seconds": round(elapsed, 1),
    }
    write_progress(progress_path, final_progress)

    status = "complete" if is_complete else "partial"
    logger.info("=" * 60)
    logger.info(
        "Phase %d %s: %.2f/%.2f GB | %d texts | %d shards | %.1f min",
        phase_id, status, total_bytes / 1024**3, target_gb,
        total_texts, shard_idx, elapsed / 60,
    )
    logger.info("=" * 60)

    return {"phase": phase_id, "status": status, **final_progress}


def main():
    parser = argparse.ArgumentParser(description="Download English datasets for CPT phases")
    parser.add_argument("--config", "-c", default="configs/cpt_4phase.yaml", help="Config YAML path")
    parser.add_argument("--phase", type=int, default=None, help="Download only this phase (default: all)")
    parser.add_argument("--output-root", default="./cache/english_data", help="Output root directory")
    parser.add_argument("--force", action="store_true", help="Re-download even if already complete")
    args = parser.parse_args()

    config = load_config(args.config)
    phases = config.get("phases", [])
    english_datasets = config.get("datasets", {}).get("english", {})
    output_root = Path(args.output_root)

    if args.phase is not None:
        phases = [p for p in phases if int(p["id"]) == args.phase]
        if not phases:
            logger.error("Phase %d not found in config", args.phase)
            sys.exit(1)

    logger.info("Config: %s", args.config)
    logger.info("Output root: %s", output_root)
    logger.info("Phases to download: %s", [int(p["id"]) for p in phases])
    logger.info("")

    results = []
    for phase in phases:
        result = download_phase_english(
            phase=phase,
            english_datasets=english_datasets,
            output_dir=output_root,
            force=args.force,
        )
        results.append(result)
        logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    for r in results:
        logger.info(
            "  Phase %d: %s (%.2f GB, %d texts)",
            r.get("phase", "?"),
            r.get("status", "unknown"),
            r.get("total_gb", r.get("total_bytes", 0) / 1024**3 if "total_bytes" in r else 0),
            r.get("texts_written", 0),
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
