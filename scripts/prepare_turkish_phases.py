#!/usr/bin/env python3
"""
prepare_turkish_phases.py — Partition Turkish dataset shards into per-phase folders.

Splits the shards of a Turkish dataset (local or from Hugging Face Hub) into
N phase-specific folders so that each CPT phase tokenizes a distinct subset.

The partition is deterministic and balanced by compressed file size using a
greedy largest-first-to-smallest-bucket algorithm.

Output structure (default 4 phases):
    <output_root>/
        turkish_partitions.json          # global manifest
        phase_1/
            phase_manifest.json
            paket_001/parca_003.jsonl.gz  # symlink or copy
            ...
        phase_2/
            ...

Usage (from HF repo):
    python3 scripts/prepare_turkish_phases.py \\
        --repo-id MRBeDev/MC4veOSCARTekrarsizBirlestirilmis \\
        --output-root /data/turkish_phases \\
        --num-phases 4 \\
        --link-mode symlink

Usage (from local directory):
    python3 scripts/prepare_turkish_phases.py \\
        --local-root /data/raw_turkish_dataset \\
        --output-root /data/turkish_phases \\
        --num-phases 4 \\
        --link-mode symlink
"""

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prepare_turkish_phases")

DEFAULT_REPO_ID = "MRBeDev/MC4veOSCARTekrarsizBirlestirilmis"
DEFAULT_FILE_PATTERN = "paket_*/parca_*.jsonl.gz"


# ── Shard discovery ───────────────────────────────────────────────────────────


def discover_shards(root: Path, file_pattern: str) -> List[Tuple[Path, int]]:
    """Find all matching shards and return (abs_path, compressed_bytes) pairs.

    Results are sorted lexicographically by relative path for determinism.
    """
    root_abs = root.absolute()
    pattern = str(root / file_pattern)
    matches = glob.glob(pattern, recursive=False)
    if not matches:
        # Fallback: recursive search for .jsonl.gz
        logger.warning(
            "No files matched pattern '%s' under '%s', trying recursive search",
            file_pattern, root,
        )
        for p in sorted(root.rglob("*.jsonl.gz")):
            matches.append(str(p))

    shards = []
    for m in matches:
        # HF snapshots store files as symlinks to ../blobs. Do not resolve here:
        # resolving would move the path outside the snapshot root and break
        # relative phase layout creation.
        p = Path(m).absolute()
        if p.is_file():
            shards.append((p, p.stat().st_size))

    # Sort by relative path from root for determinism
    shards.sort(key=lambda x: str(x[0].relative_to(root_abs)))
    return shards


# ── Partitioning ──────────────────────────────────────────────────────────────


def partition_shards(
    shards: List[Tuple[Path, int]],
    num_phases: int,
) -> List[List[Tuple[Path, int]]]:
    """Partition shards into num_phases groups balanced by compressed size.

    Uses a greedy largest-first-to-smallest-bucket algorithm:
    1. Sort shards descending by size.
    2. For each shard, assign it to the bucket with the smallest current total.

    This is deterministic given the same sorted input.
    """
    if num_phases <= 0:
        raise ValueError(f"num_phases must be > 0, got {num_phases}")
    if not shards:
        return [[] for _ in range(num_phases)]

    # Sort descending by size (stable sort, so ties break by original order)
    indexed = sorted(enumerate(shards), key=lambda x: -x[1][1])

    buckets: List[List[Tuple[Path, int]]] = [[] for _ in range(num_phases)]
    bucket_totals = [0] * num_phases

    for _orig_idx, (path, size) in indexed:
        # Assign to the bucket with the smallest total
        min_idx = bucket_totals.index(min(bucket_totals))
        buckets[min_idx].append((path, size))
        bucket_totals[min_idx] += size

    # Sort shards within each bucket by path for deterministic output
    for bucket in buckets:
        bucket.sort(key=lambda x: str(x[0]))

    return buckets


# ── HF download ──────────────────────────────────────────────────────────────


def download_from_hf(repo_id: str) -> Path:
    """Download/snapshot the dataset from Hugging Face Hub and return local path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error(
            "huggingface_hub is required for --repo-id. "
            "Install with: pip install huggingface_hub"
        )
        sys.exit(1)

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    logger.info("Downloading/verifying HF snapshot: %s", repo_id)
    logger.info("This may take a while for a large dataset on first run...")
    local_dir = snapshot_download(
        repo_id,
        repo_type="dataset",
        token=token or None,
    )
    logger.info("HF snapshot local path: %s", local_dir)
    return Path(local_dir)


# ── Phase folder creation ────────────────────────────────────────────────────


def create_phase_folders(
    root: Path,
    buckets: List[List[Tuple[Path, int]]],
    output_root: Path,
    link_mode: str,
    source_label: str,
    file_pattern: str,
    dry_run: bool = False,
) -> Dict:
    """Create phase folders with symlinks or copies and write manifests.

    Returns the global manifest dict.
    """
    created_at = datetime.now(timezone.utc).isoformat()
    global_manifest = {
        "source": source_label,
        "file_pattern": file_pattern,
        "num_phases": len(buckets),
        "link_mode": link_mode,
        "created_at": created_at,
        "phases": [],
    }

    for phase_idx, bucket in enumerate(buckets):
        phase_id = phase_idx + 1
        phase_dir = output_root / f"phase_{phase_id}"

        total_bytes = sum(size for _, size in bucket)
        phase_entry = {
            "phase_id": phase_id,
            "num_shards": len(bucket),
            "total_compressed_bytes": total_bytes,
            "shards": [],
        }

        logger.info(
            "Phase %d: %d shards, %.2f GB compressed",
            phase_id,
            len(bucket),
            total_bytes / (1024**3),
        )

        if dry_run:
            for shard_path, size in bucket:
                rel = shard_path.relative_to(root.absolute())
                phase_entry["shards"].append({
                    "original_path": str(shard_path),
                    "local_path": str(rel),
                    "compressed_bytes": size,
                })
            global_manifest["phases"].append(phase_entry)
            continue

        phase_dir.mkdir(parents=True, exist_ok=True)

        for shard_path, size in bucket:
            rel = shard_path.relative_to(root.absolute())
            dest = phase_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            if dest.exists() or dest.is_symlink():
                dest.unlink()

            if link_mode == "symlink":
                dest.symlink_to(shard_path)
            elif link_mode == "copy":
                shutil.copy2(str(shard_path), str(dest))
            else:
                raise ValueError(f"Unknown link_mode: {link_mode}")

            phase_entry["shards"].append({
                "original_path": str(shard_path),
                "local_path": str(rel),
                "compressed_bytes": size,
            })

        # Write per-phase manifest
        per_phase_manifest = {
            "source": source_label,
            "file_pattern": file_pattern,
            "created_at": created_at,
            "phase_id": phase_id,
            "num_shards": len(bucket),
            "total_compressed_bytes": total_bytes,
            "shards": phase_entry["shards"],
        }
        manifest_path = phase_dir / "phase_manifest.json"
        _write_json(manifest_path, per_phase_manifest)
        logger.info("  Wrote %s", manifest_path)

        global_manifest["phases"].append(phase_entry)

    # Write global manifest
    if not dry_run:
        global_path = output_root / "turkish_partitions.json"
        _write_json(global_path, global_manifest)
        logger.info("Wrote global manifest: %s", global_path)

    return global_manifest


def _write_json(path: Path, data: dict) -> None:
    """Atomically write JSON file."""
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Partition Turkish dataset shards into per-phase folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--repo-id",
        default=None,
        help=(
            f"Hugging Face dataset repo ID. "
            f"Default: {DEFAULT_REPO_ID}"
        ),
    )
    source.add_argument(
        "--local-root",
        default=None,
        help="Path to a local directory containing the Turkish dataset.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root directory for phase output folders.",
    )
    parser.add_argument(
        "--num-phases",
        type=int,
        default=4,
        help="Number of phases to partition into (default: 4).",
    )
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Create symlinks (default) or copies of shard files.",
    )
    parser.add_argument(
        "--file-pattern",
        default=DEFAULT_FILE_PATTERN,
        help=f"Glob pattern for shard files (default: {DEFAULT_FILE_PATTERN}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the partition plan without creating folders or links.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> Dict:
    """Main entry point. Returns the global manifest dict."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve source
    if args.local_root:
        root = Path(args.local_root).resolve()
        source_label = str(root)
        if not root.is_dir():
            logger.error("Local root does not exist or is not a directory: %s", root)
            sys.exit(1)
    else:
        repo_id = args.repo_id or DEFAULT_REPO_ID
        root = download_from_hf(repo_id).resolve()
        source_label = repo_id

    output_root = Path(args.output_root)
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    # Discover shards
    shards = discover_shards(root, args.file_pattern)
    if not shards:
        logger.error("No shards found under '%s' matching '%s'", root, args.file_pattern)
        sys.exit(1)

    total_bytes = sum(s for _, s in shards)
    logger.info(
        "Found %d shards (%.2f GB compressed) under %s",
        len(shards), total_bytes / (1024**3), root,
    )

    # Partition
    buckets = partition_shards(shards, args.num_phases)

    # Log balance
    for i, bucket in enumerate(buckets):
        bucket_bytes = sum(s for _, s in bucket)
        pct = bucket_bytes / total_bytes * 100 if total_bytes > 0 else 0
        logger.info(
            "  Phase %d: %d shards, %.2f GB (%.1f%%)",
            i + 1, len(bucket), bucket_bytes / (1024**3), pct,
        )

    if args.dry_run:
        logger.info("DRY RUN — no files or folders created.")

    # Create phase folders + manifests
    manifest = create_phase_folders(
        root=root,
        buckets=buckets,
        output_root=output_root,
        link_mode=args.link_mode,
        source_label=source_label,
        file_pattern=args.file_pattern,
        dry_run=args.dry_run,
    )

    logger.info("Done.")
    return manifest


if __name__ == "__main__":
    main()
