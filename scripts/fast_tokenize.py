#!/usr/bin/env python3
"""
fast_tokenize.py — Standalone, high-speed tokenization script for VPS.

Bypasses the full CPT pipeline and directly tokenizes Turkish JSONL(.gz) files
into the sharded token cache format consumed by ShardedMemmapPackedDataset.

Key design choices:
  - Each .gz is fully extracted via native `gunzip` before reading (no Python gzip).
  - Each file is processed in its own worker via multiprocessing.Pool.
  - Documents longer than MAX_TEXT_LENGTH are skipped (avoids tokenizer O(N^2) hang).
  - Workers write per-file shard bundles; the main process merges them into a
    single manifest.json that is 100% compatible with pack_and_tokenize_to_sharded_cache.
  - No HuggingFace `datasets` library dependency.

Usage:
    python3 scripts/fast_tokenize.py \\
        --input-dir /mnt/volume-nbg1-1/cache/datasets--MRBeDev--MC4veOSCARTekrarsizBirlestirilmis/snapshots/11ad3e89c89f60a3e9bfa3ac727f02d7302a16e6 \\
        --output-dir ./cache/token_cache/phase_2 \\
        --tokenizer ./customtokenizer \\
        --file-pattern "paket_*/parca_*.jsonl.gz" \\
        --seq-len 1024 \\
        --workers 6 \\
        --batch-size 4096

All arguments have sensible defaults matching your project config.
"""

import argparse
import glob
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from multiprocessing import Pool, current_process
from pathlib import Path
from typing import List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fast_tokenize")

# ── Safety limits ──────────────────────────────────────────────────────────────
MAX_TEXT_LENGTH = 1_000_000       # Skip documents longer than ~1 MB (garbage JS/CSS)
MIN_TEXT_LENGTH = 50              # Skip very short documents
MAX_CHARS_PER_BATCH = 30_000_000  # ~30 MB raw text per tokenizer call
LEGACY_MARKER = "<|endoftext|>"


# ── Text cleaning (matches src/data/preprocess.py exactly) ────────────────────
_RE_NEWLINES = re.compile(r"\n{3,}")
_RE_SPACES = re.compile(r"[^\S\n]+")


def clean_text(text: str) -> str:
    text = text.strip()
    text = _RE_NEWLINES.sub("\n\n", text)
    text = _RE_SPACES.sub(" ", text)
    return text


def strip_legacy_end_markers(text: str) -> str:
    text = text.rstrip()
    if text.endswith(LEGACY_MARKER):
        text = text[: -len(LEGACY_MARKER)].rstrip()
    return text


# ── File reading ──────────────────────────────────────────────────────────────
def iter_texts_from_jsonl(
    file_path: str,
    text_field: str = "text",
) -> "Iterator[str]":
    """Read texts from a single .jsonl or .jsonl.gz file.
    For .gz, extracts to /tmp with native gunzip first."""
    path = Path(file_path)
    tmp_path: Optional[str] = None

    if path.suffix == ".gz":
        fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", dir="/tmp")
        os.close(fd)
        logger.info("[%s] Extracting %s → %s", current_process().name, path.name, tmp_path)
        try:
            with open(tmp_path, "wb") as f_out:
                subprocess.run(["gunzip", "-c", str(path)], stdout=f_out, check=True)
        except Exception as exc:
            logger.error("[%s] gunzip failed for %s: %s", current_process().name, path.name, exc)
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            return
        read_path = tmp_path
    else:
        read_path = str(path)

    total = 0
    valid = 0
    skipped_long = 0
    skipped_short = 0
    try:
        with open(read_path, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get(text_field)
                if not isinstance(text, str) or not text.strip():
                    continue
                if len(text) > MAX_TEXT_LENGTH:
                    skipped_long += 1
                    continue
                text = strip_legacy_end_markers(text)
                text = clean_text(text)
                if len(text) < MIN_TEXT_LENGTH:
                    skipped_short += 1
                    continue
                valid += 1
                yield text
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        logger.info(
            "[%s] %s: total=%d valid=%d skipped_long=%d skipped_short=%d",
            current_process().name,
            path.name,
            total,
            valid,
            skipped_long,
            skipped_short,
        )


# ── Worker function ───────────────────────────────────────────────────────────

def _init_worker(tokenizer_path: str, seq_len: int, batch_size: int):
    """Initialise per-worker globals (tokenizer is not picklable)."""
    global _tokenizer, _seq_len, _batch_size
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _seq_len = seq_len
    _batch_size = batch_size
    logger.info("[%s] Tokenizer loaded (vocab=%d)", current_process().name, _tokenizer.vocab_size)


def process_single_file(args: tuple) -> dict:
    """
    Worker entry point: tokenize one JSONL(.gz) file and write shards.

    Returns a dict with shard metadata for the main process to merge.
    """
    file_path, output_dir, shard_offset, text_field = args
    global _tokenizer, _seq_len, _batch_size

    tokenizer = _tokenizer
    seq_len = _seq_len
    batch_size = _batch_size
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id or eos_token_id
    dtype = np.uint16 if tokenizer.vocab_size <= 65536 else np.uint32

    buffer: list[int] = []
    chunks: list[np.ndarray] = []
    shards_written: list[dict] = []
    texts_processed = 0
    total_chunks = 0
    shard_idx = shard_offset
    chunks_per_shard = 8192

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def emit_chunk():
        nonlocal buffer
        if len(buffer) >= seq_len:
            chunks.append(np.array(buffer[:seq_len], dtype=dtype))
            buffer = buffer[seq_len:]

    def flush_shard():
        nonlocal chunks, shard_idx, total_chunks
        if not chunks:
            return
        shard_name = f"shard_{shard_idx:06d}.bin"
        source_name = f"shard_{shard_idx:06d}.source_ids.bin"
        shard_path = out_dir / shard_name
        source_path = out_dir / source_name
        all_chunks = np.stack(chunks)
        with open(shard_path, "wb") as f:
            f.write(all_chunks.tobytes())
        # All source ids are "turkish" = 1
        np.array([1] * len(chunks), dtype=np.uint8).tofile(source_path)
        shard = {
            "path": shard_name,
            "source_ids_path": source_name,
            "num_chunks": len(chunks),
            "tokens": len(chunks) * seq_len,
        }
        shards_written.append(shard)
        total_chunks += len(chunks)
        shard_idx += 1
        chunks = []

    # ── Batch tokenization loop ───────────────────────────────────────────
    batch_texts: list[str] = []
    batch_chars = 0

    def process_batch():
        nonlocal texts_processed, buffer, batch_texts, batch_chars
        if not batch_texts:
            return
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        input_ids_list = encoded["input_ids"]
        del batch_texts
        del encoded
        batch_texts = []
        batch_chars = 0

        for i in range(len(input_ids_list)):
            token_ids = input_ids_list[i]
            input_ids_list[i] = None
            if not token_ids:
                continue
            texts_processed += 1
            # In-place append EOS
            token_ids.append(eos_token_id)
            buffer.extend(token_ids)
            while len(buffer) >= seq_len:
                emit_chunk()
            if len(chunks) >= chunks_per_shard:
                flush_shard()

    t0 = time.time()
    for text in iter_texts_from_jsonl(file_path, text_field=text_field):
        batch_texts.append(text)
        batch_chars += len(text)
        if len(batch_texts) >= batch_size or batch_chars >= MAX_CHARS_PER_BATCH:
            process_batch()

    process_batch()

    # Pad and flush the final partial chunk
    if buffer and len(buffer) >= seq_len // 2:
        while len(buffer) < seq_len:
            buffer.append(pad_token_id)
        emit_chunk()
    flush_shard()

    elapsed = time.time() - t0
    logger.info(
        "[%s] Done %s: %d texts → %d chunks (%d shards) in %.1fs",
        current_process().name,
        Path(file_path).name,
        texts_processed,
        total_chunks,
        len(shards_written),
        elapsed,
    )
    return {
        "file": file_path,
        "texts_processed": texts_processed,
        "total_chunks": total_chunks,
        "shards": shards_written,
        "elapsed": elapsed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def find_files(input_dir: str, file_pattern: str) -> List[str]:
    """Find all files matching the pattern inside input_dir."""
    pattern = str(Path(input_dir) / file_pattern)
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        # Try recursive rglob as fallback
        root = Path(input_dir)
        for p in sorted(root.rglob("*")):
            if p.is_file() and (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz")):
                files.append(str(p))
    return files


def main():
    parser = argparse.ArgumentParser(description="Fast standalone tokenizer for CPT data")
    parser.add_argument("--input-dir", required=True, help="Root dir with JSONL(.gz) files")
    parser.add_argument("--output-dir", required=True, help="Output dir for token cache shards")
    parser.add_argument("--tokenizer", default="./customtokenizer", help="Tokenizer path")
    parser.add_argument("--file-pattern", default="paket_*/parca_*.jsonl.gz")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel workers")
    parser.add_argument("--shard-offset", type=int, default=0, help="Starting shard index")
    parser.add_argument("--limit-files", type=int, default=None, help="Process only first N files (for testing)")
    args = parser.parse_args()

    files = find_files(args.input_dir, args.file_pattern)
    if not files:
        logger.error("No files found matching '%s' in '%s'", args.file_pattern, args.input_dir)
        sys.exit(1)

    if args.limit_files:
        files = files[: args.limit_files]

    logger.info("Found %d files to process", len(files))
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Workers: %d | batch_size: %d | seq_len: %d", args.workers, args.batch_size, args.seq_len)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-assign shard index ranges to each file so there are no collisions
    ESTIMATED_SHARDS_PER_FILE = 200  # generous upper bound
    tasks = []
    for i, f in enumerate(files):
        shard_base = args.shard_offset + i * ESTIMATED_SHARDS_PER_FILE
        tasks.append((f, args.output_dir, shard_base, args.text_field))

    t_start = time.time()

    all_results = []
    with Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.tokenizer, args.seq_len, args.batch_size),
    ) as pool:
        for result in pool.imap_unordered(process_single_file, tasks):
            all_results.append(result)
            logger.info(
                "Progress: %d/%d files done (%d chunks so far)",
                len(all_results),
                len(files),
                sum(r["total_chunks"] for r in all_results),
            )

    # ── Merge results into a unified manifest.json ────────────────────────
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = tok.vocab_size
    dtype = np.uint16 if vocab_size <= 65536 else np.uint32

    all_shards = []
    total_texts = 0
    total_chunks = 0
    for r in sorted(all_results, key=lambda x: x["file"]):
        all_shards.extend(r["shards"])
        total_texts += r["texts_processed"]
        total_chunks += r["total_chunks"]

    manifest = {
        "format": "sharded_token_cache_v1",
        "complete": True,
        "seq_len": args.seq_len,
        "dtype": np.dtype(dtype).name,
        "tokenizer_vocab_size": vocab_size,
        "batch_size": args.batch_size,
        "chunks_per_shard": 8192,
        "texts_processed": total_texts,
        "total_chunks": total_chunks,
        "total_tokens": total_chunks * args.seq_len,
        "shards": all_shards,
        "source_id_to_name": {"1": "turkish"},
        "source_chunk_counts": {"turkish": total_chunks},
        "source_token_counts": {"turkish": total_chunks * args.seq_len},
        "pending_buffer": [],
        "pending_source_buffer": [],
    }

    manifest_path = out_dir / "manifest.json"
    tmp_manifest = out_dir / "manifest.json.tmp"
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    tmp_manifest.replace(manifest_path)

    elapsed_total = time.time() - t_start
    logger.info("=" * 60)
    logger.info("ALL DONE")
    logger.info("  Files processed: %d", len(files))
    logger.info("  Texts processed: %d", total_texts)
    logger.info("  Total chunks:    %d", total_chunks)
    logger.info("  Total tokens:    %s", f"{total_chunks * args.seq_len:,}")
    logger.info("  Total shards:    %d", len(all_shards))
    logger.info("  Elapsed:         %.1fs (%.1f min)", elapsed_total, elapsed_total / 60)
    logger.info("  Manifest:        %s", manifest_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
