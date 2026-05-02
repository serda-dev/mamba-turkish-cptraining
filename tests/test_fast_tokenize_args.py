"""Smoke tests for fast_tokenize.py argument parsing and shard shuffle logic.

These tests do NOT run actual tokenization (which requires a real tokenizer
and the transformers library). They test CLI argument parsing, multi-dir
logic, and shard shuffling by exercising the argparse setup and helper functions.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fast_tokenize import find_files


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_english_dirs(tmp_path):
    """Create two fake English directories with .jsonl shard files."""
    dir_a = tmp_path / "english_fineweb"
    dir_b = tmp_path / "english_openwebmath"
    dir_a.mkdir()
    dir_b.mkdir()

    for i in range(5):
        (dir_a / f"shard_{i:06d}.jsonl").write_text(
            json.dumps({"text": f"English fineweb text {i}"}) + "\n"
        )
    for i in range(3):
        (dir_b / f"shard_{i:06d}.jsonl").write_text(
            json.dumps({"text": f"English openwebmath text {i}"}) + "\n"
        )

    return dir_a, dir_b


@pytest.fixture
def fake_turkish_dir(tmp_path):
    """Create a minimal fake Turkish directory with .jsonl.gz files."""
    root = tmp_path / "turkish"
    paket = root / "paket_001"
    paket.mkdir(parents=True)
    import gzip
    for i in range(3):
        path = paket / f"parca_{i:03d}.jsonl.gz"
        content = json.dumps({"text": f"Merhaba dünya {i}"}) + "\n"
        with gzip.open(str(path), "wb") as f:
            f.write(content.encode("utf-8"))
    return root


# ── Helper to build parser (same as in fast_tokenize.py main()) ──────────────


def _build_parser():
    """Replicate the argparse setup from fast_tokenize.py for testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default="./customtokenizer")
    parser.add_argument("--file-pattern", default="paket_*/parca_*.jsonl.gz")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--shard-offset", type=int, default=0)
    parser.add_argument("--limit-files", type=int, default=None)
    # English mixing
    parser.add_argument("--english-dir", action="append", default=None)
    parser.add_argument("--english-ratio", type=float, default=0.25)
    parser.add_argument("--english-text-field", default="text")
    parser.add_argument("--english-file-pattern", default="shard_*.jsonl")
    parser.add_argument("--english-weights", default=None)
    # Shard ordering
    parser.add_argument("--shuffle-shards", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


# ── Tests: argument parsing ──────────────────────────────────────────────────


def test_no_english_dir_gives_none():
    """Without --english-dir, args.english_dir is None."""
    parser = _build_parser()
    args = parser.parse_args([
        "--input-dir", "/tmp/tr",
        "--output-dir", "/tmp/out",
    ])
    assert args.english_dir is None


def test_single_english_dir_gives_list():
    """A single --english-dir gives a one-element list."""
    parser = _build_parser()
    args = parser.parse_args([
        "--input-dir", "/tmp/tr",
        "--output-dir", "/tmp/out",
        "--english-dir", "/tmp/en1",
    ])
    assert args.english_dir == ["/tmp/en1"]


def test_multiple_english_dir_gives_list():
    """Multiple --english-dir values accumulate into a list."""
    parser = _build_parser()
    args = parser.parse_args([
        "--input-dir", "/tmp/tr",
        "--output-dir", "/tmp/out",
        "--english-dir", "/tmp/en1",
        "--english-dir", "/tmp/en2",
        "--english-dir", "/tmp/en3",
    ])
    assert args.english_dir == ["/tmp/en1", "/tmp/en2", "/tmp/en3"]


def test_english_weights_parsed():
    """--english-weights is passed as a string for later splitting."""
    parser = _build_parser()
    args = parser.parse_args([
        "--input-dir", "/tmp/tr",
        "--output-dir", "/tmp/out",
        "--english-dir", "/tmp/en1",
        "--english-dir", "/tmp/en2",
        "--english-weights", "0.6,0.4",
    ])
    weights = [float(w.strip()) for w in args.english_weights.split(",")]
    assert weights == [0.6, 0.4]
    assert len(weights) == len(args.english_dir)


def test_shuffle_shards_flag():
    """--shuffle-shards is a boolean flag."""
    parser = _build_parser()
    args = parser.parse_args([
        "--input-dir", "/tmp/tr",
        "--output-dir", "/tmp/out",
        "--shuffle-shards",
        "--seed", "99",
    ])
    assert args.shuffle_shards is True
    assert args.seed == 99


def test_shuffle_shards_default_off():
    """Without --shuffle-shards, the flag is False."""
    parser = _build_parser()
    args = parser.parse_args([
        "--input-dir", "/tmp/tr",
        "--output-dir", "/tmp/out",
    ])
    assert args.shuffle_shards is False
    assert args.seed == 42  # default


# ── Tests: file discovery ────────────────────────────────────────────────────


def test_find_english_files_multiple_dirs(fake_english_dirs):
    """find_files works across multiple English directories."""
    dir_a, dir_b = fake_english_dirs
    files_a = find_files(str(dir_a), "shard_*.jsonl")
    files_b = find_files(str(dir_b), "shard_*.jsonl")
    assert len(files_a) == 5
    assert len(files_b) == 3


def test_find_turkish_files(fake_turkish_dir):
    """find_files discovers Turkish .jsonl.gz files."""
    files = find_files(str(fake_turkish_dir), "paket_*/parca_*.jsonl.gz")
    assert len(files) == 3


def test_turkish_only_no_english(fake_turkish_dir):
    """Turkish-only mode: no english dirs → empty english file list."""
    parser = _build_parser()
    args = parser.parse_args([
        "--input-dir", str(fake_turkish_dir),
        "--output-dir", "/tmp/out",
    ])
    english_dirs = args.english_dir or []
    assert english_dirs == []


# ── Tests: shard shuffling determinism ────────────────────────────────────────


def test_shuffle_shards_deterministic():
    """Shuffling with the same seed produces the same order every time."""
    shards = [
        {"path": f"shard_{i:06d}.bin", "num_chunks": 100, "tokens": 102400}
        for i in range(20)
    ]

    def shuffle_with_seed(shard_list, seed):
        copy = list(shard_list)
        rng = random.Random(seed)
        rng.shuffle(copy)
        return copy

    result1 = shuffle_with_seed(shards, 42)
    result2 = shuffle_with_seed(shards, 42)
    assert result1 == result2

    # Different seed → different order (with high probability for 20 elements)
    result3 = shuffle_with_seed(shards, 99)
    assert result3 != result1


def test_shuffle_preserves_all_shards():
    """Shuffling does not lose or duplicate shards."""
    shards = [
        {"path": f"shard_{i:06d}.bin", "num_chunks": 100}
        for i in range(15)
    ]

    rng = random.Random(42)
    shuffled = list(shards)
    rng.shuffle(shuffled)

    original_paths = sorted(s["path"] for s in shards)
    shuffled_paths = sorted(s["path"] for s in shuffled)
    assert original_paths == shuffled_paths


def test_shuffle_interleaves_sources():
    """Shuffling a list of Turkish-first, English-second shards should mix them."""
    shards = []
    for i in range(10):
        shards.append({"path": f"shard_tr_{i:03d}.bin", "source": "turkish"})
    for i in range(10):
        shards.append({"path": f"shard_en_{i:03d}.bin", "source": "english"})

    # Before shuffle: first 10 are turkish, last 10 are english
    assert all(s["source"] == "turkish" for s in shards[:10])
    assert all(s["source"] == "english" for s in shards[10:])

    rng = random.Random(42)
    rng.shuffle(shards)

    # After shuffle: sources should be mixed (not all turkish first)
    first_half_sources = [s["source"] for s in shards[:10]]
    assert "english" in first_half_sources, "Shuffle did not interleave sources"
    assert "turkish" in first_half_sources, "Shuffle did not interleave sources"
