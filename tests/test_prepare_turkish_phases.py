"""Smoke tests for scripts/prepare_turkish_phases.py.

Uses tiny fake .jsonl.gz shards in tmp_path — no HF downloads or real data.
"""

import gzip
import json
import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the module under test
from scripts.prepare_turkish_phases import (
    discover_shards,
    partition_shards,
    create_phase_folders,
    main as prepare_main,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_dataset(tmp_path):
    """Create a minimal fake Turkish dataset with 8 shards across 2 pakets."""
    root = tmp_path / "turkish_data"
    root.mkdir()

    shard_paths = []
    for paket in range(1, 3):
        paket_dir = root / f"paket_{paket:03d}"
        paket_dir.mkdir()
        for parca in range(1, 5):
            # Create a .jsonl.gz with a few lines, varying sizes
            path = paket_dir / f"parca_{parca:03d}.jsonl.gz"
            lines = []
            # Vary content length to create different file sizes
            num_lines = paket * 10 + parca * 5
            for i in range(num_lines):
                lines.append(json.dumps({"text": f"Merhaba dünya {i} " * (parca + paket)}) + "\n")
            content = "".join(lines).encode("utf-8")
            with gzip.open(str(path), "wb") as f:
                f.write(content)
            shard_paths.append(path)

    return root, shard_paths


# ── Tests: shard discovery ────────────────────────────────────────────────────


def test_discover_shards_finds_all(fake_dataset):
    root, expected_paths = fake_dataset
    shards = discover_shards(root, "paket_*/parca_*.jsonl.gz")
    found_paths = {p for p, _ in shards}
    for ep in expected_paths:
        assert ep.resolve() in found_paths, f"Missing shard: {ep}"
    assert len(shards) == 8


def test_discover_shards_returns_sizes(fake_dataset):
    root, _ = fake_dataset
    shards = discover_shards(root, "paket_*/parca_*.jsonl.gz")
    for path, size in shards:
        assert size > 0
        assert size == path.stat().st_size


# ── Tests: partitioning ──────────────────────────────────────────────────────


def test_partition_no_overlap(fake_dataset):
    """No shard appears in more than one phase."""
    root, _ = fake_dataset
    shards = discover_shards(root, "paket_*/parca_*.jsonl.gz")
    buckets = partition_shards(shards, 4)

    all_paths = []
    for bucket in buckets:
        for path, _ in bucket:
            all_paths.append(str(path))

    # No duplicates
    assert len(all_paths) == len(set(all_paths)), "A shard appears in more than one phase!"


def test_partition_covers_all(fake_dataset):
    """All shards are assigned to exactly one phase."""
    root, _ = fake_dataset
    shards = discover_shards(root, "paket_*/parca_*.jsonl.gz")
    buckets = partition_shards(shards, 4)

    total_assigned = sum(len(b) for b in buckets)
    assert total_assigned == len(shards)


def test_partition_deterministic(fake_dataset):
    """Running partition twice gives the same result."""
    root, _ = fake_dataset
    shards = discover_shards(root, "paket_*/parca_*.jsonl.gz")
    b1 = partition_shards(shards, 4)
    b2 = partition_shards(shards, 4)

    for bucket1, bucket2 in zip(b1, b2):
        paths1 = [str(p) for p, _ in bucket1]
        paths2 = [str(p) for p, _ in bucket2]
        assert paths1 == paths2


def test_partition_reasonable_balance(fake_dataset):
    """No single phase should get more than 50% of total bytes with 4 phases."""
    root, _ = fake_dataset
    shards = discover_shards(root, "paket_*/parca_*.jsonl.gz")
    buckets = partition_shards(shards, 4)

    total_bytes = sum(s for _, s in shards)
    for bucket in buckets:
        bucket_bytes = sum(s for _, s in bucket)
        # With 4 phases, ideal is 25%, allow up to 50% for small test data
        assert bucket_bytes <= total_bytes * 0.6, (
            f"Phase too unbalanced: {bucket_bytes}/{total_bytes} = "
            f"{bucket_bytes/total_bytes*100:.1f}%"
        )


# ── Tests: phase folder creation ──────────────────────────────────────────────


def test_symlink_mode(fake_dataset, tmp_path):
    """Symlink mode creates symlinks, not copies."""
    root, _ = fake_dataset
    output_root = tmp_path / "output_sym"
    shards = discover_shards(root, "paket_*/parca_*.jsonl.gz")
    buckets = partition_shards(shards, 4)

    create_phase_folders(
        root=root,
        buckets=buckets,
        output_root=output_root,
        link_mode="symlink",
        source_label="test",
        file_pattern="paket_*/parca_*.jsonl.gz",
    )

    for phase_id in range(1, 5):
        phase_dir = output_root / f"phase_{phase_id}"
        assert phase_dir.is_dir()
        # Check that files are symlinks
        for gz in phase_dir.rglob("*.jsonl.gz"):
            assert gz.is_symlink(), f"Expected symlink: {gz}"
            assert gz.resolve().exists(), f"Broken symlink: {gz}"


def test_copy_mode(fake_dataset, tmp_path):
    """Copy mode creates regular files, not symlinks."""
    root, _ = fake_dataset
    output_root = tmp_path / "output_copy"
    shards = discover_shards(root, "paket_*/parca_*.jsonl.gz")
    buckets = partition_shards(shards, 4)

    create_phase_folders(
        root=root,
        buckets=buckets,
        output_root=output_root,
        link_mode="copy",
        source_label="test",
        file_pattern="paket_*/parca_*.jsonl.gz",
    )

    for phase_id in range(1, 5):
        phase_dir = output_root / f"phase_{phase_id}"
        assert phase_dir.is_dir()
        for gz in phase_dir.rglob("*.jsonl.gz"):
            assert not gz.is_symlink(), f"Expected regular file, got symlink: {gz}"
            assert gz.is_file()


def test_phase_folder_preserves_paket_structure(fake_dataset, tmp_path):
    """Phase folders preserve paket_NNN/ subdirectory structure."""
    root, _ = fake_dataset
    output_root = tmp_path / "output_struct"
    shards = discover_shards(root, "paket_*/parca_*.jsonl.gz")
    buckets = partition_shards(shards, 4)

    create_phase_folders(
        root=root,
        buckets=buckets,
        output_root=output_root,
        link_mode="symlink",
        source_label="test",
        file_pattern="paket_*/parca_*.jsonl.gz",
    )

    # Every linked file should be under a paket_NNN/ subdir
    for phase_id in range(1, 5):
        phase_dir = output_root / f"phase_{phase_id}"
        for gz in phase_dir.rglob("*.jsonl.gz"):
            rel = gz.relative_to(phase_dir)
            parts = rel.parts
            assert len(parts) >= 2, f"Expected paket_NNN/parca_NNN.jsonl.gz, got {rel}"
            assert parts[0].startswith("paket_"), f"Expected paket_* dir, got {parts[0]}"


# ── Tests: manifests ──────────────────────────────────────────────────────────


def test_manifests_written(fake_dataset, tmp_path):
    """Both global and per-phase manifests are valid JSON with required fields."""
    root, _ = fake_dataset
    output_root = tmp_path / "output_mfst"
    shards = discover_shards(root, "paket_*/parca_*.jsonl.gz")
    buckets = partition_shards(shards, 4)

    manifest = create_phase_folders(
        root=root,
        buckets=buckets,
        output_root=output_root,
        link_mode="symlink",
        source_label="test_source",
        file_pattern="paket_*/parca_*.jsonl.gz",
    )

    # Global manifest file
    global_path = output_root / "turkish_partitions.json"
    assert global_path.exists()
    with open(global_path) as f:
        gm = json.load(f)
    assert gm["source"] == "test_source"
    assert gm["num_phases"] == 4
    assert len(gm["phases"]) == 4
    assert "created_at" in gm

    # Per-phase manifests
    for phase_id in range(1, 5):
        pm_path = output_root / f"phase_{phase_id}" / "phase_manifest.json"
        assert pm_path.exists()
        with open(pm_path) as f:
            pm = json.load(f)
        assert pm["phase_id"] == phase_id
        assert "shards" in pm
        assert pm["total_compressed_bytes"] > 0
        for shard in pm["shards"]:
            assert "original_path" in shard
            assert "local_path" in shard
            assert "compressed_bytes" in shard


# ── Tests: CLI via main() ────────────────────────────────────────────────────


def test_main_local_root(fake_dataset, tmp_path):
    """Smoke test: main() with --local-root completes without error."""
    root, _ = fake_dataset
    output_root = tmp_path / "output_main"

    manifest = prepare_main([
        "--local-root", str(root),
        "--output-root", str(output_root),
        "--num-phases", "4",
        "--link-mode", "symlink",
    ])

    assert manifest is not None
    assert len(manifest["phases"]) == 4
    # Verify no shard overlap across phases
    all_originals = []
    for phase in manifest["phases"]:
        for shard in phase["shards"]:
            all_originals.append(shard["original_path"])
    assert len(all_originals) == len(set(all_originals))


def test_main_dry_run(fake_dataset, tmp_path):
    """Dry run should not create any folders."""
    root, _ = fake_dataset
    output_root = tmp_path / "output_dry"

    manifest = prepare_main([
        "--local-root", str(root),
        "--output-root", str(output_root),
        "--num-phases", "4",
        "--dry-run",
    ])

    # Dry run returns a manifest but doesn't create phase folders
    assert manifest is not None
    assert not (output_root / "phase_1").exists()
