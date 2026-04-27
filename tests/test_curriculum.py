import gzip
import json
import subprocess
import sys
from pathlib import Path

from src.config import calculate_global_batch, load_yaml_config
from src.data.curriculum import (
    ShardInfo,
    fineweb_score_accepts,
    partition_shards,
    prepare_manifests,
)
from src.data.dataloader import ShardedMemmapPackedDataset
from src.data.jsonl_reader import read_jsonl_files
from src.data.mixing import weighted_mix_texts
from src.data.tokenize_pack import pack_and_tokenize_to_sharded_cache, token_cache_is_complete
from src.train.checkpoint import read_latest_metadata, write_latest_metadata


def write_gzip_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            if isinstance(row, str):
                f.write(row + "\n")
            else:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tiny_config(tmp_path: Path):
    tr_root = tmp_path / "tr"
    en_root = tmp_path / "en"
    for idx in range(1, 5):
        write_gzip_jsonl(
            tr_root / f"paket_{idx}" / f"parca_{idx:05d}.jsonl.gz",
            [{"text": f"Türkçe metin {idx} " * 20}],
        )
    write_gzip_jsonl(en_root / "fineweb.jsonl.gz", [{"text": "English text " * 20, "score": 4.6}])
    return {
        "project": {"seed": 42},
        "paths": {
            "manifest_dir": str(tmp_path / "manifests"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs"),
        },
        "model": {"base_model": "dummy", "tokenizer": "dummy"},
        "training": {
            "mode": "four_phase_curriculum",
            "seq_len": 16,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "resume": "auto",
        },
        "datasets": {
            "turkish": {
                "local_root": str(tr_root),
                "file_pattern": "paket_*/parca_*.jsonl.gz",
                "text_column": "text",
                "total_partitions": 4,
            },
            "english": {
                "fineweb_edu": {
                    "local_path": str(en_root),
                    "text_column": "text",
                    "score_column": "score",
                }
            },
        },
        "phases": [
            {
                "id": idx,
                "name": f"phase_{idx}",
                "turkish_partition": idx,
                "turkish_target_gb": 0.001,
                "english_dataset": "fineweb_edu",
                "english_filter": {"score_gte": 4.0},
                "english_target_gb": 0.001,
                "mix": {"turkish_ratio": 0.75, "english_ratio": 0.25},
                "optimizer": {"learning_rate": 1e-4, "warmup_ratio": 0.1},
            }
            for idx in range(1, 5)
        ],
    }


def test_read_gzip_jsonl_and_invalid_rows(tmp_path):
    path = tmp_path / "paket_1" / "parca_00001.jsonl.gz"
    write_gzip_jsonl(
        path,
        [
            {"text": "geçerli metin"},
            "{not-json",
            {"other": "missing"},
            {"text": ""},
            {"text": "ikinci geçerli metin"},
        ],
    )
    assert list(read_jsonl_files([path], text_field="text")) == ["geçerli metin", "ikinci geçerli metin"]


def test_partition_shards_balanced_by_size():
    shards = [ShardInfo(f"file_{idx}.jsonl.gz", size) for idx, size in enumerate([10, 10, 20, 20, 30, 30, 40, 40])]
    partitions = partition_shards(shards, total_partitions=4)
    assert list(partitions) == ["phase_1", "phase_2", "phase_3", "phase_4"]
    assert sum(len(value) for value in partitions.values()) == len(shards)
    sizes = [sum(item["size"] for item in value) for value in partitions.values()]
    assert max(sizes) - min(sizes) <= 40


def test_fineweb_score_filtering():
    assert fineweb_score_accepts({"score": 4.0}, score_gte=4.0)
    assert fineweb_score_accepts({"score": "4.7"}, score_gte=4.5)
    assert not fineweb_score_accepts({"score": 3.9}, score_gte=4.0)
    assert not fineweb_score_accepts({"score": "bad"}, score_gte=4.0)


def test_phase_config_and_global_batch(tmp_path):
    config_path = tmp_path / "config.yaml"
    import yaml

    yaml.safe_dump(tiny_config(tmp_path), config_path.open("w"))
    config = load_yaml_config(str(config_path))
    assert config["training"]["mode"] == "four_phase_curriculum"
    assert calculate_global_batch(1024, 1, 2048, 1)["global_batch_tokens"] == 2_097_152


def test_prepare_manifests_and_dry_run_command(tmp_path):
    config = tiny_config(tmp_path)
    manifest_dir = config["paths"]["manifest_dir"]
    result = prepare_manifests(config, manifest_dir)
    assert Path(result["turkish_manifest"]).exists()
    assert len(result["phase_manifests"]) == 4

    config_path = tmp_path / "config.yaml"
    import yaml

    yaml.safe_dump(config, config_path.open("w"))
    proc = subprocess.run(
        [sys.executable, "-m", "src.cli", "dry-run", "--config", str(config_path)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "CPT dry run" in proc.stdout
    assert "phase=1" in proc.stdout


def test_resume_metadata_read_write(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    write_latest_metadata(str(checkpoint_dir), {"latest_checkpoint": "x", "phase": 2, "step": 100})
    assert read_latest_metadata(str(checkpoint_dir))["phase"] == 2


def test_weighted_mixing_ratio_tiny_dataset():
    tr = [f"tr {idx}" for idx in range(300)]
    en = [f"en {idx}" for idx in range(100)]
    samples = list(weighted_mix_texts(tr, en, turkish_ratio=0.75, seed=7, max_items=200))
    tr_count = sum(1 for _, source in samples if source == "turkish")
    ratio = tr_count / len(samples)
    assert 0.65 <= ratio <= 0.85


class DummyTokenizer:
    eos_token_id = 9
    pad_token_id = 0
    vocab_size = 100

    def __call__(self, texts, add_special_tokens=False, padding=False, truncation=False):
        return {"input_ids": [[(ord(ch) % 20) + 1 for ch in text] for text in texts]}


def test_sharded_token_cache_reuse(tmp_path):
    cache_dir = tmp_path / "token_cache" / "phase_1"
    texts = [(f"turkish text {idx}", "turkish") for idx in range(20)]
    manifest = pack_and_tokenize_to_sharded_cache(
        iter(texts),
        DummyTokenizer(),
        seq_len=8,
        cache_dir=str(cache_dir),
        batch_size=4,
        chunks_per_shard=3,
        show_progress=False,
    )
    assert manifest["complete"] is True
    assert manifest["total_chunks"] > 0
    assert len(manifest["shards"]) > 1
    assert token_cache_is_complete(str(cache_dir), seq_len=8)

    dataset = ShardedMemmapPackedDataset(str(cache_dir / "manifest.json"))
    item = dataset[0]
    assert item["input_ids"].shape[0] == 8
    assert item["source"] == "turkish"
