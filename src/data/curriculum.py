"""Dataset manifests and streaming adapters for four-phase CPT."""

import fnmatch
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from huggingface_hub import HfApi, hf_hub_download

from .jsonl_reader import read_jsonl_files

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShardInfo:
    path: str
    size: int = 0


def gb_to_bytes(value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    return int(float(value) * 1024**3)


def get_hf_token() -> Optional[str]:
    """Return the Hugging Face token from common environment variable names."""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def _matches(path: str, pattern: str) -> bool:
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern)


def list_local_shards(local_root: str, file_pattern: str) -> List[ShardInfo]:
    root = Path(local_root)
    shards = []
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            if _matches(rel, file_pattern):
                shards.append(ShardInfo(rel, path.stat().st_size))
    return sorted(shards, key=lambda item: item.path)


def list_hf_dataset_shards(repo_id: str, file_pattern: str) -> List[ShardInfo]:
    token = get_hf_token()
    api = HfApi(token=token)
    shards: List[ShardInfo] = []
    try:
        for item in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True):
            path = getattr(item, "path", "")
            if path and _matches(path, file_pattern):
                shards.append(ShardInfo(path, int(getattr(item, "size", 0) or 0)))
    except Exception as exc:
        logger.warning("Falling back to path-only HF listing for %s: %s", repo_id, exc)
        try:
            for path in api.list_repo_files(repo_id, repo_type="dataset"):
                if _matches(path, file_pattern):
                    shards.append(ShardInfo(path, 0))
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Could not list Hugging Face dataset '{repo_id}'. "
                "If the repo is private or gated, set HF_TOKEN or use datasets.turkish.local_root."
            ) from fallback_exc
    return sorted(shards, key=lambda item: item.path)


def partition_shards(shards: List[ShardInfo], total_partitions: int = 4) -> Dict[str, List[Dict[str, Any]]]:
    """Deterministically split sorted shards into contiguous size-balanced groups."""
    if total_partitions < 1:
        raise ValueError("total_partitions must be >= 1")
    if not shards:
        return {f"phase_{idx}": [] for idx in range(1, total_partitions + 1)}

    total_size = sum(item.size for item in shards)
    partitions: Dict[str, List[Dict[str, Any]]] = {}
    if total_size <= 0:
        for idx in range(total_partitions):
            start = round(idx * len(shards) / total_partitions)
            end = round((idx + 1) * len(shards) / total_partitions)
            partitions[f"phase_{idx + 1}"] = [item.__dict__ for item in shards[start:end]]
        return partitions

    target = total_size / total_partitions
    current: List[ShardInfo] = []
    current_size = 0
    phase_idx = 1
    remaining = len(shards)
    for shard in shards:
        phases_left = total_partitions - phase_idx + 1
        current.append(shard)
        current_size += shard.size
        remaining -= 1
        should_close = (
            (current_size >= target and remaining >= phases_left - 1)
            or (remaining == phases_left - 1)
        )
        if should_close and phase_idx < total_partitions:
            partitions[f"phase_{phase_idx}"] = [item.__dict__ for item in current]
            phase_idx += 1
            current = []
            current_size = 0
    partitions[f"phase_{phase_idx}"] = [item.__dict__ for item in current]
    for idx in range(phase_idx + 1, total_partitions + 1):
        partitions[f"phase_{idx}"] = []
    return partitions


def create_turkish_partitions_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    datasets = config.get("datasets", {})
    tr_cfg = datasets.get("turkish", config.get("data", {}))
    repo = tr_cfg.get("repo") or tr_cfg.get("hf_dataset_id")
    local_root = tr_cfg.get("local_root") or tr_cfg.get("dataset_dir")
    pattern = tr_cfg.get("file_pattern", "paket_*/parca_*.jsonl.gz")
    total_partitions = int(tr_cfg.get("total_partitions", 4))

    if local_root:
        shards = list_local_shards(local_root, pattern)
        source = {"type": "local", "root": local_root}
    elif repo:
        shards = list_hf_dataset_shards(repo, pattern)
        source = {"type": "hf", "repo": repo}
    else:
        raise ValueError("Turkish dataset requires either local_root/dataset_dir or repo/hf_dataset_id")

    partitions = partition_shards(shards, total_partitions)
    manifest = {
        "turkish_dataset_repo": repo,
        "source": source,
        "file_pattern": pattern,
        "total_files": len(shards),
        "total_compressed_bytes": sum(item.size for item in shards),
        "total_partitions": total_partitions,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": config.get("project", {}).get("seed", config.get("seed", 42)),
        "partitions": partitions,
    }
    return manifest


def write_json(path: str, data: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_phase_manifests(config: Dict[str, Any], turkish_manifest: Dict[str, Any], manifest_dir: str) -> List[str]:
    written = []
    phases = config.get("phases", [])
    for phase in phases:
        phase_id = int(phase["id"])
        partition_key = f"phase_{int(phase.get('turkish_partition', phase_id))}"
        phase_manifest = {
            "phase": phase_id,
            "name": phase.get("name", f"phase_{phase_id}"),
            "turkish_files": turkish_manifest["partitions"].get(partition_key, []),
            "turkish_source": turkish_manifest.get("source", {}),
            "english_dataset": phase.get("english_dataset"),
            "english_filter": phase.get("english_filter", {}),
            "target_tr_gb": phase.get("turkish_target_gb"),
            "target_en_gb": phase.get("english_target_gb"),
            "estimated_raw_gb": (
                float(phase.get("turkish_target_gb", 0)) + float(phase.get("english_target_gb", 0))
            ),
            "mix": phase.get("mix", {"turkish_ratio": 0.75, "english_ratio": 0.25}),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": config.get("project", {}).get("seed", config.get("seed", 42)),
        }
        path = str(Path(manifest_dir) / f"phase_{phase_id}_manifest.json")
        write_json(path, phase_manifest)
        written.append(path)
    return written


def prepare_manifests(config: Dict[str, Any], manifest_dir: str) -> Dict[str, Any]:
    turkish_manifest = create_turkish_partitions_manifest(config)
    turkish_path = str(Path(manifest_dir) / "turkish_partitions.json")
    write_json(turkish_path, turkish_manifest)
    phase_paths = write_phase_manifests(config, turkish_manifest, manifest_dir)
    return {"turkish_manifest": turkish_path, "phase_manifests": phase_paths}


def load_or_prepare_manifests(config: Dict[str, Any], manifest_dir: str) -> Dict[str, Any]:
    turkish_path = Path(manifest_dir) / "turkish_partitions.json"
    if not turkish_path.exists():
        prepare_manifests(config, manifest_dir)
    return read_json(str(turkish_path))


def resolve_turkish_file_paths(phase_manifest: Dict[str, Any], cache_dir: str) -> List[str]:
    source = phase_manifest.get("turkish_source", {})
    files = [item["path"] if isinstance(item, dict) else item for item in phase_manifest.get("turkish_files", [])]
    if source.get("type") == "local":
        root = Path(source["root"])
        return [str(root / file_path) for file_path in files]
    if source.get("type") == "hf":
        repo = source["repo"]
        return [
            hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename=file_path,
                cache_dir=cache_dir,
                resume_download=True,
                token=get_hf_token(),
            )
            for file_path in files
        ]
    return files


def iter_turkish_texts(phase_manifest: Dict[str, Any], config: Dict[str, Any]) -> Iterator[str]:
    tr_cfg = config.get("datasets", {}).get("turkish", config.get("data", {}))
    text_column = tr_cfg.get("text_column", tr_cfg.get("text_field", "text"))
    cache_dir = config.get("paths", {}).get("cache_dir", "./cache")
    file_paths = resolve_turkish_file_paths(phase_manifest, cache_dir)
    yield from read_jsonl_files(file_paths, text_field=text_column)


def fineweb_score_accepts(row: Dict[str, Any], score_column: str = "score", score_gte: float = 4.0) -> bool:
    try:
        return float(row.get(score_column, 0.0)) >= float(score_gte)
    except (TypeError, ValueError):
        return False


def _iter_local_english(local_path: str, text_column: str, score_column: str, score_gte: Optional[float]) -> Iterator[str]:
    for path in Path(local_path).rglob("*"):
        if path.is_file() and (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz")):
            import json
            from .jsonl_reader import open_text_maybe_gzip

            with open_text_maybe_gzip(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if score_gte is not None and not fineweb_score_accepts(row, score_column, score_gte):
                        continue
                    text = row.get(text_column)
                    if isinstance(text, str) and text.strip():
                        yield text


def iter_english_texts(phase: Dict[str, Any], config: Dict[str, Any]) -> Iterator[str]:
    datasets = config.get("datasets", {}).get("english", {})
    key = phase.get("english_dataset")
    if not key:
        return
    en_cfg = datasets.get(key, {})
    text_column = en_cfg.get("text_column", "text")
    score_column = en_cfg.get("score_column", "score")
    score_gte = phase.get("english_filter", {}).get("score_gte")
    target_bytes = gb_to_bytes(phase.get("english_target_gb"))
    seen_bytes = 0

    local_path = en_cfg.get("local_path")
    if local_path:
        iterator = _iter_local_english(local_path, text_column, score_column, score_gte)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Remote English streaming requires the optional 'datasets' package. "
                "Install requirements.txt or set datasets.english.<name>.local_path for tests."
            ) from exc

        dataset_name = en_cfg.get("repo") or en_cfg.get("name")
        subset = en_cfg.get("subset")
        split = en_cfg.get("split", "train")
        load_kwargs = {"split": split, "streaming": True}
        token = get_hf_token()
        if token:
            load_kwargs["token"] = token
        if en_cfg.get("data_dir"):
            load_kwargs["data_dir"] = en_cfg["data_dir"]
        iterator = load_dataset(dataset_name, subset, **load_kwargs)

    accepted = 0
    rejected = 0
    for row_or_text in iterator:
        if isinstance(row_or_text, str):
            text = row_or_text
        else:
            if score_gte is not None and not fineweb_score_accepts(row_or_text, score_column, score_gte):
                rejected += 1
                continue
            text = row_or_text.get(text_column)
        if not isinstance(text, str) or not text.strip():
            continue
        text_bytes = len(text.encode("utf-8"))
        if target_bytes is not None and seen_bytes + text_bytes > target_bytes:
            break
        seen_bytes += text_bytes
        accepted += 1
        yield text
    logger.info(
        "English stream complete: dataset=%s accepted=%s rejected=%s bytes=%.3fGB",
        key,
        accepted,
        rejected,
        seen_bytes / 1024**3,
    )
