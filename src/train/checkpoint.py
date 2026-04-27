"""Checkpoint metadata helpers."""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def latest_metadata_path(checkpoint_dir: str) -> Path:
    return Path(checkpoint_dir) / "latest.json"


def write_latest_metadata(checkpoint_dir: str, metadata: Dict[str, Any]) -> None:
    path = latest_metadata_path(checkpoint_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    tmp.replace(path)


def read_latest_metadata(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    path = latest_metadata_path(checkpoint_dir)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_checkpoint(checkpoint_dir: str, phase_id: Optional[int] = None) -> Optional[str]:
    metadata = read_latest_metadata(checkpoint_dir)
    if metadata and metadata.get("latest_checkpoint"):
        latest = Path(metadata["latest_checkpoint"])
        if latest.exists() and (phase_id is None or int(metadata.get("phase", phase_id)) == int(phase_id)):
            return str(latest)

    root = Path(checkpoint_dir)
    if not root.exists():
        return None
    candidates = []
    if phase_id is not None:
        search_roots = [root / f"phase_{phase_id}"]
    else:
        search_roots = [p for p in root.glob("phase_*") if p.is_dir()]
        search_roots.append(root)
    for search_root in search_roots:
        if not search_root.exists():
            continue
        final = search_root / "final"
        if final.exists():
            candidates.append(final)
        candidates.extend(search_root.glob("step_*"))
    if not candidates:
        return None
    return str(max(candidates, key=lambda path: path.stat().st_mtime))
