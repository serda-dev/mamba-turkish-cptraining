#!/usr/bin/env python3
"""Merge a local PEFT LoRA adapter into a local base model folder.

This script is designed for local Hugging Face folders and supports:
- validating base model vs adapter-only checkpoints,
- detecting practical dtype from config + safetensors,
- merging via PEFT merge_and_unload,
- saving a standalone merged model + tokenizer.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def _has_model_weights(folder: Path) -> bool:
    return bool(list(folder.glob("model*.safetensors"))) or (folder / "pytorch_model.bin").exists()


def _is_base_model_folder(folder: Path) -> bool:
    return (folder / "config.json").exists() and _has_model_weights(folder) and not (folder / "adapter_model.safetensors").exists()


def _is_adapter_folder(folder: Path) -> bool:
    return (folder / "adapter_config.json").exists() and (folder / "adapter_model.safetensors").exists()


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_dtype_from_str(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if not dtype_str:
        return None
    name = str(dtype_str).strip().lower()
    if name.startswith("torch."):
        name = name.split(".", 1)[1]
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(name)


def _iter_safetensor_files(base_model_path: Path) -> List[Path]:
    shards = sorted(base_model_path.glob("model-*.safetensors"))
    if shards:
        return shards
    one_piece = base_model_path / "model.safetensors"
    if one_piece.exists():
        return [one_piece]
    return []


def _inspect_safetensor_dtypes(files: Iterable[Path]) -> Dict[str, int]:
    dtype_counts: Dict[str, int] = {}
    for file_path in files:
        with safe_open(file_path, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                dt = str(sf.get_tensor(key).dtype)
                dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
    return dtype_counts


def detect_merge_dtype(base_model_path: Path) -> Tuple[torch.dtype, str, Dict[str, int]]:
    """Detect a practical merge dtype.

    Priority:
    1) safetensors tensor dtype distribution (most reliable),
    2) config dtype fields,
    3) safe fallback.
    """
    config = _read_json(base_model_path / "config.json")
    config_dtype = _parse_dtype_from_str(config.get("torch_dtype") or config.get("dtype"))

    safetensor_files = _iter_safetensor_files(base_model_path)
    tensor_dtype_counts = _inspect_safetensor_dtypes(safetensor_files) if safetensor_files else {}

    resolved_dtype: Optional[torch.dtype] = None
    reason = ""

    if tensor_dtype_counts:
        sorted_counts = sorted(tensor_dtype_counts.items(), key=lambda x: x[1], reverse=True)
        top_dtype_name, top_count = sorted_counts[0]
        if len(sorted_counts) == 1:
            resolved_dtype = _parse_dtype_from_str(top_dtype_name)
            reason = f"All base safetensors tensors are {top_dtype_name} ({top_count} tensors)."
        else:
            resolved_dtype = _parse_dtype_from_str(top_dtype_name)
            reason = (
                "Base safetensors have mixed dtypes; selecting the dominant dtype "
                f"{top_dtype_name} ({top_count}/{sum(tensor_dtype_counts.values())})."
            )

    if resolved_dtype is None and config_dtype is not None:
        resolved_dtype = config_dtype
        reason = "Could not infer tensor dtype from safetensors; using config dtype."

    if resolved_dtype is None:
        # Safe fallback: bf16 is preferred in this project and avoids forcing fp32 memory growth.
        resolved_dtype = torch.bfloat16
        reason = (
            "Could not reliably detect dtype from config or tensors; "
            "falling back to torch.bfloat16 as a safe default for this Jamba workflow."
        )

    return resolved_dtype, reason, tensor_dtype_counts


def discover_paths(output_root: Path, name_hint: str = "jamba2-3b") -> Tuple[Path, Path]:
    candidates = [p for p in output_root.iterdir() if p.is_dir()]
    base_candidates = [p for p in candidates if _is_base_model_folder(p)]
    adapter_candidates = [p for p in candidates if _is_adapter_folder(p)]

    hint = name_hint.lower().strip()
    if hint:
        hinted_base = [p for p in base_candidates if hint in p.name.lower()]
        hinted_adapter = [p for p in adapter_candidates if hint in p.name.lower()]
        if hinted_base:
            base_candidates = hinted_base
        if hinted_adapter:
            adapter_candidates = hinted_adapter

    if len(base_candidates) != 1:
        raise RuntimeError(
            "Could not uniquely resolve base model folder. "
            f"Found: {[str(p) for p in base_candidates]}"
        )
    if len(adapter_candidates) != 1:
        raise RuntimeError(
            "Could not uniquely resolve adapter folder. "
            f"Found: {[str(p) for p in adapter_candidates]}"
        )

    return base_candidates[0], adapter_candidates[0]


def validate_base_model(base_model_path: Path) -> Dict:
    if not base_model_path.exists() or not base_model_path.is_dir():
        raise FileNotFoundError(f"Base model path not found: {base_model_path}")
    if not _is_base_model_folder(base_model_path):
        raise RuntimeError(
            "Base model folder does not look like a full HF model checkpoint "
            f"(missing config/model files or it is adapter-only): {base_model_path}"
        )

    config = _read_json(base_model_path / "config.json")
    quantization_cfg = config.get("quantization_config")
    if quantization_cfg:
        raise RuntimeError(
            "Base model config contains quantization_config. "
            "This script requires a full-precision (non-4bit) base for merge."
        )

    return config


def validate_adapter(adapter_path: Path) -> Dict:
    if not adapter_path.exists() or not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    if not _is_adapter_folder(adapter_path):
        raise RuntimeError(
            "Adapter folder does not look like a PEFT adapter checkpoint "
            f"(missing adapter files): {adapter_path}"
        )

    adapter_config = _read_json(adapter_path / "adapter_config.json")
    peft_type = str(adapter_config.get("peft_type", "")).upper()
    if peft_type != "LORA":
        raise RuntimeError(f"Unsupported adapter peft_type: {peft_type!r}. Expected 'LORA'.")

    return adapter_config


def choose_tokenizer_source(base_model_path: Path, adapter_path: Path) -> Path:
    adapter_has_tokenizer = (adapter_path / "tokenizer.json").exists() or (adapter_path / "tokenizer_config.json").exists()
    return adapter_path if adapter_has_tokenizer else base_model_path


def ensure_output_ready(output_path: Path, force: bool) -> None:
    if output_path.exists() and any(output_path.iterdir()):
        if _is_base_model_folder(output_path) and not force:
            print(f"[INFO] Output already contains a standalone model, skipping merge: {output_path}")
            raise SystemExit(0)
        if not force:
            raise RuntimeError(
                f"Output path exists and is not empty: {output_path}. "
                "Use --force to overwrite."
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


def maybe_copy_chat_template(tokenizer_src: Path, output_path: Path) -> None:
    src = tokenizer_src / "chat_template.jinja"
    dst = output_path / "chat_template.jinja"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)


def run_merge(
    base_model_path: Path,
    adapter_path: Path,
    output_path: Path,
    device_map: str,
    force: bool,
    max_shard_size: str,
) -> None:
    _ = validate_base_model(base_model_path)
    adapter_config = validate_adapter(adapter_path)

    ensure_output_ready(output_path, force=force)

    merge_dtype, dtype_reason, tensor_dtype_counts = detect_merge_dtype(base_model_path)

    tokenizer_source = choose_tokenizer_source(base_model_path, adapter_path)

    print("[INFO] Base model:", base_model_path)
    print("[INFO] Adapter:", adapter_path)
    print("[INFO] Output:", output_path)
    print("[INFO] Tokenizer source:", tokenizer_source)
    print("[INFO] Tensor dtype distribution:", tensor_dtype_counts if tensor_dtype_counts else "N/A")
    print("[INFO] Selected merge dtype:", merge_dtype)
    print("[INFO] Dtype reason:", dtype_reason)

    declared_base = adapter_config.get("base_model_name_or_path")
    if declared_base:
        print("[INFO] Adapter declares base_model_name_or_path:", declared_base)

    effective_device_map: Optional[str] = device_map
    if device_map == "auto" and not torch.cuda.is_available():
        effective_device_map = None

    model_kwargs = {
        "dtype": merge_dtype,
        "low_cpu_mem_usage": True,
    }
    if effective_device_map is not None:
        model_kwargs["device_map"] = effective_device_map

    print("[INFO] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(str(base_model_path), **model_kwargs)

    print("[INFO] Loading adapter onto base...")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)

    print("[INFO] Merging adapter into base weights...")
    merged_model = peft_model.merge_and_unload(progressbar=True)

    print("[INFO] Saving merged model...")
    merged_model.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )

    print("[INFO] Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source))
    tokenizer.save_pretrained(str(output_path))
    maybe_copy_chat_template(tokenizer_source, output_path)

    # Persist a minimal merge report for reproducibility.
    report = {
        "base_model_path": str(base_model_path),
        "adapter_path": str(adapter_path),
        "output_path": str(output_path),
        "selected_dtype": str(merge_dtype),
        "dtype_reason": dtype_reason,
        "tensor_dtype_counts": tensor_dtype_counts,
        "tokenizer_source": str(tokenizer_source),
        "adapter_base_model_name_or_path": declared_base,
    }
    with (output_path / "merge_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[OK] Merge completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge local LoRA adapter into local base model.")
    parser.add_argument("--base_model_path", type=str, default=None, help="Local HF base model folder")
    parser.add_argument("--adapter_path", type=str, default=None, help="Local PEFT adapter folder")
    parser.add_argument("--output_path", type=str, default=None, help="Output folder for merged standalone model")
    parser.add_argument("--output_root", type=str, default="output", help="Root folder used for auto-discovery")
    parser.add_argument("--name_hint", type=str, default="jamba2-3b", help="Name hint used for auto-discovery")
    parser.add_argument("--device_map", type=str, default="auto", help="device_map for from_pretrained (e.g., auto, cpu)")
    parser.add_argument("--max_shard_size", type=str, default="5GB", help="Shard size when saving merged model")
    parser.add_argument("--force", action="store_true", help="Overwrite output path if it already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root).resolve()

    if args.base_model_path and args.adapter_path:
        base_model_path = Path(args.base_model_path).resolve()
        adapter_path = Path(args.adapter_path).resolve()
    else:
        base_model_path, adapter_path = discover_paths(output_root=output_root, name_hint=args.name_hint)

    if args.output_path:
        output_path = Path(args.output_path).resolve()
    else:
        output_path = (output_root / f"{base_model_path.name}-merged").resolve()

    run_merge(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        output_path=output_path,
        device_map=args.device_map,
        force=args.force,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
