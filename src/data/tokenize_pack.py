"""Tokenization and sequence packing for efficient training."""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

from tqdm import tqdm

logger = logging.getLogger(__name__)


class PackedChunk(list):
    """List-compatible packed sequence carrying its unpadded length."""

    def __init__(self, tokens, valid_length):
        super().__init__(tokens)
        self.valid_length = valid_length


def pack_and_tokenize(
    texts: Iterator[str], tokenizer, seq_len: int = 1024,
    show_progress: bool = True, max_chunks: Optional[int] = None,
) -> List[List[int]]:
    """In-memory packing preserving a masked final tail and exact chunk limit."""
    if max_chunks is not None and max_chunks < 0:
        raise ValueError("max_chunks must be nonnegative")
    if max_chunks == 0:
        return []
    chunks = []
    for chunk in pack_and_tokenize_streaming(
        tqdm(texts, desc="Tokenizing", disable=not show_progress), tokenizer, seq_len
    ):
        chunks.append(chunk)
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
    return chunks


def pack_and_tokenize_streaming(
    texts: Iterator[str], tokenizer, seq_len: int = 1024,
) -> Iterator[List[int]]:
    """Stream full sequences and a final PackedChunk with explicit valid length."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    buffer = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            continue
        buffer.extend(tokens)
        buffer.append(eos)
        while len(buffer) >= seq_len:
            yield PackedChunk(buffer[:seq_len], seq_len)
            buffer = buffer[seq_len:]
    if buffer:
        valid = len(buffer)
        yield PackedChunk(buffer + [pad] * (seq_len - valid), valid)


TextOrSource = Union[str, Tuple[str, str]]


def _split_text_source(item: TextOrSource) -> tuple[str, Optional[str]]:
    if isinstance(item, tuple):
        return item[0], item[1]
    return item, None


def _token_cache_manifest_path(cache_dir: str) -> Path:
    return Path(cache_dir) / "manifest.json"


def load_token_cache_manifest(cache_dir: str) -> Optional[dict]:
    manifest_path = _token_cache_manifest_path(cache_dir)
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_cache_shards(cache_dir: str, manifest: dict) -> None:
    import hashlib
    import numpy as np
    if manifest.get("format") != "sharded_token_cache_v2":
        raise ValueError("Legacy token cache must be rebuilt to preserve padding and integrity")
    root = Path(cache_dir).resolve()
    for shard in manifest.get("shards", []):
        n = int(shard["num_chunks"])
        expected = {
            "path": n * int(manifest["seq_len"]) * np.dtype(manifest["dtype"]).itemsize,
            "source_ids_path": n,
            "token_source_ids_path": n * int(manifest["seq_len"]),
            "valid_lengths_path": n * 4,
        }
        for key, size in expected.items():
            path = (root / shard[key]).resolve()
            if path.parent != root or not path.is_file() or path.stat().st_size != size:
                raise ValueError(f"Invalid token cache shard size/path: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != shard["sha256"][key]:
                raise ValueError(f"Token cache shard integrity failure: {path}")


def token_cache_is_complete(cache_dir: str, seq_len: Optional[int] = None,
                            cache_identity: Optional[dict] = None) -> bool:
    try:
        manifest = load_token_cache_manifest(cache_dir)
        if not manifest or not manifest.get("complete"):
            return False
        if seq_len is not None and int(manifest.get("seq_len", -1)) != int(seq_len):
            return False
        if cache_identity is not None and manifest.get("cache_identity") != cache_identity:
            return False
        _validate_cache_shards(cache_dir, manifest)
        return True
    except (ValueError, KeyError, OSError):
        return False


def _write_token_cache_manifest(cache_dir: Path, manifest: dict) -> None:
    manifest_path = cache_dir / "manifest.json"
    tmp_path = cache_dir / "manifest.json.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    tmp_path.replace(manifest_path)


def _skip_items(items: Iterator[TextOrSource], count: int) -> Iterator[TextOrSource]:
    skipped = 0
    for item in items:
        if skipped < count:
            skipped += 1
            continue
        yield item


def tokenization_fingerprint(tokenizer) -> str:
    """Content identity independent of the directory holding a saved tokenizer."""
    import hashlib
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size
    state = {"vocab_size": vocab_size, "eos": tokenizer.eos_token_id,
             "pad": tokenizer.pad_token_id}
    if hasattr(tokenizer, "get_vocab"):
        state["vocab"] = tokenizer.get_vocab()
    if hasattr(tokenizer, "backend_tokenizer"):
        # Canonicalize JSON serialization while retaining normalizer, pre-tokenizer,
        # merges, added-token properties and post-processing configuration.
        state["backend"] = json.loads(tokenizer.backend_tokenizer.to_str())
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


# Keep the natural tokenizer-name spelling available to other repository callers.
tokenizer_fingerprint = tokenization_fingerprint


def pack_and_tokenize_to_sharded_cache(
    texts: Iterator[TextOrSource], tokenizer, seq_len: int = 1024,
    cache_dir: str = "./cache/token_cache/phase_1", batch_size: int = 1024,
    chunks_per_shard: int = 8192, resume: bool = True,
    force_rebuild: bool = False, show_progress: bool = True,
    cache_identity: Optional[dict] = None, max_tokens: Optional[int] = None,
) -> dict:
    """Pack documents exactly once with durable document-boundary checkpoints.

    max_tokens counts actual tokens including document EOS, excluding padding.
    Data/order provenance belongs in cache_identity. Iterator exceptions propagate;
    only fully processed documents are checkpointed. Resuming requires the same
    deterministic input iterator and tokenizer.
    """
    import hashlib
    import numpy as np
    import shutil

    if min(seq_len, batch_size, chunks_per_shard) <= 0:
        raise ValueError("Sequence, batch and shard sizes must be positive")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    cache_path = Path(cache_dir)
    if force_rebuild and cache_path.exists():
        shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size
    dtype = np.dtype("uint16" if vocab_size <= 65536 else "uint32")
    fingerprint = tokenization_fingerprint(tokenizer)
    manifest = load_token_cache_manifest(str(cache_path))
    if manifest and not resume:
        raise ValueError("Cache exists; use force_rebuild=True or resume=True")
    if manifest:
        if (manifest.get("seq_len") != seq_len or manifest.get("dtype") != dtype.name
                or manifest.get("tokenizer_fingerprint") != fingerprint
                or manifest.get("cache_identity") != cache_identity
                or manifest.get("max_tokens") != max_tokens):
            raise ValueError("Token cache identity does not match requested tokenizer/data/budget")
        _validate_cache_shards(str(cache_path), manifest)
        if manifest.get("complete"):
            return manifest
    else:
        manifest = {"format": "sharded_token_cache_v2", "complete": False,
                    "seq_len": seq_len, "dtype": dtype.name,
                    "tokenizer_vocab_size": vocab_size, "tokenizer_fingerprint": fingerprint,
                    "cache_identity": cache_identity, "max_tokens": max_tokens,
                    "shards": [], "texts_processed": 0, "total_chunks": 0,
                    "total_tokens": 0, "source_id_to_name": {},
                    "source_chunk_counts": {}, "source_token_counts": {},
                    "pending_buffer": [], "pending_source_buffer": []}
        _write_token_cache_manifest(cache_path, manifest)
    shards = manifest["shards"]
    sources = {int(k): v for k, v in manifest["source_id_to_name"].items()}
    source_ids = {v: k for k, v in sources.items()}
    buffer = list(manifest["pending_buffer"])
    source_buffer = list(manifest["pending_source_buffer"])
    processed = manifest["texts_processed"]
    total_chunks = manifest["total_chunks"]
    total_tokens = manifest["total_tokens"]
    chunk_counts = Counter(manifest["source_chunk_counts"])
    token_counts = Counter(manifest["source_token_counts"])
    rows, row_sources, majorities, valid_lengths = [], [], [], []

    def emit(valid):
        nonlocal buffer, source_buffer, total_chunks
        counts = Counter(source_buffer[:valid])
        majority = counts.most_common(1)[0][0]
        rows.append(buffer + [pad] * (seq_len - valid))
        row_sources.append(source_buffer + [0] * (seq_len - valid))
        majorities.append(majority)
        valid_lengths.append(valid)
        chunk_counts[sources[majority]] += 1
        total_chunks += 1
        buffer, source_buffer = [], []

    def checkpoint(complete=False):
        if rows:
            idx = len(shards)
            shard = {"num_chunks": len(rows), "tokens": sum(valid_lengths), "sha256": {}}
            arrays = {"path": (f"shard_{idx:06d}.bin", rows, dtype),
                      "source_ids_path": (f"shard_{idx:06d}.source_ids.bin", majorities, np.uint8),
                      "token_source_ids_path": (f"shard_{idx:06d}.token_sources.bin", row_sources, np.uint8),
                      "valid_lengths_path": (f"shard_{idx:06d}.valid_lengths.bin", valid_lengths, np.uint32)}
            for key, (name, values, arr_dtype) in arrays.items():
                raw = np.asarray(values, dtype=arr_dtype).tobytes()
                temp = cache_path / (name + ".tmp")
                temp.write_bytes(raw)
                temp.replace(cache_path / name)
                shard[key] = name
                shard["sha256"][key] = hashlib.sha256(raw).hexdigest()
            shards.append(shard)
            rows.clear(); row_sources.clear(); majorities.clear(); valid_lengths.clear()
        manifest.update(complete=complete, texts_processed=processed, total_chunks=total_chunks,
                        total_tokens=total_tokens, stored_tokens=total_chunks * seq_len,
                        shards=shards, source_id_to_name={str(k): v for k, v in sources.items()},
                        source_chunk_counts=dict(chunk_counts), source_token_counts=dict(token_counts),
                        pending_buffer=buffer, pending_source_buffer=source_buffer)
        _write_token_cache_manifest(cache_path, manifest)

    def process_batch(items):
        nonlocal processed, total_tokens
        encoded = tokenizer([_split_text_source(x)[0] for x in items],
                            add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        if len(encoded) != len(items):
            raise ValueError("Tokenizer returned a different number of documents")
        for item, ids in zip(items, encoded):
            if max_tokens is not None and total_tokens >= max_tokens:
                break
            text, source = _split_text_source(item)
            name = source or "unknown"
            if ids:
                tokens = list(ids) + [eos]
                if any(t < 0 or t >= vocab_size for t in tokens):
                    raise ValueError("Tokenizer returned token outside its vocabulary")
                if max_tokens is not None:
                    tokens = tokens[:max_tokens - total_tokens]
                if name not in source_ids:
                    if len(source_ids) >= 255:
                        raise ValueError("Too many source types for uint8 source ids")
                    source_ids[name] = len(source_ids) + 1
                    sources[source_ids[name]] = name
                source_id = source_ids[name]
                for offset in range(0, len(tokens), seq_len):
                    # Slices also account for a partial buffer from previous documents.
                    remaining = tokens[offset:offset + seq_len]
                    while remaining:
                        take = min(seq_len - len(buffer), len(remaining))
                        buffer.extend(remaining[:take]); source_buffer.extend([source_id] * take)
                        remaining = remaining[take:]
                        if len(buffer) == seq_len:
                            emit(seq_len)
                total_tokens += len(tokens)
                token_counts[name] += len(tokens)
            processed += 1  # Empty documents still advance the input cursor.
            if len(rows) >= chunks_per_shard:
                checkpoint()

    pending = []
    chars = 0
    progress = tqdm(desc="Batch tokenizing", disable=not show_progress, initial=processed)
    try:
        for item in _skip_items(iter(texts), processed):
            if max_tokens is not None and total_tokens >= max_tokens:
                break
            pending.append(item); chars += len(_split_text_source(item)[0])
            if len(pending) >= batch_size or chars >= 30_000_000:
                process_batch(pending); progress.update(len(pending))
                pending = []; chars = 0
        if pending:
            process_batch(pending)
    except Exception:
        # Pending un-tokenized documents are replayed from the saved cursor.
        checkpoint()
        raise
    finally:
        progress.close()
    if buffer:
        emit(len(buffer))
    checkpoint(complete=True)
    return manifest


def pack_and_tokenize_to_memmap(
    texts: Iterator[TextOrSource], tokenizer, seq_len: int = 1024,
    cache_dir: str = "./output/token_cache", show_progress: bool = True,
) -> tuple:
    """Legacy single-file interface using the validated, masked v2 packer."""
    import shutil
    root = Path(cache_dir)
    manifest = pack_and_tokenize_to_sharded_cache(
        texts, tokenizer, seq_len=seq_len, cache_dir=str(root / "sharded"),
        force_rebuild=True, show_progress=show_progress,
    )
    files = {"path": "packed_tokens.bin", "source_ids_path": "source_ids.bin",
             "token_source_ids_path": "token_source_ids.bin",
             "valid_lengths_path": "valid_lengths.bin"}
    for key, name in files.items():
        with (root / name).open("wb") as out:
            for shard in manifest["shards"]:
                with (root / "sharded" / shard[key]).open("rb") as src:
                    shutil.copyfileobj(src, out)
    metadata = dict(manifest)
    metadata["num_chunks"] = manifest["total_chunks"]
    for key, name in files.items():
        metadata[key] = str((root / name).resolve())
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return str(root / files["path"]), manifest["total_chunks"]
