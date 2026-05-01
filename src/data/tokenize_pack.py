"""Tokenization and sequence packing for efficient training."""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

from tqdm import tqdm

logger = logging.getLogger(__name__)


def pack_and_tokenize(
    texts: Iterator[str],
    tokenizer,
    seq_len: int = 1024,
    show_progress: bool = True,
    max_chunks: Optional[int] = None,
) -> List[List[int]]:
    """
    Tokenize texts and pack into fixed-length chunks.
    
    Concatenates all texts with EOS tokens, then splits into 
    seq_len chunks. This minimizes padding waste compared to 
    padding each sample individually.
    
    Args:
        texts: Iterator of text strings
        tokenizer: HuggingFace tokenizer with encode() method
        seq_len: Target sequence length for each chunk
        show_progress: Show tqdm progress bar
        max_chunks: Optional limit on chunks to create
        
    Returns:
        List of token ID lists, each of length seq_len
    """
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        # Fallback to pad token or a sentinel
        eos_token_id = tokenizer.pad_token_id or 0
        logger.warning(f"No EOS token found, using token ID {eos_token_id}")
    
    chunks = []
    buffer = []
    texts_processed = 0
    
    texts_iter = tqdm(texts, desc="Tokenizing", disable=not show_progress)
    
    for text in texts_iter:
        # Tokenize without special tokens (we add EOS manually)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if not tokens:
            continue
        
        texts_processed += 1
        
        # Add tokens + EOS to buffer
        buffer.extend(tokens)
        buffer.append(eos_token_id)
        
        # Extract complete chunks from buffer
        while len(buffer) >= seq_len:
            chunks.append(buffer[:seq_len])
            buffer = buffer[seq_len:]
            
            if max_chunks and len(chunks) >= max_chunks:
                logger.info(f"Reached max_chunks limit: {max_chunks}")
                break
        
        if max_chunks and len(chunks) >= max_chunks:
            break
    
    # Handle remaining buffer (drop if too short, as it would need padding)
    if buffer and len(buffer) >= seq_len // 2:
        # Pad the last chunk if it's at least half full
        pad_token_id = tokenizer.pad_token_id or eos_token_id
        while len(buffer) < seq_len:
            buffer.append(pad_token_id)
        chunks.append(buffer)
    
    logger.info(
        f"Packing complete: {texts_processed} texts -> {len(chunks)} chunks "
        f"(seq_len={seq_len}, ~{len(chunks) * seq_len:,} tokens)"
    )
    
    return chunks


def pack_and_tokenize_streaming(
    texts: Iterator[str],
    tokenizer,
    seq_len: int = 1024,
) -> Iterator[List[int]]:
    """
    Streaming version that yields chunks one at a time.
    Memory-efficient for very large datasets.
    """
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.pad_token_id or 0
    
    buffer = []
    
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if not tokens:
            continue
        
        buffer.extend(tokens)
        buffer.append(eos_token_id)
        
        while len(buffer) >= seq_len:
            yield buffer[:seq_len]
            buffer = buffer[seq_len:]
    
    # Optionally yield padded final chunk
    if buffer and len(buffer) >= seq_len // 2:
        pad_token_id = tokenizer.pad_token_id or eos_token_id
        while len(buffer) < seq_len:
            buffer.append(pad_token_id)
        yield buffer


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


def token_cache_is_complete(cache_dir: str, seq_len: Optional[int] = None) -> bool:
    manifest = load_token_cache_manifest(cache_dir)
    if not manifest or not manifest.get("complete"):
        return False
    if seq_len is not None and int(manifest.get("seq_len", -1)) != int(seq_len):
        return False
    cache_path = Path(cache_dir)
    for shard in manifest.get("shards", []):
        if not (cache_path / shard["path"]).exists():
            return False
        if not (cache_path / shard["source_ids_path"]).exists():
            return False
    return True


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


def pack_and_tokenize_to_sharded_cache(
    texts: Iterator[TextOrSource],
    tokenizer,
    seq_len: int = 1024,
    cache_dir: str = "./cache/token_cache/phase_1",
    batch_size: int = 1024,
    chunks_per_shard: int = 8192,
    resume: bool = True,
    force_rebuild: bool = False,
    show_progress: bool = True,
) -> dict:
    """
    Batch-tokenize text/source pairs into resumable packed-token cache shards.

    Completed shards are never rewritten during resume. The manifest stores the
    pending token buffer at shard boundaries so continuation preserves exact
    concatenate-then-pack semantics after the last completed shard.
    """
    import numpy as np
    import shutil

    cache_path = Path(cache_dir)
    if force_rebuild and cache_path.exists():
        shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.pad_token_id or 0
        logger.warning("No EOS token found, using token ID %s", eos_token_id)
    pad_token_id = tokenizer.pad_token_id or eos_token_id
    dtype = np.uint16 if tokenizer.vocab_size <= 65536 else np.uint32

    manifest = load_token_cache_manifest(str(cache_path)) if resume else None
    if manifest and manifest.get("complete"):
        logger.info("Token cache already complete: %s", cache_path)
        return manifest

    if manifest:
        if int(manifest.get("seq_len", seq_len)) != int(seq_len):
            raise ValueError(
                f"Existing token cache seq_len={manifest.get('seq_len')} does not match requested {seq_len}"
            )
        if manifest.get("dtype") != np.dtype(dtype).name:
            raise ValueError(
                f"Existing token cache dtype={manifest.get('dtype')} does not match tokenizer dtype {np.dtype(dtype).name}"
            )
        shards = manifest.get("shards", [])
        source_id_to_name = {int(k): v for k, v in manifest.get("source_id_to_name", {}).items()}
        source_name_to_id = {v: k for k, v in source_id_to_name.items()}
        buffer = list(manifest.get("pending_buffer", []))
        source_buffer = list(manifest.get("pending_source_buffer", []))
        texts_processed = int(manifest.get("texts_processed", 0))
        total_chunks = int(manifest.get("total_chunks", 0))
        source_chunk_counts = Counter(manifest.get("source_chunk_counts", {}))
        source_token_counts = Counter(manifest.get("source_token_counts", {}))
        next_shard_idx = len(shards)
        texts = _skip_items(texts, texts_processed)
        logger.info(
            "Resuming token cache: %s completed shard(s), %s chunks, %s texts processed",
            len(shards),
            total_chunks,
            texts_processed,
        )
    else:
        shards = []
        source_id_to_name = {}
        source_name_to_id = {}
        buffer = []
        source_buffer = []
        texts_processed = 0
        total_chunks = 0
        source_chunk_counts = Counter()
        source_token_counts = Counter()
        next_shard_idx = 0
        manifest = {
            "format": "sharded_token_cache_v1",
            "complete": False,
            "seq_len": seq_len,
            "dtype": np.dtype(dtype).name,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "batch_size": batch_size,
            "chunks_per_shard": chunks_per_shard,
            "texts_processed": 0,
            "total_chunks": 0,
            "total_tokens": 0,
            "shards": [],
            "source_id_to_name": {},
            "source_chunk_counts": {},
            "source_token_counts": {},
            "pending_buffer": [],
            "pending_source_buffer": [],
        }

    current_chunks = []
    current_source_ids = []

    def source_id_for(source: Optional[str]) -> int:
        source_name = source or "unknown"
        if source_name not in source_name_to_id:
            new_id = len(source_name_to_id) + 1
            if new_id > 255:
                raise ValueError("Too many source types for uint8 source ids")
            source_name_to_id[source_name] = new_id
            source_id_to_name[new_id] = source_name
        return source_name_to_id[source_name]

    def emit_current_chunk() -> None:
        nonlocal buffer, source_buffer, total_chunks
        if len(buffer) != seq_len:
            return
        current_chunks.append(np.array(buffer, dtype=dtype))
        counts = Counter(source_buffer)
        majority_source_id = counts.most_common(1)[0][0]
        current_source_ids.append(majority_source_id)
        source_name = source_id_to_name.get(majority_source_id, "unknown")
        source_chunk_counts[source_name] += 1
        for chunk_source_id, count in counts.items():
            source_token_counts[source_id_to_name.get(chunk_source_id, "unknown")] += count
        total_chunks += 1
        buffer = []
        source_buffer = []

    def flush_shard() -> None:
        nonlocal current_chunks, current_source_ids, next_shard_idx
        if not current_chunks:
            return
        shard_name = f"shard_{next_shard_idx:06d}.bin"
        source_name = f"shard_{next_shard_idx:06d}.source_ids.bin"
        shard_path = cache_path / shard_name
        source_path = cache_path / source_name
        all_chunks = np.stack(current_chunks)
        with open(shard_path, "wb") as f:
            f.write(all_chunks.tobytes())
        np.array(current_source_ids, dtype=np.uint8).tofile(source_path)
        shard = {
            "path": shard_name,
            "source_ids_path": source_name,
            "num_chunks": len(current_chunks),
            "tokens": len(current_chunks) * seq_len,
        }
        shards.append(shard)
        current_chunks = []
        current_source_ids = []
        next_shard_idx += 1
        manifest.update({
            "complete": False,
            "texts_processed": texts_processed,
            "total_chunks": total_chunks,
            "total_tokens": total_chunks * seq_len,
            "shards": shards,
            "source_id_to_name": {str(k): v for k, v in source_id_to_name.items()},
            "source_chunk_counts": dict(source_chunk_counts),
            "source_token_counts": dict(source_token_counts),
            "pending_buffer": buffer,
            "pending_source_buffer": source_buffer,
        })
        _write_token_cache_manifest(cache_path, manifest)
        logger.info(
            "Wrote token cache shard %s: chunks=%s total_chunks=%s texts_processed=%s",
            shard_name,
            shard["num_chunks"],
            total_chunks,
            texts_processed,
        )

    def append_sequence(token_ids: list[int], source_id: int) -> None:
        if not token_ids:
            return
        token_ids.append(eos_token_id)
        sequence = token_ids
        offset = 0
        while offset < len(sequence):
            remaining = seq_len - len(buffer)
            take = min(remaining, len(sequence) - offset)
            buffer.extend(sequence[offset : offset + take])
            source_buffer.extend([source_id] * take)
            offset += take
            emit_current_chunk()

    # Maximum 30 million characters per batch (~30MB raw text) to prevent RAM swap thrashing
    MAX_CHARS_PER_BATCH = 30_000_000

    def process_batch(items: list[TextOrSource]) -> None:
        nonlocal texts_processed
        if not items:
            return
        batch_texts = []
        batch_sources = []
        for item in items:
            text, source = _split_text_source(item)
            batch_texts.append(text)
            batch_sources.append(source)
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        input_ids_batch = encoded["input_ids"]
        
        # Free memory of strings and tokenizer dictionary early
        del batch_texts
        del encoded
        
        for i in range(len(input_ids_batch)):
            token_ids = input_ids_batch[i]
            source = batch_sources[i]
            
            # Release reference to allow garbage collection of this token list
            # once append_sequence is done processing it.
            input_ids_batch[i] = None
            
            if not token_ids:
                continue
            source_id = source_id_for(source)
            texts_processed += 1
            append_sequence(token_ids, source_id)
            if len(current_chunks) >= chunks_per_shard:
                flush_shard()

    import queue
    import threading

    batch_queue = queue.Queue(maxsize=3)

    def producer():
        local_batch = []
        local_chars = 0
        try:
            for item in texts:
                local_batch.append(item)
                text, _ = _split_text_source(item)
                local_chars += len(text)
                
                if len(local_batch) >= batch_size or local_chars >= MAX_CHARS_PER_BATCH:
                    batch_queue.put(local_batch)
                    local_batch = []
                    local_chars = 0
            if local_batch:
                batch_queue.put(local_batch)
        finally:
            batch_queue.put(None)

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    items_iter = tqdm(desc="Batch tokenizing", disable=not show_progress, initial=texts_processed)
    
    while True:
        batch = batch_queue.get()
        if batch is None:
            break
        process_batch(batch)
        items_iter.update(len(batch))
        
    items_iter.close()
    producer_thread.join()

    if buffer and len(buffer) >= seq_len // 2:
        while len(buffer) < seq_len:
            buffer.append(pad_token_id)
            source_buffer.append(source_buffer[-1] if source_buffer else source_id_for("unknown"))
        emit_current_chunk()

    flush_shard()
    manifest.update({
        "complete": True,
        "texts_processed": texts_processed,
        "total_chunks": total_chunks,
        "total_tokens": total_chunks * seq_len,
        "shards": shards,
        "source_id_to_name": {str(k): v for k, v in source_id_to_name.items()},
        "source_chunk_counts": dict(source_chunk_counts),
        "source_token_counts": dict(source_token_counts),
        "pending_buffer": [],
        "pending_source_buffer": [],
    })
    _write_token_cache_manifest(cache_path, manifest)
    logger.info(
        "Token cache complete: %s texts -> %s chunks (%s tokens), shards=%s",
        texts_processed,
        total_chunks,
        f"{total_chunks * seq_len:,}",
        len(shards),
    )
    return manifest


def pack_and_tokenize_to_memmap(
    texts: Iterator[TextOrSource],
    tokenizer,
    seq_len: int = 1024,
    cache_dir: str = "./output/token_cache",
    show_progress: bool = True,
) -> tuple:
    """
    Tokenize, pack, and write chunks to a numpy memmap file.
    
    Uses the same concatenate-then-chunk logic as pack_and_tokenize(),
    but stores results on disk as a memory-mapped numpy array.
    RAM usage is O(buffer) during tokenization, O(1) during training.
    
    Args:
        texts: Iterator of text strings
        tokenizer: HuggingFace tokenizer with encode() method
        seq_len: Target sequence length for each chunk
        cache_dir: Directory to write memmap and metadata files
        show_progress: Show tqdm progress bar
        
    Returns:
        Tuple of (memmap_path: str, num_chunks: int)
    """
    import json
    import numpy as np
    from pathlib import Path
    
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    memmap_path = str(Path(cache_dir) / "packed_tokens.bin")
    meta_path = str(Path(cache_dir) / "metadata.json")
    source_ids_path = str(Path(cache_dir) / "source_ids.bin")
    
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.pad_token_id or 0
        logger.warning(f"No EOS token found, using token ID {eos_token_id}")
    
    pad_token_id = tokenizer.pad_token_id or eos_token_id
    
    # Jamba2 uses vocab_size=65536, which still fits into uint16 IDs (0..65535).
    dtype = np.uint16 if tokenizer.vocab_size <= 65536 else np.uint32
    
    buffer = []
    source_buffer = []
    texts_processed = 0
    num_chunks = 0
    source_name_to_id = {}
    source_id_to_name = {}
    source_chunk_counts = Counter()
    source_token_counts = Counter()
    saw_sources = False
    
    texts_iter = tqdm(texts, desc="Tokenizing", disable=not show_progress)

    with open(memmap_path, "wb") as token_f, open(source_ids_path, "wb") as source_f:
        for item in texts_iter:
            text, source = _split_text_source(item)
            if source is not None:
                saw_sources = True
                if source not in source_name_to_id:
                    source_id = len(source_name_to_id) + 1
                    source_name_to_id[source] = source_id
                    source_id_to_name[source_id] = source
            else:
                source_id = 0

            tokens = tokenizer.encode(text, add_special_tokens=False)

            if not tokens:
                continue

            texts_processed += 1
            buffer.extend(tokens)
            buffer.append(eos_token_id)
            source_buffer.extend([source_id] * (len(tokens) + 1))

            while len(buffer) >= seq_len:
                chunk = np.array(buffer[:seq_len], dtype=dtype)
                token_f.write(chunk.tobytes())

                if saw_sources:
                    counts = Counter(source_buffer[:seq_len])
                    majority_source_id = counts.most_common(1)[0][0]
                    source_f.write(np.array([majority_source_id], dtype=np.uint8).tobytes())
                    source_name = source_id_to_name.get(majority_source_id, "unknown")
                    source_chunk_counts[source_name] += 1
                    for chunk_source_id, count in counts.items():
                        source_token_counts[source_id_to_name.get(chunk_source_id, "unknown")] += count

                num_chunks += 1
                buffer = buffer[seq_len:]
                source_buffer = source_buffer[seq_len:]

        if buffer and len(buffer) >= seq_len // 2:
            while len(buffer) < seq_len:
                buffer.append(pad_token_id)
                source_buffer.append(source_buffer[-1] if source_buffer else 0)
            chunk = np.array(buffer[:seq_len], dtype=dtype)
            token_f.write(chunk.tobytes())
            if saw_sources:
                counts = Counter(source_buffer[:seq_len])
                majority_source_id = counts.most_common(1)[0][0]
                source_f.write(np.array([majority_source_id], dtype=np.uint8).tobytes())
                source_name = source_id_to_name.get(majority_source_id, "unknown")
                source_chunk_counts[source_name] += 1
                for chunk_source_id, count in counts.items():
                    source_token_counts[source_id_to_name.get(chunk_source_id, "unknown")] += count
            num_chunks += 1

    if not saw_sources and Path(source_ids_path).exists():
        Path(source_ids_path).unlink()

    if num_chunks == 0:
        logger.error("No chunks produced from tokenization!")
        return memmap_path, 0

    file_size_mb = num_chunks * seq_len * np.dtype(dtype).itemsize / 1e6
    metadata = {
        "num_chunks": num_chunks,
        "seq_len": seq_len,
        "dtype": np.dtype(dtype).name,
        "texts_processed": texts_processed,
        "total_tokens": num_chunks * seq_len,
        "file_size_mb": round(file_size_mb, 1),
    }
    if saw_sources:
        metadata.update({
            "source_ids_path": source_ids_path,
            "source_id_to_name": {str(k): v for k, v in source_id_to_name.items()},
            "source_chunk_counts": dict(source_chunk_counts),
            "source_token_counts": dict(source_token_counts),
        })

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Memmap packing complete: {texts_processed} texts -> {num_chunks} chunks "
        f"(seq_len={seq_len}, ~{num_chunks * seq_len:,} tokens, "
        f"file size: {file_size_mb:.1f} MB)"
    )

    return memmap_path, num_chunks
