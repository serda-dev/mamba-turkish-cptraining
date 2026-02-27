"""Tokenization and sequence packing for efficient training."""

import logging
from typing import Iterator, List, Optional

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


def pack_and_tokenize_to_memmap(
    texts: Iterator[str],
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
    memmap_path = str(Path(cache_dir) / "packed_tokens.npy")
    meta_path = str(Path(cache_dir) / "metadata.json")
    
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.pad_token_id or 0
        logger.warning(f"No EOS token found, using token ID {eos_token_id}")
    
    pad_token_id = tokenizer.pad_token_id or eos_token_id
    
    # Use uint16 if vocab fits (GPT-NeoX vocab=50280 fits in uint16 max=65535)
    dtype = np.uint16 if tokenizer.vocab_size < 65536 else np.uint32
    
    # Tokenize and pack — store chunks as compact numpy arrays
    # (each np.uint16 array of 1024 = 2 KB, vs ~37 KB as Python list)
    buffer = []
    chunks_list = []
    texts_processed = 0
    
    texts_iter = tqdm(texts, desc="Tokenizing", disable=not show_progress)
    
    for text in texts_iter:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if not tokens:
            continue
        
        texts_processed += 1
        buffer.extend(tokens)
        buffer.append(eos_token_id)
        
        # Extract complete chunks from buffer
        while len(buffer) >= seq_len:
            chunk = np.array(buffer[:seq_len], dtype=dtype)
            chunks_list.append(chunk)
            buffer = buffer[seq_len:]
    
    # Handle remaining buffer (pad if at least half full)
    if buffer and len(buffer) >= seq_len // 2:
        while len(buffer) < seq_len:
            buffer.append(pad_token_id)
        chunks_list.append(np.array(buffer[:seq_len], dtype=dtype))
    
    num_chunks = len(chunks_list)
    if num_chunks == 0:
        logger.error("No chunks produced from tokenization!")
        return memmap_path, 0
    
    # Stack into a single array, write to memmap, then free
    logger.info(f"Writing {num_chunks} chunks to memmap: {memmap_path}")
    all_chunks = np.stack(chunks_list)  # shape: (num_chunks, seq_len)
    del chunks_list  # free the list of small arrays
    
    mm = np.memmap(memmap_path, dtype=dtype, mode='w+',
                   shape=(num_chunks, seq_len))
    mm[:] = all_chunks
    mm.flush()
    del mm, all_chunks  # free RAM
    
    # Save metadata for later loading
    file_size_mb = num_chunks * seq_len * np.dtype(dtype).itemsize / 1e6
    with open(meta_path, 'w') as f:
        json.dump({
            "num_chunks": num_chunks,
            "seq_len": seq_len,
            "dtype": str(dtype),
            "texts_processed": texts_processed,
            "total_tokens": num_chunks * seq_len,
            "file_size_mb": round(file_size_mb, 1),
        }, f, indent=2)
    
    logger.info(
        f"Memmap packing complete: {texts_processed} texts -> {num_chunks} chunks "
        f"(seq_len={seq_len}, ~{num_chunks * seq_len:,} tokens, "
        f"file size: {file_size_mb:.1f} MB)"
    )
    
    return memmap_path, num_chunks
