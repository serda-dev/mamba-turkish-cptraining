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
