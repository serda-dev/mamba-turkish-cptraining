"""PyTorch Dataset and DataLoader for packed sequences."""

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class PackedDataset(Dataset):
    """
    Dataset for pre-packed token sequences (in-RAM).
    
    Each item is a dict with:
      - input_ids: token IDs [seq_len]
      - labels: same as input_ids (for causal LM)
      - attention_mask: all ones [seq_len]
    """
    
    def __init__(self, chunks: List[List[int]]):
        """
        Args:
            chunks: List of token ID lists, each of same length
        """
        self.chunks = chunks
        if chunks:
            self.seq_len = len(chunks[0])
        else:
            self.seq_len = 0
            
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> dict:
        tokens = self.chunks[idx]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),  # causal LM: labels = inputs
            "attention_mask": torch.ones_like(input_ids),
        }


class MemmapPackedDataset(Dataset):
    """
    Memory-mapped dataset for pre-packed token sequences.
    
    Reads from a numpy memmap file — O(1) RAM regardless of dataset size.
    Drop-in replacement for PackedDataset with identical output format.
    """

    @staticmethod
    def _parse_dtype(dtype_value) -> np.dtype:
        """Parse dtype from metadata, including legacy class-string format."""
        if isinstance(dtype_value, np.dtype):
            return dtype_value

        if isinstance(dtype_value, type):
            return np.dtype(dtype_value)

        if isinstance(dtype_value, str):
            s = dtype_value.strip()
            # Legacy value looked like: "<class 'numpy.uint16'>"
            if s.startswith("<class 'numpy.") and s.endswith("'>"):
                s = s[len("<class 'numpy."):-2]
            try:
                return np.dtype(s)
            except TypeError:
                pass

        logger.warning(
            f"Unrecognized dtype in metadata: {dtype_value!r}; falling back to uint16"
        )
        return np.dtype("uint16")
    
    def __init__(self, memmap_path: str, num_chunks: int, seq_len: int):
        """
        Args:
            memmap_path: Path to the .npy memmap file
            num_chunks: Number of chunks in the file
            seq_len: Sequence length of each chunk
        """
        # Load dtype from metadata if available
        meta_path = str(Path(memmap_path).parent / "metadata.json")
        dtype = np.uint16  # default
        if Path(meta_path).exists():
            with open(meta_path) as f:
                meta = json.load(f)
                dtype_str = meta.get("dtype", "uint16")
                dtype = self._parse_dtype(dtype_str)
        
        self.data = np.memmap(
            memmap_path, dtype=dtype, mode='r',
            shape=(num_chunks, seq_len)
        )
        self.num_chunks = num_chunks
        self.seq_len = seq_len
        
        logger.info(
            f"MemmapPackedDataset: {num_chunks} chunks, seq_len={seq_len}, "
            f"dtype={dtype}, memmap file={memmap_path}"
        )
    
    def __len__(self) -> int:
        return self.num_chunks
    
    def __getitem__(self, idx: int) -> dict:
        # Read row from memmap → convert to int64 tensor
        tokens = self.data[idx]
        input_ids = torch.from_numpy(tokens.astype(np.int64))
        
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),  # causal LM: labels = inputs
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
        }


def create_dataloader(
    dataset: PackedDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create DataLoader with optimized settings.
    
    Args:
        dataset: PackedDataset instance
        batch_size: Micro batch size
        shuffle: Shuffle data each epoch
        num_workers: Number of data loading workers
        prefetch_factor: Batches to prefetch per worker
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop incomplete final batch
        
    Returns:
        Configured DataLoader
    """
    # Disable prefetch if no workers
    if num_workers == 0:
        prefetch_factor = None
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
    
    logger.info(
        f"DataLoader created: {len(dataset)} samples, "
        f"batch_size={batch_size}, workers={num_workers}"
    )
    
    return loader
