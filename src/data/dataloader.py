"""PyTorch Dataset and DataLoader for packed sequences."""

import logging
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class PackedDataset(Dataset):
    """
    Dataset for pre-packed token sequences.
    
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
