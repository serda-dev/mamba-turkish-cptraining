"""PyTorch Dataset and DataLoader for packed sequences."""

import json
import logging
import bisect
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
        source_ids_path = None
        source_id_to_name = {}
        if Path(meta_path).exists():
            with open(meta_path) as f:
                meta = json.load(f)
                dtype_str = meta.get("dtype", "uint16")
                dtype = self._parse_dtype(dtype_str)
                source_ids_path = meta.get("source_ids_path")
                source_id_to_name = {
                    int(k): v for k, v in meta.get("source_id_to_name", {}).items()
                }
        
        self.data = np.memmap(
            memmap_path, dtype=dtype, mode='r',
            shape=(num_chunks, seq_len)
        )
        self.num_chunks = num_chunks
        self.seq_len = seq_len
        self.source_ids = None
        self.source_id_to_name = source_id_to_name
        if source_ids_path and Path(source_ids_path).exists():
            self.source_ids = np.memmap(
                source_ids_path, dtype=np.uint8, mode="r", shape=(num_chunks,)
            )
        
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
        
        item = {
            "input_ids": input_ids,
            "labels": input_ids.clone(),  # causal LM: labels = inputs
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
        }
        if self.source_ids is not None:
            source_id = int(self.source_ids[idx])
            item["source_id"] = torch.tensor(source_id, dtype=torch.long)
            item["source"] = self.source_id_to_name.get(source_id, str(source_id))
        return item


class ShardedMemmapPackedDataset(Dataset):
    """Memory-mapped packed-token dataset split across multiple shard files."""

    def __init__(self, manifest_path: str):
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        if not self.manifest.get("complete"):
            raise ValueError(f"Token cache is not complete: {manifest_path}")

        self.manifest_path = manifest_path
        self.cache_dir = manifest_path.parent
        self.seq_len = int(self.manifest["seq_len"])
        self.dtype = MemmapPackedDataset._parse_dtype(self.manifest.get("dtype", "uint16"))
        self.shards = self.manifest.get("shards", [])
        self.source_id_to_name = {
            int(k): v for k, v in self.manifest.get("source_id_to_name", {}).items()
        }
        self.cumulative = []
        total = 0
        for shard in self.shards:
            total += int(shard["num_chunks"])
            self.cumulative.append(total)
        self.num_chunks = total
        self._open_shard_idx = None
        self._open_data = None
        self._open_source_ids = None

        logger.info(
            "ShardedMemmapPackedDataset: %s chunks, %s shard(s), seq_len=%s, dtype=%s",
            self.num_chunks,
            len(self.shards),
            self.seq_len,
            self.dtype,
        )

    def __len__(self) -> int:
        return self.num_chunks

    def _open_shard(self, shard_idx: int):
        if self._open_shard_idx == shard_idx:
            return
        shard = self.shards[shard_idx]
        num_chunks = int(shard["num_chunks"])
        token_path = self.cache_dir / shard["path"]
        source_path = self.cache_dir / shard["source_ids_path"]
        self._open_data = np.memmap(
            token_path, dtype=self.dtype, mode="r", shape=(num_chunks, self.seq_len)
        )
        self._open_source_ids = np.memmap(
            source_path, dtype=np.uint8, mode="r", shape=(num_chunks,)
        )
        self._open_shard_idx = shard_idx

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self.num_chunks:
            raise IndexError(idx)
        shard_idx = bisect.bisect_right(self.cumulative, idx)
        shard_start = 0 if shard_idx == 0 else self.cumulative[shard_idx - 1]
        local_idx = idx - shard_start
        self._open_shard(shard_idx)

        tokens = self._open_data[local_idx]
        input_ids = torch.from_numpy(tokens.astype(np.int64))
        source_id = int(self._open_source_ids[local_idx])
        item = {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "source_id": torch.tensor(source_id, dtype=torch.long),
            "source": self.source_id_to_name.get(source_id, str(source_id)),
        }
        return item


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
