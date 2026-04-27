"""Data processing modules for Mamba CPT pipeline."""

from .jsonl_reader import read_jsonl_files
from .preprocess import preprocess_texts
from .tokenize_pack import (
    pack_and_tokenize,
    pack_and_tokenize_to_memmap,
    pack_and_tokenize_to_sharded_cache,
    token_cache_is_complete,
)
from .mixing import weighted_mix_texts


def __getattr__(name):
    if name in {
        "PackedDataset",
        "MemmapPackedDataset",
        "ShardedMemmapPackedDataset",
        "create_dataloader",
    }:
        from . import dataloader

        return getattr(dataloader, name)
    raise AttributeError(name)

__all__ = [
    "read_jsonl_files",
    "preprocess_texts", 
    "pack_and_tokenize",
    "pack_and_tokenize_to_memmap",
    "pack_and_tokenize_to_sharded_cache",
    "token_cache_is_complete",
    "PackedDataset",
    "MemmapPackedDataset",
    "ShardedMemmapPackedDataset",
    "create_dataloader",
    "weighted_mix_texts",
]
