"""CPU regression tests for expensive single-GPU training state contracts."""
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.train.trainer import Trainer
from src.train.checkpoint import read_latest_metadata, find_latest_checkpoint


class Rows(Dataset):
    def __init__(self, count=5):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        ids = torch.tensor([i + 1, i + 2, 0, 0])
        return {"input_ids": ids, "labels": ids.clone(),
                "attention_mask": torch.tensor([1, 1, 0, 0]), "source": "turkish"}


class TinyModel(torch.nn.Module):
    def __init__(self, nan=False, dropout=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.7))
        self.config = SimpleNamespace(use_cache=True)
        self.nan = nan
        self.dropout = dropout

    def forward(self, input_ids, attention_mask, labels):
        valid = labels[:, 1:] != -100
        targets = input_ids[:, 1:].float()[valid]
        noise = torch.rand(()) if self.dropout else 1.0
        loss = ((self.weight * noise - targets) ** 2).mean()
        return SimpleNamespace(loss=loss * float("nan") if self.nan else loss)

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path / "model.pt")


def config(**training):
    return {"training": {"optimizer": "adamw", "mixed_precision": False,
            "gradient_checkpointing": False, "gradient_accumulation_steps": 2,
            "warmup_steps": 0, "max_steps": 100, "time_budget_seconds": None,
            "learning_rate": 0.01, "cache_fingerprint": "fixture-v1", **training},
            "data": {"seq_len": 4}, "checkpointing": {"checkpoint_every_steps": 1},
            "logging": {"log_every_steps": 1}}


def trainer(path, *, count=5, model=None, cfg=None, resume=None, shuffle=False):
    loader = DataLoader(Rows(count), batch_size=1, shuffle=shuffle, num_workers=0)
    return Trainer(model or TinyModel(), loader, cfg or config(), str(path), resume)


def test_finite_pass_counts_padding_and_flushes_partial(tmp_path):
    result = trainer(tmp_path).train()
    assert result["final_step"] == 3
    assert result["total_tokens"] == 10
    assert result["loss_tokens"] == 5
    assert result["repeated_tokens"] == 0
    assert result["source_token_counts"] == {"turkish": 10}
    assert result["stop_reason"] == "dataset_exhausted"
    assert result["phase_completed"]
    assert read_latest_metadata(str(tmp_path / "checkpoints"))["phase_completed"]


def test_hard_token_budget_normalizes_final_accumulation(tmp_path):
    result = trainer(tmp_path, cfg=config(max_tokens=6)).train()
    assert result["total_tokens"] == 6
    assert result["final_step"] == 2
    assert result["phase_completed"]
    result = trainer(tmp_path / "insufficient", count=2, cfg=config(max_tokens=100)).train()
    assert result["stop_reason"] == "dataset_exhausted"
    assert not result["phase_completed"]


def test_time_stop_is_resumable_not_completed(tmp_path):
    result = trainer(tmp_path, cfg=config(time_budget_seconds=0)).train()
    assert result["stop_reason"] == "time_budget"
    metadata = read_latest_metadata(str(tmp_path / "checkpoints"))
    assert not metadata["phase_completed"]
    assert "stopped_step" in metadata["latest_checkpoint"]


def test_nonfinite_loss_does_not_publish_checkpoint(tmp_path):
    with pytest.raises(FloatingPointError, match="Non-finite"):
        trainer(tmp_path, model=TinyModel(nan=True)).train()
    assert not (tmp_path / "checkpoints" / "latest.json").exists()


def test_exact_resume_preserves_shuffled_cursor_and_dropout_rng(tmp_path):
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    original = trainer(tmp_path / "whole", model=TinyModel(dropout=True), cfg=config(max_steps=3), shuffle=True)
    original.train()
    checkpoint = tmp_path / "whole" / "checkpoints" / "step_000001"
    resumed_model = TinyModel(dropout=True)
    resumed_model.load_state_dict(torch.load(checkpoint / "model.pt", weights_only=True))
    # Perturb RNG before resume to ensure restoration is doing actual work.
    torch.manual_seed(900)
    resumed = trainer(tmp_path / "resumed", model=resumed_model, cfg=config(max_steps=3), resume=str(checkpoint), shuffle=True)
    result = resumed.train()
    assert torch.equal(original.model.weight, resumed.model.weight)
    assert result["total_tokens"] == original.tokens_seen
    assert result["source_token_counts"] == original.source_token_counts


def test_resume_rejects_changed_data_and_legacy_cursor(tmp_path):
    initial = trainer(tmp_path, cfg=config(max_steps=1))
    initial.train()
    checkpoint = read_latest_metadata(str(tmp_path / "checkpoints"))["latest_checkpoint"]
    with pytest.raises(ValueError, match="configuration changed"):
        trainer(tmp_path / "changed", count=6, resume=checkpoint)
    with pytest.raises(ValueError, match="requires training_state"):
        trainer(tmp_path / "missing", resume=str(tmp_path))


def test_failed_checkpoint_save_leaves_latest_pointer_intact(tmp_path):
    instance = trainer(tmp_path, cfg=config(max_steps=1))
    instance.train()
    metadata = read_latest_metadata(str(tmp_path / "checkpoints"))
    def fail(path):
        (path / "partial").write_text("incomplete")
        raise OSError("disk full")
    instance.model.save_pretrained = fail
    with pytest.raises(OSError, match="disk full"):
        instance._save_checkpoint(2)
    assert read_latest_metadata(str(tmp_path / "checkpoints")) == metadata
    assert not list((tmp_path / "checkpoints").glob(".incomplete-*"))
    assert find_latest_checkpoint(str(tmp_path / "checkpoints")) == metadata["latest_checkpoint"]


def test_operator_pilot_limit_preserves_scheduler_and_can_resume(tmp_path):
    initial = trainer(tmp_path / "pilot", count=10, cfg=config(max_steps=10, max_tokens=20, stop_after_steps=1))
    result = initial.train()
    assert result["stop_reason"] == "operator_limit"
    assert not result["phase_completed"]
    checkpoint = read_latest_metadata(str(tmp_path / "pilot" / "checkpoints"))["latest_checkpoint"]
    model = TinyModel()
    model.load_state_dict(torch.load(__import__('pathlib').Path(checkpoint) / "model.pt", weights_only=True))
    resumed = trainer(tmp_path / "continue", count=10, model=model, cfg=config(max_steps=10, max_tokens=20), resume=checkpoint)
    assert resumed.global_step == 1
    assert resumed.scheduler.state_dict() == initial.scheduler.state_dict()
    assert resumed.train()["total_tokens"] == 20


def test_step_ceiling_does_not_claim_unreached_token_target(tmp_path):
    result = trainer(tmp_path, cfg=config(max_steps=1, max_tokens=20)).train()
    assert result["total_tokens"] == 4
    assert result["stop_reason"] == "max_steps"
    assert not result["phase_completed"]


def test_partial_accumulation_has_same_gradient_as_one_actual_batch(tmp_path):
    partial = trainer(tmp_path / "partial", count=1, cfg=config(gradient_accumulation_steps=8))
    complete = trainer(tmp_path / "complete", count=1, cfg=config(gradient_accumulation_steps=1))
    partial.train()
    complete.train()
    assert torch.equal(partial.model.weight, complete.model.weight)
    first = partial.optimizer.state[partial.model.weight]["exp_avg"]
    second = complete.optimizer.state[complete.model.weight]["exp_avg"]
    assert torch.equal(first, second)
