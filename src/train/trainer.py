"""Training loop with time/steps budget and gradient accumulation."""

import json
import logging
import os
import random
import shutil
import tempfile

import numpy as np
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.time_budget import TimeBudget
from ..utils.logging import MetricsLogger
from .checkpoint import write_latest_metadata

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training loop for Mamba CPT with:
    - Mixed precision (fp16)
    - Gradient accumulation
    - Time and steps budgets
    - Periodic checkpointing
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        config: Dict[str, Any],
        output_dir: str = "./output",
        resume_from_checkpoint: Optional[str] = None,
        tokenizer=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.resume_from_checkpoint = resume_from_checkpoint
        self.tokenizer = tokenizer

        # Extract training config
        train_cfg = config.get("training", {})
        self.lr = train_cfg.get("learning_rate", 5e-5)
        self.weight_decay = train_cfg.get("weight_decay", 0.01)
        self.warmup_steps = train_cfg.get("warmup_steps", 100)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 16)
        self.max_steps = train_cfg.get("max_steps", 5000)
        self.time_budget_seconds = train_cfg.get("time_budget_seconds", 10800)
        self.mixed_precision = train_cfg.get("mixed_precision", True)
        self.amp_dtype = train_cfg.get("amp_dtype", "bfloat16")  # bf16 for Ada Lovelace
        self.optimizer_name = train_cfg.get("optimizer", "adamw_8bit").lower()
        self.adam_beta1 = train_cfg.get("adam_beta1", 0.9)
        self.adam_beta2 = train_cfg.get("adam_beta2", 0.999)
        self.adam_epsilon = train_cfg.get("adam_epsilon", 1e-8)
        self.gradient_checkpointing = train_cfg.get("gradient_checkpointing", True)
        self.empty_cache_every_steps = train_cfg.get("empty_cache_every_steps", 0)
        self.scheduler_name = train_cfg.get("scheduler", "constant").lower()
        self.warmup_ratio = train_cfg.get("warmup_ratio")
        self.phase_id = train_cfg.get("phase_id")
        self.phase_name = train_cfg.get("phase_name")
        self.max_tokens = train_cfg.get("max_tokens")
        self.repeat_dataset = bool(train_cfg.get("repeat_dataset", False))
        self.stop_after_steps = train_cfg.get("stop_after_steps")
        self.stop_file = train_cfg.get("stop_file")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.gradient_accumulation_steps <= 0 or self.max_steps <= 0:
            raise ValueError("max_steps and gradient_accumulation_steps must be positive")
        if getattr(train_loader, "num_workers", 0) != 0:
            raise ValueError("Exact resumable training requires num_workers=0")
        self.data_signature = {
            "cache_fingerprint": train_cfg.get("cache_fingerprint"),
            "dataset_length": len(train_loader.dataset),
            "batch_size": train_loader.batch_size,
            "drop_last": train_loader.drop_last,
            "sampler": type(train_loader.sampler).__name__,
        }

        self.training_signature = {key: train_cfg.get(key) for key in (
            "max_steps", "max_tokens", "gradient_accumulation_steps", "learning_rate",
            "weight_decay", "scheduler", "warmup_ratio", "warmup_steps", "optimizer",
            "amp_dtype", "mixed_precision", "repeat_dataset")}

        # Checkpointing config
        ckpt_cfg = config.get("checkpointing", {})
        self.checkpoint_every_steps = ckpt_cfg.get("checkpoint_every_steps", 500)
        self.save_total_limit = ckpt_cfg.get("save_total_limit", 3)
        self.save_final = ckpt_cfg.get("save_final", True)

        # Logging config
        log_cfg = config.get("logging", {})
        self.log_every_steps = log_cfg.get("log_every_steps", 10)

        # Sequence length for tokens/sec calculation
        data_cfg = config.get("data", {})
        self.seq_len = data_cfg.get("seq_len", 1024)
        self.seq_len = train_cfg.get("seq_len", self.seq_len)
        self.micro_batch_size = train_cfg.get("micro_batch_size", 1)
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if self.world_size != 1:
            raise ValueError("This trainer supports one GPU only; WORLD_SIZE must be 1")
        if len(train_loader) == 0:
            raise ValueError("Training loader is empty (check dataset size and drop_last)")
        if self.save_total_limit is not None and self.save_total_limit < 1:
            raise ValueError("save_total_limit must be at least 1 or null")

        # Device
        self.device = next(model.parameters()).device

        if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing: enabled")

        if hasattr(getattr(self.model, "config", None), "use_cache"):
            self.model.config.use_cache = False
            logger.info("Model config override: use_cache=False for training")

        # Setup directories
        checkpoint_root = ckpt_cfg.get("checkpoint_dir")
        self.checkpoint_root = Path(checkpoint_root) if checkpoint_root else self.output_dir / "checkpoints"
        self.checkpoint_dir = self.checkpoint_root
        if self.phase_id is not None:
            self.checkpoint_dir = self.checkpoint_root / f"phase_{int(self.phase_id)}"
        self.log_dir = Path(log_cfg.get("log_dir")) if log_cfg.get("log_dir") else self.output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_scaler()
        self._setup_logger()

        # Training state
        self.global_step = 0
        self.tokens_seen = 0
        self.source_sample_counts: Dict[str, int] = {}
        self.source_token_counts: Dict[str, int] = {}
        self.start_time = None
        self.epoch = 0
        self.batches_consumed = 0
        self.repeated_tokens = 0
        self.loss_tokens_seen = 0
        self._epoch_rng = None
        self._resume_rng = None
        self.stop_reason = None
        self.phase_completed = False

        # Resume from checkpoint if provided
        if self.resume_from_checkpoint:
            self._load_training_state(self.resume_from_checkpoint)

    def _load_training_state(self, checkpoint_path: str):
        """Load training state from a checkpoint."""
        ckpt_path = Path(checkpoint_path)
        state_file = ckpt_path / "training_state.pt"

        if not state_file.exists():
            raise ValueError(f"Exact resume requires training_state.pt in {checkpoint_path}; use weights-only continuation explicitly")

        logger.info(f"Loading training state from {state_file}")
        state = torch.load(state_file, map_location="cpu", weights_only=False)

        if not self.data_signature["cache_fingerprint"]:
            raise ValueError("Exact resume requires training.cache_fingerprint from the immutable cache manifest")
        if state.get("state_version") != 2 or state.get("data_signature") != self.data_signature or state.get("training_signature") != self.training_signature:
            raise ValueError("Checkpoint lacks an exact data cursor or dataset configuration changed")
        self.epoch = state["epoch"]
        self.batches_consumed = state["batches_consumed"]
        self.repeated_tokens = state["repeated_tokens"]
        self.loss_tokens_seen = state["loss_tokens_seen"]
        self._epoch_rng = state["epoch_rng"]
        self._resume_rng = state["rng"]
        # Restore training state
        self.global_step = state.get("step", 0)
        self.tokens_seen = state.get("tokens_seen", 0)
        self.source_sample_counts = state.get("source_sample_counts", {})
        self.source_token_counts = state.get("source_token_counts", {})

        # Restore optimizer state
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            logger.info("Restored optimizer state")

        # Restore scheduler state
        if "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
            logger.info("Restored scheduler state")

        # Restore scaler state if using fp16
        if self.use_grad_scaler and "scaler_state_dict" in state:
            self.scaler.load_state_dict(state["scaler_state_dict"])
            logger.info("Restored GradScaler state")

        logger.info(f"Resumed from step {self.global_step}, tokens seen: {self.tokens_seen:,}")

    def _setup_optimizer(self):
        """Setup AdamW optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "ln" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if self.optimizer_name == "adamw":
            self.optimizer = AdamW(
                param_groups,
                lr=self.lr,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
            )
            logger.info("Optimizer: AdamW (torch)")
        elif self.optimizer_name == "adamw_8bit":
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise ImportError(
                    "optimizer=adamw_8bit requires bitsandbytes. "
                    "Install the pinned package from requirements.txt."
                ) from exc

            self.optimizer = bnb.optim.AdamW8bit(
                param_groups,
                lr=self.lr,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
            )
            logger.info("Optimizer: AdamW8bit (bitsandbytes)")
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        logger.info(
            "Optimizer settings: lr=%s, weight_decay=%s, betas=(%s, %s), eps=%s",
            self.lr,
            self.weight_decay,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup."""
        if self.warmup_ratio is not None:
            self.warmup_steps = int(max(0, self.max_steps * float(self.warmup_ratio)))

        if self.warmup_steps <= 0:
            if self.scheduler_name == "cosine":
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, self.max_steps))
            elif self.scheduler_name in ("constant", "linear_constant"):
                self.scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=max(1, self.max_steps))
            else:
                raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")
            return

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        remaining_steps = max(1, self.max_steps - self.warmup_steps)
        if self.scheduler_name == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=0.0,
            )
        elif self.scheduler_name in ("constant", "linear_constant"):
            main_scheduler = ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=remaining_steps,
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.warmup_steps],
        )

        logger.info(
            "Scheduler: %s with linear warmup for %s steps",
            self.scheduler_name,
            self.warmup_steps,
        )

    def _setup_scaler(self):
        """Setup dtype for mixed precision (bf16 doesn't need GradScaler)."""
        if self.mixed_precision and self.device.type == "cuda":
            # Determine amp dtype
            if self.amp_dtype in ("bfloat16", "bf16"):
                self.amp_torch_dtype = torch.bfloat16
                self.use_grad_scaler = False  # bf16 doesn't need scaling
                logger.info("Mixed precision: enabled (bfloat16, no GradScaler)")
            else:
                self.amp_torch_dtype = torch.float16
                self.use_grad_scaler = True
                from torch.amp import GradScaler
                self.scaler = GradScaler('cuda')
                logger.info("Mixed precision: enabled (fp16 with GradScaler)")
        else:
            self.amp_torch_dtype = None
            self.use_grad_scaler = False
            logger.info("Mixed precision: disabled")

    def _setup_logger(self):
        """Setup metrics logger."""
        self.metrics_logger = MetricsLogger(
            log_dir=self.log_dir,
            log_json=self.config.get("logging", {}).get("log_json", True),
        )

    def _get_gpu_memory(self) -> Optional[float]:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return None

    def _capture_rng(self):
        return {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader": self.train_loader.generator.get_state() if self.train_loader.generator is not None else None,
        }

    def _restore_rng(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
        if state["loader"] is not None:
            self.train_loader.generator.set_state(state["loader"].cpu())

    def _data_iterator(self):
        if self._epoch_rng is None:
            self._epoch_rng = self._capture_rng()
        else:
            self._restore_rng(self._epoch_rng)
        iterator = iter(self.train_loader)
        for _ in range(self.batches_consumed):
            try:
                next(iterator)
            except StopIteration as exc:
                raise ValueError("Resume cursor exceeds dataset") from exc
        if self._resume_rng is not None:
            self._restore_rng(self._resume_rng)
            self._resume_rng = None
        return iterator

    def _save_checkpoint(self, step: int, is_final: bool = False):
        """Publish complete checkpoints before updating the latest pointer."""
        ckpt_name = "final" if is_final and self.phase_completed else (
            f"stopped_step_{step:06d}" if is_final else f"step_{step:06d}"
        )
        ckpt_path = self.checkpoint_dir / ckpt_name
        if ckpt_path.exists():
            # Never overwrite a published checkpoint in place.
            ckpt_path = self.checkpoint_dir / f"{ckpt_name}_{time.time_ns()}"
        checkpoint_rng = self._capture_rng()
        temporary = Path(tempfile.mkdtemp(prefix=".incomplete-", dir=self.checkpoint_dir))
        try:
            self.model.save_pretrained(temporary)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(temporary)
            state = {
                "state_version": 2, "step": step,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "tokens_seen": self.tokens_seen, "loss_tokens_seen": self.loss_tokens_seen,
                "phase": self.phase_id, "phase_name": self.phase_name,
                "source_sample_counts": self.source_sample_counts,
                "source_token_counts": self.source_token_counts,
                "epoch": self.epoch, "batches_consumed": self.batches_consumed,
                "repeated_tokens": self.repeated_tokens,
                "epoch_rng": self._epoch_rng, "rng": checkpoint_rng,
                "data_signature": self.data_signature,
                "training_signature": self.training_signature,
                "phase_completed": self.phase_completed, "stop_reason": self.stop_reason,
            }
            if self.use_grad_scaler:
                state["scaler_state_dict"] = self.scaler.state_dict()
            torch.save(state, temporary / "training_state.pt")
            (temporary / "checkpoint_metadata.json").write_text(json.dumps({
                "phase": self.phase_id, "phase_completed": self.phase_completed,
                "stop_reason": self.stop_reason, "step": step, "tokens_seen": self.tokens_seen,
                "data_signature": self.data_signature, "training_signature": self.training_signature,
            }, indent=2))
            from ..config import save_yaml_config
            save_yaml_config(self.config, str(temporary / "resolved_config.yaml"))
            temporary.rename(ckpt_path)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        hub_repo = self.config.get("checkpointing", {}).get("hub_repo")
        if hub_repo:
            from .hub import upload_checkpoint
            upload_checkpoint(ckpt_path, hub_repo)
        write_latest_metadata(str(self.checkpoint_root), {
            "latest_checkpoint": str(ckpt_path.resolve()),
            "phase": self.phase_id, "phase_name": self.phase_name,
            "step": step, "tokens_seen": self.tokens_seen,
            "phase_completed": self.phase_completed, "stop_reason": self.stop_reason,
        })
        if not is_final:
            self._cleanup_checkpoints()
        if hub_repo:
            from .hub import prune_remote_checkpoints
            try:
                prune_remote_checkpoints(hub_repo, keep=self.save_total_limit)
            except Exception as exc:
                logger.warning("Remote checkpoint pruning failed: %s", type(exc).__name__)

    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        checkpoints = sorted([
            d for d in self.checkpoint_dir.iterdir()
            if d.is_dir() and d.name.startswith("step_")
        ], key=lambda x: x.stat().st_mtime)

        if self.save_total_limit is None:
            return

        while len(checkpoints) > self.save_total_limit:
            old_ckpt = checkpoints.pop(0)
            logger.info(f"Removing old checkpoint: {old_ckpt}")
            import shutil
            shutil.rmtree(old_ckpt)

    def train(self) -> Dict[str, Any]:
        """Train a finite pass unless repetition is explicitly enabled.

        Checkpoints are taken only at optimizer boundaries. A partial final
        accumulation is normalized by its actual supervised token count.
        """
        self.model.train()
        self.start_time = time.time()
        budget = TimeBudget(self.time_budget_seconds)
        iterator = self._data_iterator()
        self.optimizer.zero_grad(set_to_none=True)
        count = supervised = step_tokens = 0
        loss_sum = 0.0
        step_start = time.time()
        normalization = max(1, self.seq_len * self.micro_batch_size * self.gradient_accumulation_steps)
        pbar = tqdm(total=self.max_steps, initial=self.global_step, desc="Training", unit="step")

        def update():
            nonlocal count, supervised, step_tokens, loss_sum, step_start
            if self.use_grad_scaler:
                self.scaler.unscale_(self.optimizer)
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(normalization / supervised)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm, error_if_nonfinite=True)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient norm")
            if self.use_grad_scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            duration = max(time.time() - step_start, 1e-9)
            if self.log_every_steps and self.global_step % self.log_every_steps == 0:
                elapsed = time.time() - self.start_time
                self.metrics_logger.log({
                    "step": self.global_step, "phase": self.phase_id, "phase_name": self.phase_name,
                    "loss": loss_sum / supervised, "lr": self.scheduler.get_last_lr()[0],
                    "tokens_per_sec": step_tokens / duration, "step_time": duration,
                    "gpu_memory_gb": self._get_gpu_memory(), "elapsed_seconds": elapsed,
                    "total_tokens": self.tokens_seen, "loss_tokens": self.loss_tokens_seen,
                    "repeated_tokens": self.repeated_tokens,
                    "source_token_counts": dict(self.source_token_counts),
                })
            if self.checkpoint_every_steps and self.global_step % self.checkpoint_every_steps == 0:
                self._save_checkpoint(self.global_step)
            if self.empty_cache_every_steps and self.global_step % self.empty_cache_every_steps == 0:
                torch.cuda.empty_cache()
            count = supervised = step_tokens = 0
            loss_sum = 0.0
            step_start = time.time()
            pbar.update(1)

        try:
            while self.global_step < self.max_steps:
                if self.stop_file and Path(self.stop_file).exists():
                    self.stop_reason = "budget_or_operator_stop"
                    break
                if self.stop_after_steps is not None and self.global_step >= self.stop_after_steps:
                    self.stop_reason = "operator_limit"
                    break
                if self.max_tokens is not None and self.tokens_seen >= self.max_tokens:
                    self.stop_reason = "max_tokens"
                    break
                if budget.is_expired():
                    self.stop_reason = "time_budget"
                    break
                try:
                    batch = next(iterator)
                except StopIteration:
                    if not self.repeat_dataset:
                        self.stop_reason = "dataset_exhausted"
                        break
                    if self.batches_consumed == 0:
                        raise ValueError("Training dataset is empty")
                    self.epoch += 1
                    self.batches_consumed = 0
                    self._epoch_rng = None
                    iterator = self._data_iterator()
                    continue
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device).clone()
                mask = batch["attention_mask"].to(self.device).clone()
                if self.max_tokens is not None:
                    remaining = self.max_tokens - self.tokens_seen
                    keep = mask.bool() & (mask.reshape(-1).cumsum(0).reshape_as(mask) <= remaining)
                    mask = keep.to(mask.dtype)
                labels.masked_fill_(~mask.bool(), -100)
                # HF causal losses shift labels by one position.
                n_supervised = int((labels[:, 1:] != -100).sum().item())
                n_tokens = int(mask.sum().item())
                if n_supervised == 0:
                    if self.max_tokens is not None and remaining == 1:
                        self.stop_reason = "unsupervised_token_remainder"
                        break
                    raise ValueError("Batch/token remainder contains no causal training targets")
                with autocast(device_type=self.device.type, dtype=self.amp_torch_dtype, enabled=self.amp_torch_dtype is not None):
                    outputs = self.model(input_ids=input_ids, attention_mask=mask, labels=labels)
                    raw_loss = outputs.loss
                if not torch.isfinite(raw_loss).all():
                    raise FloatingPointError("Non-finite training loss; checkpoint was not published")
                scaled_loss = raw_loss * (n_supervised / normalization)
                if self.use_grad_scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                count += 1
                supervised += n_supervised
                loss_sum += raw_loss.detach().item() * n_supervised
                step_tokens += n_tokens
                self.tokens_seen += n_tokens
                self.loss_tokens_seen += n_supervised
                self.batches_consumed += 1
                if self.epoch > 0:
                    self.repeated_tokens += n_tokens
                token_sources = batch.get("token_source_ids")
                names = batch.get("source")
                if token_sources is not None:
                    source_map = getattr(self.train_loader.dataset, "source_id_to_name", {})
                    for source_row, valid_row in zip(token_sources, mask.cpu()):
                        for source_id in source_row[valid_row.bool()].unique().tolist():
                            if source_id == 0:
                                continue
                            name = source_map.get(int(source_id), str(source_id))
                            tokens = int(((source_row == source_id) & valid_row.bool()).sum().item())
                            self.source_sample_counts[name] = self.source_sample_counts.get(name, 0) + 1
                            self.source_token_counts[name] = self.source_token_counts.get(name, 0) + tokens
                elif names is not None:
                    names = [names] if isinstance(names, str) else names
                    for name, row in zip(names, mask):
                        tokens = int(row.sum().item())
                        if tokens:
                            self.source_sample_counts[name] = self.source_sample_counts.get(name, 0) + 1
                            self.source_token_counts[name] = self.source_token_counts.get(name, 0) + tokens
                if count >= self.gradient_accumulation_steps:
                    update()
            if self.stop_reason is None:
                self.stop_reason = "max_tokens" if self.max_tokens is not None and self.tokens_seen >= self.max_tokens else "max_steps"
            if count:
                update()
            self.phase_completed = self.stop_reason == "max_tokens" or (
                self.max_tokens is None and self.stop_reason in ("max_steps", "dataset_exhausted")
            )
            if self.save_final:
                self._save_checkpoint(self.global_step, is_final=True)
        finally:
            pbar.close()
        return {
            "final_step": self.global_step, "total_tokens": self.tokens_seen,
            "loss_tokens": self.loss_tokens_seen, "repeated_tokens": self.repeated_tokens,
            "epoch": self.epoch, "batches_consumed": self.batches_consumed,
            "total_time_seconds": time.time() - self.start_time,
            "stop_reason": self.stop_reason, "phase_completed": self.phase_completed,
            "phase": self.phase_id, "phase_name": self.phase_name,
            "source_sample_counts": self.source_sample_counts,
            "source_token_counts": self.source_token_counts,
        }
