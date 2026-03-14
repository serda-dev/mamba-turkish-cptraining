"""Training loop with time/steps budget and gradient accumulation."""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.time_budget import TimeBudget
from ..utils.logging import MetricsLogger

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
        self.micro_batch_size = train_cfg.get("micro_batch_size", 1)
        
        # Device
        self.device = next(model.parameters()).device

        if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing: enabled")

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
            logger.info("Model config override: use_cache=False for training")
        
        # Setup directories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_scaler()
        self._setup_logger()
        
        # Training state
        self.global_step = 0
        self.tokens_seen = 0
        self.start_time = None
        
        # Resume from checkpoint if provided
        if self.resume_from_checkpoint:
            self._load_training_state(self.resume_from_checkpoint)
    
    def _load_training_state(self, checkpoint_path: str):
        """Load training state from a checkpoint."""
        ckpt_path = Path(checkpoint_path)
        state_file = ckpt_path / "training_state.pt"
        
        if not state_file.exists():
            logger.warning(f"No training_state.pt found in {checkpoint_path}, starting from step 0")
            return
        
        logger.info(f"Loading training state from {state_file}")
        state = torch.load(state_file, map_location=self.device, weights_only=False)
        
        # Restore training state
        self.global_step = state.get("step", 0)
        self.tokens_seen = state.get("tokens_seen", 0)
        
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
        # Linear warmup then constant
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        constant_scheduler = ConstantLR(
            self.optimizer,
            factor=1.0,
            total_iters=self.max_steps - self.warmup_steps,
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, constant_scheduler],
            milestones=[self.warmup_steps],
        )
        
        logger.info(f"Scheduler: Linear warmup for {self.warmup_steps} steps")
    
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
    
    def _save_checkpoint(self, step: int, is_final: bool = False):
        """Save model checkpoint."""
        if is_final:
            ckpt_name = "final"
        else:
            ckpt_name = f"step_{step:06d}"
        
        ckpt_path = self.checkpoint_dir / ckpt_name
        
        logger.info(f"Saving checkpoint: {ckpt_path}")
        
        # Save model
        self.model.save_pretrained(ckpt_path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(ckpt_path)
        
        # Save training state
        state = {
            "step": step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "tokens_seen": self.tokens_seen,
        }
        if self.use_grad_scaler:
            state["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(state, ckpt_path / "training_state.pt")
        
        # Cleanup old checkpoints
        if not is_final:
            self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        checkpoints = sorted([
            d for d in self.checkpoint_dir.iterdir()
            if d.is_dir() and d.name.startswith("step_")
        ], key=lambda x: int(x.name.split("_")[1]))
        
        while len(checkpoints) > self.save_total_limit:
            old_ckpt = checkpoints.pop(0)
            logger.info(f"Removing old checkpoint: {old_ckpt}")
            import shutil
            shutil.rmtree(old_ckpt)
    
    def train(self) -> Dict[str, Any]:
        """
        Run training loop.
        
        Returns:
            Dict with final training stats
        """
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info(f"  Max steps: {self.max_steps}")
        logger.info(f"  Time budget: {self.time_budget_seconds}s ({self.time_budget_seconds/3600:.1f}h)")
        logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.micro_batch_size * self.gradient_accumulation_steps}")
        logger.info("=" * 60)
        
        self.model.train()
        self.start_time = time.time()
        
        time_budget = TimeBudget(self.time_budget_seconds)
        
        accumulated_loss = 0.0
        accumulation_count = 0
        step_start_time = time.time()
        
        # Create infinite data iterator
        data_iter = iter(self.train_loader)
        
        # Progress bar
        pbar = tqdm(total=self.max_steps, desc="Training", unit="step")
        
        stop_reason = None
        
        while self.global_step < self.max_steps:
            # Check time budget
            if time_budget.is_expired():
                stop_reason = "time_budget"
                logger.info("Time budget exceeded, stopping training")
                break
            
            # Get next batch (with wraparound)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
            
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Forward pass with mixed precision
            if self.amp_torch_dtype is not None:
                with autocast('cuda', dtype=self.amp_torch_dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / self.gradient_accumulation_steps
                
                if self.use_grad_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()
            
            accumulated_loss += loss.item()
            accumulation_count += 1
            self.tokens_seen += input_ids.numel()
            
            # Gradient accumulation step
            if accumulation_count >= self.gradient_accumulation_steps:
                # Gradient clipping
                if self.use_grad_scaler:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
                
                # Optimizer step
                if self.use_grad_scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                if self.empty_cache_every_steps and self.global_step % self.empty_cache_every_steps == 0:
                    torch.cuda.empty_cache()
                
                self.global_step += 1
                
                # Calculate metrics
                step_time = time.time() - step_start_time
                tokens_per_sec = (
                    self.micro_batch_size * 
                    self.gradient_accumulation_steps * 
                    self.seq_len / step_time
                )
                
                # Log metrics
                if self.global_step % self.log_every_steps == 0:
                    elapsed = time.time() - self.start_time
                    remaining_steps = self.max_steps - self.global_step
                    eta_seconds = (elapsed / self.global_step) * remaining_steps
                    
                    # Time budget remaining
                    time_remaining = max(0, self.time_budget_seconds - elapsed)
                    
                    gpu_mem = self._get_gpu_memory()
                    
                    metrics = {
                        "step": self.global_step,
                        "loss": accumulated_loss,
                        "lr": self.scheduler.get_last_lr()[0],
                        "tokens_per_sec": tokens_per_sec,
                        "step_time": step_time,
                        "gpu_memory_gb": gpu_mem,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta_seconds,
                        "time_remaining_seconds": time_remaining,
                    }
                    
                    self.metrics_logger.log(metrics)
                    
                    pbar.set_postfix({
                        "loss": f"{accumulated_loss:.4f}",
                        "tok/s": f"{tokens_per_sec:.0f}",
                        "mem": f"{gpu_mem:.1f}GB" if gpu_mem else "N/A",
                    })
                
                # Save checkpoint
                if self.global_step % self.checkpoint_every_steps == 0:
                    self._save_checkpoint(self.global_step)
                
                # Reset accumulators
                accumulated_loss = 0.0
                accumulation_count = 0
                step_start_time = time.time()
                
                pbar.update(1)
        
        pbar.close()
        
        if stop_reason is None:
            stop_reason = "max_steps"
        
        # Save final checkpoint
        if self.save_final:
            self._save_checkpoint(self.global_step, is_final=True)
        
        # Training summary
        total_time = time.time() - self.start_time
        
        summary = {
            "final_step": self.global_step,
            "total_tokens": self.tokens_seen,
            "total_time_seconds": total_time,
            "stop_reason": stop_reason,
        }
        
        logger.info("=" * 60)
        logger.info("Training complete")
        logger.info(f"  Final step: {self.global_step}")
        logger.info(f"  Total tokens: {self.tokens_seen:,}")
        logger.info(f"  Total time: {total_time/3600:.2f}h")
        logger.info(f"  Stop reason: {stop_reason}")
        logger.info("=" * 60)
        
        return summary
