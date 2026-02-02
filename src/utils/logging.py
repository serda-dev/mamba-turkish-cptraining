"""Logging utilities for training metrics."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Format
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
    
    # Reduce verbosity of some libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


class MetricsLogger:
    """
    Log training metrics to text and JSON files.
    """
    
    def __init__(
        self,
        log_dir: str = "./output/logs",
        log_json: bool = True,
        experiment_name: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_json = log_json
        
        # Generate experiment name if not provided
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        
        # File paths
        self.text_log_path = self.log_dir / f"metrics_{experiment_name}.log"
        self.json_log_path = self.log_dir / f"metrics_{experiment_name}.jsonl"
        
        # Write headers
        with open(self.text_log_path, "w") as f:
            f.write(f"# Training metrics - {experiment_name}\n")
            f.write(f"# Started: {datetime.now().isoformat()}\n\n")
    
    def log(self, metrics: Dict[str, Any]) -> None:
        """
        Log a metrics dictionary.
        
        Args:
            metrics: Dict with metric names and values
        """
        # Add timestamp
        metrics["timestamp"] = datetime.now().isoformat()
        
        # Format for text log
        step = metrics.get("step", "?")
        loss = metrics.get("loss", 0)
        lr = metrics.get("lr", 0)
        tokens_per_sec = metrics.get("tokens_per_sec", 0)
        gpu_mem = metrics.get("gpu_memory_gb", 0)
        eta = metrics.get("eta_seconds", 0)
        
        text_line = (
            f"step={step:>6} | "
            f"loss={loss:.4f} | "
            f"lr={lr:.2e} | "
            f"tok/s={tokens_per_sec:>6.0f} | "
            f"mem={gpu_mem:.1f}GB | "
            f"eta={eta/60:.0f}min"
        )
        
        # Write to text log
        with open(self.text_log_path, "a") as f:
            f.write(text_line + "\n")
        
        # Write to JSON log
        if self.log_json:
            with open(self.json_log_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        
        # Also log to Python logger
        logger.info(text_line)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log the full config at the start of training."""
        config_path = self.log_dir / f"config_{self.experiment_name}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        logger.info(f"Config saved to {config_path}")
