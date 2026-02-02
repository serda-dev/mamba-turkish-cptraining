"""Time budget management for training."""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TimeBudget:
    """
    Track elapsed time and check against a budget.
    
    Usage:
        budget = TimeBudget(10800)  # 3 hours
        while not budget.is_expired():
            # training step
            pass
    """
    
    def __init__(self, budget_seconds: Optional[float] = None):
        """
        Args:
            budget_seconds: Time budget in seconds. None for unlimited.
        """
        self.budget_seconds = budget_seconds
        self.start_time = time.time()
        
    def elapsed(self) -> float:
        """Return elapsed time in seconds."""
        return time.time() - self.start_time
    
    def remaining(self) -> Optional[float]:
        """Return remaining time in seconds, or None if unlimited."""
        if self.budget_seconds is None:
            return None
        return max(0, self.budget_seconds - self.elapsed())
    
    def is_expired(self) -> bool:
        """Check if time budget is exhausted."""
        if self.budget_seconds is None:
            return False
        return self.elapsed() >= self.budget_seconds
    
    def check_and_log(self, interval: float = 300) -> bool:
        """
        Check budget and log progress periodically.
        
        Args:
            interval: Logging interval in seconds
            
        Returns:
            True if expired
        """
        elapsed = self.elapsed()
        
        if self.budget_seconds is not None:
            remaining = self.remaining()
            pct = (elapsed / self.budget_seconds) * 100
            logger.debug(
                f"Time budget: {elapsed:.0f}s / {self.budget_seconds:.0f}s "
                f"({pct:.1f}%), {remaining:.0f}s remaining"
            )
        
        return self.is_expired()
    
    def __repr__(self) -> str:
        if self.budget_seconds is None:
            return f"TimeBudget(unlimited, elapsed={self.elapsed():.0f}s)"
        return (
            f"TimeBudget({self.budget_seconds:.0f}s, "
            f"elapsed={self.elapsed():.0f}s, "
            f"remaining={self.remaining():.0f}s)"
        )
