"""Training module."""

def __getattr__(name):
    if name == "Trainer":
        from .trainer import Trainer

        return Trainer
    raise AttributeError(name)

__all__ = ["Trainer"]
