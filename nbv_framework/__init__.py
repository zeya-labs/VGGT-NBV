# Copyright (c) 2025 NBV Framework
# Objective-Driven Policy Learning for Active 3D Reconstruction

"""NBV Framework package."""

__version__ = "0.1.0"
__author__ = "NBV Research Team"

__all__ = [
    "MapAnythingWrapper",
    "BaseNBVPolicy",
    "AttentionNBVPolicy",
    "DifferentiableRenderer",
    "NBVTrainer",
]


def __getattr__(name: str):
    if name in {"MapAnythingWrapper", "BaseNBVPolicy", "AttentionNBVPolicy"}:
        from .models import AttentionNBVPolicy, BaseNBVPolicy, MapAnythingWrapper

        return {
            "MapAnythingWrapper": MapAnythingWrapper,
            "BaseNBVPolicy": BaseNBVPolicy,
            "AttentionNBVPolicy": AttentionNBVPolicy,
        }[name]
    if name == "DifferentiableRenderer":
        from .rendering import DifferentiableRenderer

        return DifferentiableRenderer
    if name == "NBVTrainer":
        from .training import NBVTrainer

        return NBVTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
