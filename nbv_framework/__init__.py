# Copyright (c) 2025 NBV Framework
# Objective-Driven Policy Learning for Active 3D Reconstruction

"""NBV Framework package."""

__version__ = "0.1.0"
__author__ = "NBV Research Team"

from .models import AttentionNBVPolicy, BaseNBVPolicy, MapAnythingWrapper
from .rendering import DifferentiableRenderer
from .training import NBVTrainer

__all__ = [
    "MapAnythingWrapper",
    "BaseNBVPolicy",
    "AttentionNBVPolicy",
    "DifferentiableRenderer",
    "NBVTrainer",
]
