# Copyright (c) 2025 NBV Framework
# Objective-Driven Policy Learning for Active 3D Reconstruction

"""NBV Framework package."""

__version__ = "0.1.0"
__author__ = "NBV Research Team"

__all__ = [
    "MapAnythingWrapper",
    "DepthAnything3Wrapper",
    "BaseNBVPolicy",
    "AttentionNBVPolicy",
    "DifferentiableRenderer",
    "LightningNBVModule",
    "NBVDataModule",
    "NBVConfig",
]


def __getattr__(name: str):
    if name in {
        "MapAnythingWrapper",
        "DepthAnything3Wrapper",
        "BaseNBVPolicy",
        "AttentionNBVPolicy",
    }:
        from .models import (
            AttentionNBVPolicy,
            BaseNBVPolicy,
            DepthAnything3Wrapper,
            MapAnythingWrapper,
        )

        return {
            "MapAnythingWrapper": MapAnythingWrapper,
            "DepthAnything3Wrapper": DepthAnything3Wrapper,
            "BaseNBVPolicy": BaseNBVPolicy,
            "AttentionNBVPolicy": AttentionNBVPolicy,
        }[name]
    if name == "DifferentiableRenderer":
        from .infrastructure.rendering import DifferentiableRenderer

        return DifferentiableRenderer
    if name == "LightningNBVModule":
        from .training import LightningNBVModule

        return LightningNBVModule
    if name == "NBVDataModule":
        from .training import NBVDataModule

        return NBVDataModule
    if name == "NBVConfig":
        from .config import NBVConfig

        return NBVConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
