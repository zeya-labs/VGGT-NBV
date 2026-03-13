"""Concrete adapter implementations for NBV ports."""

from .depth import DepthVisualizationAdapter
from .loss import ReconstructionLossAdapter
from .mesh_repository import PyTorch3DMeshRepositoryAdapter
from .metrics import ChamferMetricsAdapter
from .renderer import PyTorch3DRendererAdapter
from .scene_encoder import DepthAnything3SceneEncoderAdapter, MapAnythingSceneEncoderAdapter

__all__ = [
    "DepthVisualizationAdapter",
    "PyTorch3DMeshRepositoryAdapter",
    "MapAnythingSceneEncoderAdapter",
    "DepthAnything3SceneEncoderAdapter",
    "PyTorch3DRendererAdapter",
    "ReconstructionLossAdapter",
    "ChamferMetricsAdapter",
]
