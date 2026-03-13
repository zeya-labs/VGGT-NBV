"""Concrete adapter implementations for NBV ports."""

__all__ = [
    "DepthVisualizationAdapter",
    "PyTorch3DMeshRepositoryAdapter",
    "MapAnythingSceneEncoderAdapter",
    "DepthAnything3SceneEncoderAdapter",
    "PyTorch3DRendererAdapter",
    "ReconstructionLossAdapter",
    "ChamferMetricsAdapter",
]


def __getattr__(name: str):
    if name == "DepthVisualizationAdapter":
        from .depth import DepthVisualizationAdapter

        return DepthVisualizationAdapter
    if name == "PyTorch3DMeshRepositoryAdapter":
        from .mesh_repository import PyTorch3DMeshRepositoryAdapter

        return PyTorch3DMeshRepositoryAdapter
    if name == "MapAnythingSceneEncoderAdapter":
        from .scene_encoder.mapanything_adapter import MapAnythingSceneEncoderAdapter

        return MapAnythingSceneEncoderAdapter
    if name == "DepthAnything3SceneEncoderAdapter":
        from .scene_encoder.depthanything3_adapter import DepthAnything3SceneEncoderAdapter

        return DepthAnything3SceneEncoderAdapter
    if name == "PyTorch3DRendererAdapter":
        from .renderer import PyTorch3DRendererAdapter

        return PyTorch3DRendererAdapter
    if name == "ReconstructionLossAdapter":
        from .loss import ReconstructionLossAdapter

        return ReconstructionLossAdapter
    if name == "ChamferMetricsAdapter":
        from .metrics import ChamferMetricsAdapter

        return ChamferMetricsAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
