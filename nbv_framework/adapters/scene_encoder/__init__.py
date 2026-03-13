__all__ = ["MapAnythingSceneEncoderAdapter", "DepthAnything3SceneEncoderAdapter"]


def __getattr__(name: str):
    if name == "MapAnythingSceneEncoderAdapter":
        from .mapanything_adapter import MapAnythingSceneEncoderAdapter

        return MapAnythingSceneEncoderAdapter
    if name == "DepthAnything3SceneEncoderAdapter":
        from .depthanything3_adapter import DepthAnything3SceneEncoderAdapter

        return DepthAnything3SceneEncoderAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
