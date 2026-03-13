"""Scene-encoder model implementations."""

__all__ = ["MapAnythingWrapper", "DepthAnything3Wrapper"]


def __getattr__(name: str):
    if name == "MapAnythingWrapper":
        from .mapanything_encoder import MapAnythingWrapper

        return MapAnythingWrapper
    if name == "DepthAnything3Wrapper":
        from .depthanything3_encoder import DepthAnything3Wrapper

        return DepthAnything3Wrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
