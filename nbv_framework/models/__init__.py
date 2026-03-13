"""Core model implementations used by the NBV training pipeline."""

__all__ = [
    "AttentionNBVPolicy",
    "BaseNBVPolicy",
    "MapAnythingWrapper",
    "DepthAnything3Wrapper",
]


def __getattr__(name: str):
    if name in {"AttentionNBVPolicy", "BaseNBVPolicy"}:
        from .policy import AttentionNBVPolicy, BaseNBVPolicy

        return {
            "AttentionNBVPolicy": AttentionNBVPolicy,
            "BaseNBVPolicy": BaseNBVPolicy,
        }[name]
    if name in {"MapAnythingWrapper", "DepthAnything3Wrapper"}:
        from .scene_encoder import DepthAnything3Wrapper, MapAnythingWrapper

        return {
            "MapAnythingWrapper": MapAnythingWrapper,
            "DepthAnything3Wrapper": DepthAnything3Wrapper,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
