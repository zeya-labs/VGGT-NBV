"""Training entrypoints and builders for the NBV pipeline."""

__all__ = [
    "NBVDataModule",
    "LightningNBVModule",
    "build_datamodule",
    "build_lightning_module",
    "build_trainer",
]


def __getattr__(name: str):
    if name == "NBVDataModule":
        from .data_module import NBVDataModule

        return NBVDataModule
    if name == "LightningNBVModule":
        from .lightning_module import LightningNBVModule

        return LightningNBVModule
    if name == "build_datamodule":
        from .datamodule_factory import build_datamodule

        return build_datamodule
    if name == "build_lightning_module":
        from .factory import build_lightning_module

        return build_lightning_module
    if name == "build_trainer":
        from .trainer import build_trainer

        return build_trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
