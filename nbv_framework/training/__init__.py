"""
训练模块

包含端到端的NBV策略训练框架。
"""

__all__ = ["NBVTrainer", "ReconstructionLoss", "ChamferDistance", "ViewpointLoss"]


def __getattr__(name: str):
    if name == "NBVTrainer":
        from .trainer import NBVTrainer

        return NBVTrainer
    if name in {"ReconstructionLoss", "ChamferDistance", "ViewpointLoss"}:
        from .loss import ChamferDistance, ReconstructionLoss, ViewpointLoss

        return {
            "ReconstructionLoss": ReconstructionLoss,
            "ChamferDistance": ChamferDistance,
            "ViewpointLoss": ViewpointLoss,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
