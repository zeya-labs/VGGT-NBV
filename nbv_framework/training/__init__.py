"""
训练模块

包含端到端的NBV策略训练框架。
"""

from .loss import ChamferDistance, ReconstructionLoss, ViewpointLoss
from .trainer import NBVTrainer

__all__ = ["NBVTrainer", "ReconstructionLoss", "ChamferDistance", "ViewpointLoss"]
