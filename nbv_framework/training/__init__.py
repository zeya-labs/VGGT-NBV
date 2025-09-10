"""
训练模块

包含端到端的NBV策略训练框架。
"""

from .trainer import NBVTrainer
from .losses import ReconstructionLoss, ChamferDistance

__all__ = ["NBVTrainer", "ReconstructionLoss", "ChamferDistance"]