# Copyright (c) 2025 NBV Framework
# Objective-Driven Policy Learning for Active 3D Reconstruction

"""
NBV Framework: 目标驱动的策略学习框架
基于基础模型监督的主动三维重建

核心组件:
- VGGTWrapper: 冻结的VGGT基础模型封装
- BaseNBVPolicy: 可训练的NBV策略网络基类
- DifferentiableRenderer: 可微分渲染器
- NBVTrainer: 端到端训练框架
"""

__version__ = "0.1.0"
__author__ = "NBV Research Team"

from .models import VGGTWrapper, BaseNBVPolicy, BasicNBVPolicy, AttentionNBVPolicy, IterativeNBVPolicy, MultiScaleNBVPolicy, HybridNBVPolicy, GeometryAwareNBVPolicy, create_nbv_policy
from .rendering import DifferentiableRenderer
from .training import NBVTrainer

__all__ = [
    "VGGTWrapper",
    "BaseNBVPolicy", 
    "DifferentiableRenderer",
    "NBVTrainer",
    "BasicNBVPolicy",
    "AttentionNBVPolicy",
    "IterativeNBVPolicy",
    "MultiScaleNBVPolicy",
    "HybridNBVPolicy",
    "GeometryAwareNBVPolicy",
    "create_nbv_policy"
]