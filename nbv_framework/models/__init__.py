"""
模型组件模块

包含：
- VGGTWrapper: VGGT基础模型的封装类
- NBVPolicyNetwork: 基础NBV策略网络
- IterativeNBVPolicy: 迭代细化NBV策略网络
- MultiScaleNBVPolicy: 多尺度特征NBV策略网络
- HybridNBVPolicy: 混合架构NBV策略网络
- GeometryAwareNBVPolicy: 几何感知NBV策略网络
- create_nbv_policy: NBV策略网络工厂函数
"""

from .vggt_wrapper import VGGTWrapper
from .nbv_policy_networks import BaseNBVPolicy, BasicNBVPolicy, AttentionNBVPolicy, IterativeNBVPolicy, MultiScaleNBVPolicy, HybridNBVPolicy, GeometryAwareNBVPolicy, create_nbv_policy

__all__ = [
    "VGGTWrapper",
    "BaseNBVPolicy", 
    "BasicNBVPolicy", 
    "AttentionNBVPolicy",
    "IterativeNBVPolicy",
    "MultiScaleNBVPolicy",
    "HybridNBVPolicy", 
    "GeometryAwareNBVPolicy",
    "create_nbv_policy",
]