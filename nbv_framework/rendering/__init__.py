"""
渲染模块

包含可微分渲染器，用于从给定相机位姿和GT mesh生成新视图。
"""

from .differentiable_renderer import DifferentiableRenderer

__all__ = ["DifferentiableRenderer"]