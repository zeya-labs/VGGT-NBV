"""
数据处理模块（聚合导出）

为向后兼容保留原入口：
 - load_mesh_as_pytorch3d, normalize_mesh, load_and_normalize_mesh 来自 mesh_utils
"""

from .mesh_utils import load_mesh_as_pytorch3d, normalize_mesh, load_and_normalize_mesh

__all__ = [
    "load_mesh_as_pytorch3d",
    "normalize_mesh",
    "load_and_normalize_mesh",
]

