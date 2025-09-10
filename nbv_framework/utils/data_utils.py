"""
数据处理模块（聚合导出）

为向后兼容保留原入口：
 - NBVDataset, create_data_loader 来自 datasets
 - load_mesh_as_pytorch3d, normalize_mesh, load_and_normalize_mesh 来自 mesh_utils
 - TextureGenerator 来自 textures
 - create_synthetic_training_data 来自 synth_data
"""

from .datasets import NBVDataset, create_data_loader
from .mesh_utils import load_mesh_as_pytorch3d, normalize_mesh, load_and_normalize_mesh
from .textures import TextureGenerator
from .synth_data import create_synthetic_training_data

__all__ = [
    "NBVDataset",
    "create_data_loader",
    "load_mesh_as_pytorch3d",
    "normalize_mesh",
    "load_and_normalize_mesh",
    "TextureGenerator",
    "create_synthetic_training_data",
]


