"""
工具模块

包含数据加载、可视化、评估等辅助功能。
"""

from .datasets import NBVDataset, create_data_loader
from .mesh_utils import load_mesh_as_pytorch3d, normalize_mesh, load_and_normalize_mesh
from .textures import TextureGenerator
from .synth_data import create_synthetic_training_data, SyntheticDataGenerator
from .visualization import visualize_reconstruction, plot_training_curves
from .evaluation import evaluate_nbv_policy

from .coordinate_utils import get_up_vector, get_coordinate_transform_matrix, apply_transform_to_vertices
from .mesh_generator import MeshGenerator, create_pytorch3d_mesh
from .camera_utils import CameraPoseGenerator, pose_dict_to_tensor, tensor_to_pose_dict

__all__ = [
    "NBVDataset",
    "create_data_loader",
    "load_mesh_as_pytorch3d",
    "normalize_mesh",
    "load_and_normalize_mesh",
    "TextureGenerator",
    "create_synthetic_training_data",
    "SyntheticDataGenerator",
    "visualize_reconstruction",
    "plot_training_curves",
    "evaluate_nbv_policy",
    # 新增的模块
    "get_up_vector",
    "get_coordinate_transform_matrix", 
    "apply_transform_to_vertices",
    "MeshGenerator",
    "create_pytorch3d_mesh",
    "CameraPoseGenerator",
    "pose_dict_to_tensor",
    "tensor_to_pose_dict"
]