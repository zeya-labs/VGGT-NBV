"""
相机工具模块
处理相机位姿生成、保存和转换
"""

import json
import math
import numpy as np
import torch
from typing import Dict, List, Optional
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

from .coordinate_utils import get_up_vector, generate_fibonacci_sphere_points, generate_fibonacci_upper_hemisphere_points



class CameraPoseGenerator:
    """相机位姿生成器"""
    
    def __init__(self, up_axis: str = "Y"):
        """
        初始化相机位姿生成器
        
        Args:
            up_axis: 上朝向轴
        """
        self.up_axis = up_axis
        self.up_vector = get_up_vector(up_axis)
    
    def _generate_poses_from_positions(
        self,
        sphere_positions: np.ndarray,
        seed: int = 0,
        base_radius: float = 2.5,
        radius_variation: float = 0
    ) -> List[Dict[str, List[float]]]:
        """
        从球面位置生成相机位姿的通用方法
        
        Args:
            sphere_positions: 球面位置数组
            seed: 随机种子
            base_radius: 基础相机距离
            radius_variation: 距离变化范围
            
        Returns:
            camera_poses: 相机位姿列表，每个元素包含position和quaternion
        """
        # 使用局部随机数生成器，避免影响全局随机状态
        rng = np.random.RandomState(seed)
        
        poses = []
        
        for i, direction in enumerate(sphere_positions):
            # 添加随机距离变化
            radius = base_radius + rng.uniform(-radius_variation, radius_variation)
            
            # 计算相机位置
            position = direction * radius
            
            # 使用PyTorch3D的look_at_view_transform
            # 将numpy数组转换为张量以避免性能警告
            eye = torch.tensor(position.reshape(1, 3), dtype=torch.float32)
            at = torch.tensor([[0, 0, 0]], dtype=torch.float32)  # 看向原点
            
            up = torch.tensor(self.up_vector.reshape(1, -1), dtype=torch.float32)  # 使用指定的up向量
            
            R, T = look_at_view_transform(eye=eye, at=at, up=up)
            
            # 将旋转矩阵转换为四元数
            quaternions_wxyz = matrix_to_quaternion(R)
            q = quaternions_wxyz[0]  # 获取批次中的第一个四元数
            quaternion_xyzw = [q[1].item(), q[2].item(), q[3].item(), q[0].item()]

            poses.append({
                "position": [float(position[0]), float(position[1]), float(position[2])],
                "quaternion": quaternion_xyzw
            })
        
        return poses
    
    def generate_camera_poses(
        self, 
        num_views: int, 
        seed: int = 0,
        base_radius: float = 2.5, # 0.9, 0.8是2.5，0.7是2.86
        radius_variation: float = 0,
        hemisphere: str = 'full'
    ) -> List[Dict[str, List[float]]]:
        """
        生成相机位姿
        
        Args:
            num_views: 视图数量
            seed: 随机种子
            base_radius: 基础相机距离
            radius_variation: 距离变化范围
            hemisphere: 球面类型，'full'表示全球面，'upper'表示上半球面
            
        Returns:
            camera_poses: 相机位姿列表，每个元素包含position和quaternion
        """
        if hemisphere == 'upper':
            # 生成上半球面上的均匀分布点
            sphere_positions, _ = generate_fibonacci_upper_hemisphere_points(num_views, radius=1.0, up_axis=self.up_axis)
        else:
            # 生成球面上的均匀分布点
            sphere_positions, _ = generate_fibonacci_sphere_points(num_views, radius=1.0)
        
        return self._generate_poses_from_positions(
            sphere_positions, seed, base_radius, radius_variation
        )
    
    def save_camera_poses(self, camera_poses: List[Dict], filepath: str):
        """
        保存相机位姿到JSON文件
        
        Args:
            camera_poses: 相机位姿列表
            filepath: 保存路径
        """
        with open(filepath, 'w') as f:
            json.dump(camera_poses, f, indent=2)
    
    def load_camera_poses(self, filepath: str) -> List[Dict]:
        """
        从JSON文件加载相机位姿
        
        Args:
            filepath: 文件路径
            
        Returns:
            camera_poses: 相机位姿列表
        """
        with open(filepath, 'r') as f:
            return json.load(f)


def pose_dict_to_tensor(pose_dict: Dict[str, List[float]], device: str = "cuda") -> torch.Tensor:
    """
    将相机位姿字典转换为张量格式
    
    Args:
        pose_dict: 包含position和quaternion的字典
        device: 计算设备
        
    Returns:
        pose_tensor: 相机位姿张量 [1, 7] (x, y, z, qx, qy, qz, qw)
    """
    position = pose_dict["position"]
    quaternion = pose_dict["quaternion"]
    
    pose_tensor = torch.tensor([
        position + quaternion
    ], dtype=torch.float32, device=device)
    
    return pose_tensor


def tensor_to_pose_dict(pose_tensor: torch.Tensor) -> Dict[str, List[float]]:
    """
    将相机位姿张量转换为字典格式
    
    Args:
        pose_tensor: 相机位姿张量 [1, 7] (x, y, z, qx, qy, qz, qw)
        
    Returns:
        pose_dict: 包含position和quaternion的字典
    """
    pose_array = pose_tensor.cpu().numpy()[0]
    
    return {
        "position": pose_array[:3].tolist(),
        "quaternion": pose_array[3:].tolist()
    }


def position_to_pose_tensor(positions: torch.Tensor, up_axis: str = "Y") -> torch.Tensor:
    """
    将位置张量转换为完整的相机位姿张量（包含位置和四元数）
    
    Args:
        positions: 相机位置张量 [B, 3] (x, y, z)
        up_axis: 上朝向轴，默认为"Y"
        
    Returns:
        pose_tensor: 相机位姿张量 [B, 7] (x, y, z, qx, qy, qz, qw)
    """
    batch_size = positions.shape[0]
    device = positions.device
    
    # 获取up向量
    up_vector = get_up_vector(up_axis)
    up = torch.tensor(up_vector, dtype=torch.float32).to(device).unsqueeze(0).expand(batch_size, -1)
    
    # 目标点（看向原点）
    at = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)
    
    positions_float32 = positions.to(torch.float32)
    # 使用PyTorch3D的look_at_view_transform生成旋转矩阵
    R, T = look_at_view_transform(eye=positions_float32, at=at, up=up)
    
    # 确保旋转矩阵在正确的设备上
    R = R.to(device)
    
    # 将旋转矩阵转换为四元数
    quaternions_wxyz = matrix_to_quaternion(R)
    # 转换为xyzw格式
    quaternions_xyzw = torch.stack([
        quaternions_wxyz[:, 1],  # x
        quaternions_wxyz[:, 2],  # y
        quaternions_wxyz[:, 3],  # z
        quaternions_wxyz[:, 0]   # w
    ], dim=1)
    
    # 拼接位置和四元数
    pose_tensor = torch.cat([positions, quaternions_xyzw], dim=1)
    
    return pose_tensor


def world_points_to_camera_depth(
    point_maps: torch.Tensor,
    camera_poses: torch.Tensor,
    *,
    valid_masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    将世界坐标系点云转换为相机坐标系 Z 深度。
    此版本完全遵循 PyTorch3D 的行向量约定，避免了不必要的转置。

    Args:
        point_maps: [..., 3] 世界坐标点，支持 [S, H, W, 3] 或 [B, S, H, W, 3]
        camera_poses: [..., 7] 相机位姿 (position xyz + quaternion qx,qy,qz,qw，一致为 W2C 旋转)
        valid_masks: 与 point_maps 前几维一致的有效像素掩码，缺省表示全部有效

    Returns:
        depth_z: 与 point_maps 同维度的深度张量，最后一维为 1
    """
    if point_maps.shape[-1] != 3:
        raise ValueError(f"point_maps last dim must be 3, got {point_maps.shape}")
    if camera_poses.shape[-1] != 7:
        raise ValueError(f"camera_poses last dim must be 7, got {camera_poses.shape}")

    is_batched = point_maps.ndim == 5 and camera_poses.ndim == 3
    if point_maps.ndim == 4 and camera_poses.ndim == 2:
        batch_views = 1
        views = point_maps.shape[0]
        height, width = point_maps.shape[1:3]
        points_flat = point_maps
        poses_flat = camera_poses
        masks_flat = valid_masks if valid_masks is not None else None
    elif is_batched:
        batch_views = point_maps.shape[0]
        views = point_maps.shape[1]
        height, width = point_maps.shape[2:4]
        points_flat = point_maps.reshape(batch_views * views, height, width, 3)
        poses_flat = camera_poses.reshape(batch_views * views, 7)
        if valid_masks is not None:
            masks_flat = valid_masks.reshape(batch_views * views, height, width)
        else:
            masks_flat = None
    else:
        raise ValueError(
            f"Unsupported shapes for point_maps {point_maps.shape} and camera_poses {camera_poses.shape}."
        )

    device = points_flat.device
    dtype = points_flat.dtype

    positions = poses_flat[:, :3].to(device=device, dtype=dtype)
    quaternions = poses_flat[:, 3:].to(device=device, dtype=dtype)
    quaternion_wxyz = torch.stack(
        (quaternions[:, 3], quaternions[:, 0], quaternions[:, 1], quaternions[:, 2]),
        dim=-1,
    )
    # quaternion_to_matrix 返回的是为行向量设计的矩阵，我们直接使用它
    rotation_w2c_row = quaternion_to_matrix(quaternion_wxyz)  # [N, 3, 3]

    points_vec = points_flat.view(points_flat.shape[0], -1, 3) # [N, H*W, 3]
    relative = points_vec - positions.unsqueeze(1) # [N, H*W, 3], 这就是一批行向量

    camera_points = torch.bmm(relative, rotation_w2c_row)

    depth = camera_points[..., 2:3].view(points_flat.shape[0], height, width, 1)

    # --- 掩码和恢复形状部分，完全正确，无需修改 ---
    if masks_flat is not None:
        mask = masks_flat.to(device=device).unsqueeze(-1)
        depth = depth.masked_fill(~mask, 0.0)

    depth = depth.to(dtype=torch.float32)

    if point_maps.ndim == 4:
        return depth

    depth = depth.view(batch_views, views, height, width, 1)
    return depth
