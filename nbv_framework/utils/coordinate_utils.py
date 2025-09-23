"""
坐标变换工具模块
处理不同坐标系统之间的转换和变换
"""

import numpy as np
import math
from typing import Tuple


def get_up_vector(up_axis: str) -> np.ndarray:
    """
    根据up_axis参数获取上朝向向量
    
    Args:
        up_axis: 上朝向轴，可选 "X", "Y", "Z", "-X", "-Y", "-Z"
        
    Returns:
        up_vector: 上朝向向量 [3]
    """
    up_axis = up_axis.upper()
    up_vectors = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
        "-X": np.array([-1.0, 0.0, 0.0]),
        "-Y": np.array([0.0, -1.0, 0.0]),
        "-Z": np.array([0.0, 0.0, -1.0])
    }
    
    if up_axis not in up_vectors:
        raise ValueError(f"Invalid up_axis: {up_axis}. Must be one of: X, Y, Z, -X, -Y, -Z")
    
    return up_vectors[up_axis]


def get_coordinate_transform_matrix(up_axis: str) -> np.ndarray:
    up_axis = up_axis.upper()
    
    transform_matrices = {
        "Y": np.eye(4),  # Y-up to Y-up: 无需变换
        
        # Y-up to Z-up: 绕X轴旋转-90度
        "Z": np.array([
            [1,  0,  0, 0],
            [0,  0, -1, 0],  # y -> -z
            [0,  1,  0, 0],  # z -> y
            [0,  0,  0, 1]
        ]),
        
        # Y-up to X-up: 绕Z轴旋转90度
        "X": np.array([
            [0, -1, 0, 0],   # x -> -y
            [1,  0, 0, 0],   # y -> x
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ]),
        
        # Y-up to -Y-up: 绕Z轴旋转180度
        "-Y": np.array([
            [-1, 0,  0, 0],
            [0, -1,  0, 0],
            [0,  0,  1, 0],
            [0,  0,  0, 1]
        ]),
        
        # Y-up to -Z-up: 绕X轴旋转90度
        "-Z": np.array([
            [1,  0,  0, 0],
            [0,  0,  1, 0],  # y -> z
            [0, -1,  0, 0],  # z -> -y
            [0,  0,  0, 1]
        ]),
        
        # Y-up to -X-up: 绕Z轴旋转-90度
        "-X": np.array([
            [0,  1, 0, 0],   # x -> y
            [-1, 0, 0, 0],   # y -> -x
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])
    }
    
    if up_axis not in transform_matrices:
        raise ValueError(f"Invalid up_axis: {up_axis}")
    
    return transform_matrices[up_axis]


def apply_transform_to_vertices(vertices: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    对顶点应用变换矩阵
    
    Args:
        vertices: 顶点数组 [N, 3]
        transform_matrix: 4x4变换矩阵
        
    Returns:
        transformed_vertices: 变换后的顶点数组 [N, 3]
    """
    # 添加齐次坐标
    homogeneous_vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    
    # 应用变换
    transformed_homogeneous = (transform_matrix @ homogeneous_vertices.T).T
    
    # 返回前3列（去掉齐次坐标）
    return transformed_homogeneous[:, :3]


def generate_fibonacci_sphere_points(
    num_points: int, radius: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成斐波那契球面分布的点
    
    Args:
        num_points: 点的数量
        radius: 球面半径
        
    Returns:
        positions: 球面上的点位置 [num_points, 3]
        directions: 从原点到各点的方向向量 [num_points, 3]
    """
    
    # 创建一个从 0.5 到 num_points - 0.5 的数组
    i = np.arange(num_points, dtype=float) + 0.5
    
    # 黄金比例
    golden_ratio = (1 + math.sqrt(5)) / 2
    
    # 计算所有点的 phi 和 theta
    phi = np.arccos(1 - 2 * i / num_points)
    theta = 2 * math.pi * i / golden_ratio
    
    # 一次性计算所有点的球面坐标
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    # 将坐标堆叠成 [num_points, 3] 的数组
    positions = np.stack([x, y, z], axis=-1)
    
    # 方向向量是位置向量的归一化
    # 由于我们是从单位球生成的，除以半径即可
    directions = positions / radius
    
    return positions, directions


def generate_fibonacci_upper_hemisphere_points(
    num_points: int, radius: float = 1.0, up_axis: str = "Y"
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成斐波那契上半球面分布的点
    
    Args:
        num_points: 点的数量
        radius: 半球半径
        up_axis: 上朝向轴，可选 "X", "Y", "Z", "-X", "-Y", "-Z"
        
    Returns:
        positions: 上半球面上的点位置 [num_points, 3]
        directions: 从原点到各点的方向向量 [num_points, 3]
    """
    import math
    
    positions = []
    directions = []
    
    i = np.arange(num_points, dtype=float) + 0.5
    golden_ratio = (1 + math.sqrt(5)) / 2
    
    # 垂直分量
    z_coord = 1 - i / num_points
    phi = np.arccos(z_coord)
    
    # 黄金螺旋角度
    theta = 2 * math.pi * i / golden_ratio
    
    # 标准 Z-up 球面坐标
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * z_coord
    
    # 根据up_axis调整坐标系
    up = up_axis.upper()
    if up == "Y":
        positions = np.stack([x, z, y], axis=-1)
    elif up == "Z":
        positions = np.stack([x, y, z], axis=-1)
    elif up == "X":
        positions = np.stack([z, y, x], axis=-1)
    elif up == "-Y":
        positions = np.stack([x, -z, y], axis=-1)
    elif up == "-Z":
        positions = np.stack([x, y, -z], axis=-1)
    elif up == "-X":
        positions = np.stack([-z, y, x], axis=-1)
    else: # 默认 Z 轴向上
        positions = np.stack([x, y, z], axis=-1)
        
    directions = positions / radius
    
    return positions, directions