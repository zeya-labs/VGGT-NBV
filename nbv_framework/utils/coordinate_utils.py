"""
坐标变换工具模块
处理不同坐标系统之间的转换和变换
"""

import numpy as np
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


def generate_fibonacci_sphere_points(num_points: int, radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成斐波那契球面分布的点
    
    Args:
        num_points: 点的数量
        radius: 球面半径
        
    Returns:
        positions: 球面上的点位置 [num_points, 3]
        directions: 从原点到各点的方向向量 [num_points, 3]
    """
    import math
    
    positions = []
    directions = []
    
    for i in range(num_points):
        # 使用斐波那契球面分布
        phi = math.acos(1 - 2 * (i + 0.5) / num_points)
        theta = 2 * math.pi * (i + 0.5) / (1 + math.sqrt(5)) * 2
        
        # 计算球面坐标
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        
        positions.append([x, y, z])
        directions.append([x/radius, y/radius, z/radius])
    
    return np.array(positions), np.array(directions)






