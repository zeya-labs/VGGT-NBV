"""
Mesh 工具函数

提供 mesh 的加载、归一化与点采样功能。
"""

from typing import Dict, List

import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes, load_ply
from pytorch3d.ops import sample_points_from_meshes


from pytorch3d.renderer import TexturesVertex

def load_mesh_as_pytorch3d(mesh_path: str) -> Meshes:
    """
    加载 mesh 为 PyTorch3D Meshes 对象，支持 .obj 与 .ply。
    确保所有加载操作都在 CPU 上进行，并正确加载纹理。
    如果纹理缺失，应用默认白色纹理。
    """
    cpu_device = torch.device("cpu")

    if mesh_path.endswith('.obj'):
        mesh = load_objs_as_meshes(
            [mesh_path],
            device=cpu_device,
            load_textures=True
        )

        # 优化：立即将顶点转换为float32
        verts_f32 = mesh.verts_packed().to(torch.float32)

        # 如果纹理加载失败，创建默认的白色纹理
        if mesh.textures is None:
            raise ValueError(f"Mesh {mesh_path} 没有加载到纹理信息")

        # 重新创建mesh对象，确保顶点是float32
        mesh = Meshes(
            verts=[verts_f32.to(cpu_device)],
            faces=mesh.faces_list(),
            textures=mesh.textures
        )

    elif mesh_path.endswith('.ply'):
        verts, faces = load_ply(mesh_path)

        # 为 .ply 文件创建一个默认的纯白顶点颜色纹理
        # 这是为了满足 SoftPhongShader 的要求
        # 优化：确保顶点是float32类型
        verts = verts.to(cpu_device, dtype=torch.float32)
        faces = faces.to(cpu_device)
        verts_rgb = torch.ones_like(verts, dtype=torch.float32)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(cpu_device))

        # 创建 Meshes 对象，这次包含了纹理信息
        mesh = Meshes(
            verts=[verts.to(cpu_device)],
            faces=[faces.to(cpu_device)],
            textures=textures
        )
    else:
        raise ValueError(f"Unsupported file type: {mesh_path}. Only .obj and .ply are supported.")

    return mesh


def normalize_mesh(mesh: Meshes, method: str = "quantile") -> Meshes:
    """
    对 mesh 进行归一化处理（支持 Batch）。
    """
    if method == 'none':
        return mesh.clone()

    # 获取顶点列表（将 Batch 拆分为单独的 Tensor 列表）
    # 这样我们可以针对每个 mesh 独立计算 mean 和 scale
    verts_list = mesh.verts_list()
    new_verts_list: List[torch.Tensor] = []

    for verts in verts_list:
        # 1. 中心化 (Centering)
        centroid = verts.mean(dim=0)
        verts_centered = verts - centroid

        # 2. 计算缩放因子 (Scaling)
        scale = 1.0
        # 计算距离 (N,)
        distances = torch.norm(verts_centered, p=2, dim=1)

        if method == 'unit_sphere':
            scale = distances.max()
        elif method == 'unit_cube':
            scale = torch.abs(verts_centered).max()
        elif method == 'std':
            scale = torch.sqrt(torch.mean(distances ** 2))
        elif method in {'mean', 'mean_radius'}:
            scale = torch.mean(distances)
        elif method == 'quantile':
            # 95% 分位数，去除离群点影响
            scale = torch.quantile(distances, q=0.95)
        elif method == 'centered':
            pass
        else:
            raise ValueError(f"不支持的归一化方法: {method}")

        # 应用缩放
        if scale > 1e-8 and method != 'centered':
            verts_centered = verts_centered / scale
        
        new_verts_list.append(verts_centered)

    # 3. 重建 Meshes 对象
    # 使用原始的 faces 和 textures，但使用新的顶点位置
    # 注意：这里我们使用 mesh.faces_list() 来保持 batch 结构
    new_mesh = Meshes(
        verts=new_verts_list,
        faces=mesh.faces_list(),
        textures=mesh.textures
    )

    return new_mesh


def load_and_normalize_mesh(
    mesh_path: str,
    normalize_method: str = "quantile",
    num_samples: int = 10000,
) -> Dict[str, torch.Tensor]:
    """
    加载 mesh，执行归一化，并从表面采样点云。
    """
    mesh = load_mesh_as_pytorch3d(mesh_path)
    normalized_mesh = normalize_mesh(mesh, normalize_method)

    # # 输出mesh点距离原点的范围
    # vertices = normalized_mesh.verts_packed()  # 获取所有顶点坐标
    # distances = torch.norm(vertices, dim=1)  # 计算每个点到原点的距离
    # min_dist = distances.min().item()
    # max_dist = distances.max().item()
    # mean_dist = distances.mean().item()
    # print(f"Mesh顶点距离原点范围: 最小={min_dist:.4f}, 最大={max_dist:.4f}, 平均={mean_dist:.4f}")

    sampled_points = sample_points_from_meshes(normalized_mesh, num_samples=num_samples)

    return {
        "mesh_path": mesh_path,
        "original_mesh": mesh,
        "normalized_mesh": normalized_mesh,
        # "vertices": normalized_mesh.verts_list()[0],
        # "faces": normalized_mesh.faces_list()[0],
        "gt_points": sampled_points.squeeze(0),
        "normalize_method": normalize_method,
        "num_samples": num_samples,
    }
