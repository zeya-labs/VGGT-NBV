"""
Mesh 工具函数

提供 mesh 的加载、归一化与点采样功能。
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

import torch
from pytorch3d.structures import Meshes, join_meshes_as_batch
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

        verts_list = [v.to(dtype=torch.float32, device=cpu_device) for v in mesh.verts_list()]
        faces_list = mesh.faces_list()

        # 如果纹理加载失败，创建默认的白色纹理
        if mesh.textures is None:
            raise ValueError(f"Mesh {mesh_path} 没有加载到纹理信息")

        # 重新创建mesh对象，确保顶点是float32
        mesh = Meshes(
            verts=verts_list,
            faces=faces_list,
            textures=mesh.textures
        )

    elif mesh_path.endswith('.ply'):
        # 1. 一次性加载并转换类型（load_ply 通常返回的是 float32 或 double）
        verts, faces = load_ply(mesh_path)
        verts = verts.to(dtype=torch.float32) # 只转类型，设备默认就是 CPU
        
        # 2. 构造纹理：直接在创建时处理维度
        # 使用 unsqueeze(0) 比 [None] 语义更清晰
        # 确保颜色也是 float32
        textures = TexturesVertex(verts_features=torch.ones_like(verts).unsqueeze(0))

        # 3. 直接组装，不要再写 .to(cpu_device)
        mesh = Meshes(
            verts=[verts], 
            faces=[faces.to(torch.int64)], # 确保 faces 是 long 类型
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
            scale = torch.quantile(distances, q=0.95)
        elif method == 'centered':
            pass
        else:
            raise ValueError(f"不支持的归一化方法: {method}")

        # 应用缩放
        eps = 1e-8
        scale = torch.clamp(scale, min=eps)
        verts_normalized = verts_centered / scale
        
        new_verts_list.append(verts_normalized) 

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


def load_and_normalize_mesh_to_device(
    mesh_path: str,
    normalize_method: Optional[str],
    device: torch.device,
) -> Meshes:
    method = str(normalize_method) if normalize_method else "mean"
    mesh = normalize_mesh(load_mesh_as_pytorch3d(mesh_path), method)
    return mesh.to(device)


def load_and_normalize_mesh_cpu(
    mesh_path: str,
    normalize_method: Optional[str],
) -> Meshes:
    method = str(normalize_method) if normalize_method else "mean"
    return normalize_mesh(load_mesh_as_pytorch3d(mesh_path), method)


def load_meshes_as_batch(
    mesh_paths: Sequence[Optional[str]],
    normalize_methods: Optional[Sequence[Optional[str]]],
    device: torch.device,
    num_workers: int = 0,
) -> Meshes:
    use_parallel = num_workers > 1 and len(mesh_paths) > 1
    if use_parallel:
        max_workers = min(num_workers, len(mesh_paths))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(load_and_normalize_mesh_cpu, mesh_path, normalize_method)
                for mesh_path, normalize_method in zip(mesh_paths, normalize_methods)
            ]
            meshes = [future.result().to(device) for future in futures]
    else:
        meshes = [
            load_and_normalize_mesh_to_device(mesh_path, normalize_method, device)
            for mesh_path, normalize_method in zip(mesh_paths, normalize_methods)
        ]
    return join_meshes_as_batch(meshes)
