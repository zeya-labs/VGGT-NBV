from typing import Tuple, Optional
import torch
from pytorch3d.structures import Meshes

from ..rendering import DifferentiableRenderer

def render_gt_point_maps(
    renderer: DifferentiableRenderer,
    mesh_batch: Meshes,
    camera_poses: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    可微渲染点云图与掩码。
    
    Args:
        renderer: 可微渲染器实例.
        mesh_batch: Meshes 对象，长度为 B.
        camera_poses: 相机位姿，支持 [B, S, 7] 或 [B*S, 7].
        device: 计算设备 (必须是 GPU 以支持渲染).

    Returns:
        point_maps: [B, S, H, W, 3] (带梯度)
        valid_masks: [B, S, H, W]   (通常无梯度，视光栅化器实现而定)
    """
    if device is None:
        if hasattr(renderer, "device"):
            device = renderer.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 维度检查与预处理
    batch_size = len(mesh_batch)
    
    # 期望 camera_poses 为 [B, S, 7]
    if camera_poses.dim() == 2:
        # 如果输入是 [B, 7]，假设 S=1 -> [B, 1, 7]
        if camera_poses.shape[0] == batch_size:
            camera_poses = camera_poses.unsqueeze(1)
        # 如果输入是 [S, 7]，且 B=1 -> [1, S, 7]
        elif batch_size == 1:
            camera_poses = camera_poses.unsqueeze(0)
        else:
            # 无法推断，可能是已经 Flatten 过的 [B*S, 7]，但为了安全建议传入 [B, S, 7]
             raise ValueError(f"Ambiguous camera_poses shape: {camera_poses.shape} for batch size {batch_size}")

    B, S, _ = camera_poses.shape

    # 2. 构建大 Batch 进行并行渲染 (Batching Strategy)
    # 我们需要将 B 个 Mesh 和 B*S 个相机对应起来。
    # 目标：渲染 B*S 张图。
    # 策略：将 Mesh 复制扩展，使其长度为 B*S，顺序为 [M0, M0... M1, M1...]
    
    # 展平相机位姿: [B, S, 7] -> [B*S, 7]
    cameras_flat = camera_poses.reshape(B * S, -1)
    
    # 扩展 Meshes: 使用索引复制，比深拷贝更省内存
    # 索引序列: [0, 0, ..(S次).., 1, 1, ..(S次).., ...]
    mesh_indices = torch.arange(B, device=device).repeat_interleave(S)
    
    # PyTorch3D 支持通过索引切片来构建新的 Meshes 对象 (只是引用，速度快)
    mesh_batch_flat = mesh_batch[mesh_indices]

    # 3. 执行渲染 (Differentiable Rendering)
    # 此时 renderer 接收的是 B*S 大小的 Mesh 和 Camera
    render_out = renderer(
        gt_mesh=mesh_batch_flat,
        camera_poses=cameras_flat,
        out_rgb=False,
        out_points=True,
        out_mask=True
    )

    point_maps_flat, valid_masks_flat = render_out["points"], render_out["mask"]
    
    # point_maps_flat: [B*S, 3, H, W] (取决于 renderer 配置，通常 PyTorch3D 输出是 NCHW 或 NHWC)
    # 注意：PyTorch3D 的 MeshRasterizer 默认输出通常是 (N, H, W, K, C) 或 (N, H, W, C)
    # 这里假设你的 renderer 包装器返回的是 Tensor。
    
    # 确保维度符合预期，假设 renderer 输出为 [B*S, C, H, W] 或 [B*S, H, W, C]
    # 我们统一调整为 [B*S, H, W, 3]
    point_maps_flat = point_maps_flat.permute(0, 2, 3, 1)
        
    # valid_masks_flat 通常是 [B*S, 1, H, W] 或 [B*S, H, W]
    if valid_masks_flat.dim() == 4:
        valid_masks_flat = valid_masks_flat.squeeze(1)

    H, W = point_maps_flat.shape[1:3]
    # 4. 恢复维度 [B*S, ...] -> [B, S, ...]
    point_maps = point_maps_flat.view(B, S, H, W, 3).to(device=output_device)
    valid_masks = valid_masks_flat.view(B, S, H, W).to(device=output_device)

    # 5. 返回结果
    return point_maps, valid_masks