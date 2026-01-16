"""
可微分渲染器 (Refactored)

使用PyTorch3D实现可微分渲染。
- 质量锁定为 Medium (抗锯齿)
- 光照锁定为 Ambient (环境光)
- 支持灵活输出 RGB / Depth / Point Cloud / Mask
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Union
import math
import warnings
import torch.nn.functional as F

from loguru import logger

try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        FoVPerspectiveCameras, RasterizationSettings, 
        MeshRasterizer, SoftPhongShader, PointLights, BlendParams,
        Materials
    )
    from pytorch3d.renderer.mesh.utils import interpolate_face_attributes
    from pytorch3d.transforms import quaternion_to_matrix
    PYTORCH3D_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch3D not available.")
    PYTORCH3D_AVAILABLE = False


class DifferentiableRenderer(nn.Module):
    def __init__(self, 
                 image_size: int = 518,
                 fov: float = 60.0):
        """
        初始化可微分渲染器
        """
        super().__init__()
        
        if not PYTORCH3D_AVAILABLE:
            raise ImportError("PyTorch3D is required.")
        
        self.image_size = image_size
        self.default_fov = fov
        
        # --- 1. 渲染设置 ---
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=1e-4, 
            faces_per_pixel=8, 
            perspective_correct=False,
        )
        
        # --- 2. 材质与混合参数 ---
        self.blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))
        self.materials = Materials(shininess=128.0)
        
        # --- 3. 组件初始化 ---
        # 光栅化器 (负责几何投影)
        self.rasterizer = MeshRasterizer(raster_settings=self.raster_settings)
        # 着色器 (负责颜色计算)
        self.shader = SoftPhongShader(
            materials=self.materials,
            blend_params=self.blend_params
        )

    def _get_cameras(self, camera_poses: torch.Tensor, pose_format: str, fov: float):
        """内部帮助函数：构建相机对象"""
        device = camera_poses.device
        # 1. 数据清洗与转换
        camera_poses = camera_poses.float()
        
        if pose_format == "spherical":
            # 球坐标 -> 笛卡尔坐标
            spherical = camera_poses[:, :3] # theta, phi, radius
            quats = camera_poses[:, 3:7]    # qx, qy, qz, qw
            
            theta, phi, radius = spherical[:, 0], spherical[:, 1], spherical[:, 2]
            x = radius * torch.sin(phi) * torch.cos(theta)
            y = radius * torch.sin(phi) * torch.sin(theta)
            z = radius * torch.cos(phi)
            pos_xyz = torch.stack([x, y, z], dim=1)
            
            # 重组为 cartesian 格式以便统一处理
            camera_poses = torch.cat([pos_xyz, quats], dim=1)

        # 2. 计算 R, T
        positions = camera_poses[:, :3]
        quaternions = camera_poses[:, 3:7] # qx, qy, qz, qw

        # 归一化四元数
        quat_norm = quaternions.norm(dim=1, keepdim=True)
        quaternions = quaternions / quat_norm.clamp_min(1e-6)

        # PyTorch3D 需要 (w, x, y, z) 顺序
        q_wxyz = quaternions[:, [3, 0, 1, 2]]
        R = quaternion_to_matrix(q_wxyz)
        
        # 处理旋转矩阵奇异值 (可选，保留原逻辑)
        with torch.no_grad():
             if (torch.linalg.det(R).abs() < 1e-3).any():
                 warnings.warn("Detected singular rotation, skipping fix for simplicity in refactor.")

        # T = -C @ R
        T = -(positions.unsqueeze(1) @ R).squeeze(1)

        return FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)

    def forward(self, 
                gt_mesh: Meshes,
                camera_poses: torch.Tensor,
                pose_format: str = "cartesian",
                fov: float = 60.0,
                out_rgb: bool = True,
                out_depth: bool = False,
                out_points: bool = False,
                out_mask: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            gt_mesh: 目标 Mesh
            camera_poses: [B, 7] 相机位姿
            pose_format: "cartesian" or "spherical"
            out_*: 布尔值，控制输出内容
            
        Returns:
            Dict containing keys: 'rgb', 'depth', 'points', 'mask' based on requests.
            All tensors are [B, C, H, W].
        """
        # PyTorch3D requires meshes / cameras / lights / materials to be on the same device.
        device = gt_mesh.device
        if camera_poses.device != device:
            camera_poses = camera_poses.to(device)

        # Move shader-side tensor properties (materials/lights/cameras) to the render device.
        # This fixes device mismatch errors when the renderer module was constructed on CPU
        # but later used on GPU (or vice versa).
        self.shader.to(device)
        fov = fov if fov is not None else self.default_fov
        
        with torch.amp.autocast(device_type=device.type, enabled=False):
            # 1. 创建相机
            cameras = self._get_cameras(camera_poses, pose_format, fov)
            
            # 2. 光栅化
            # fragments 包含: pix_to_face, zbuf, bary_coords, dists
            # shape: [B, H, W, K], K=8 (Medium settings)
            fragments = self.rasterizer(gt_mesh, cameras=cameras)
            
            outputs = {}
            
            # --- Mask 计算 ---
            # pix_to_face >= 0 表示击中面。取 K=0 (最近面) 判断 Mask 即可
            valid_mask = (fragments.pix_to_face[..., 0] >= 0).float().unsqueeze(1) # [B, 1, H, W]
            if out_mask:
                outputs['mask'] = valid_mask.bool()

            # --- RGB 计算 (纯环境光) ---
            if out_rgb:
                # 构造纯白环境光，无镜面反射
                lights = PointLights(device=device, ambient_color=((1.0, 1.0, 1.0),))
                # Shader 会利用所有 K=8 个面进行加权混合 (抗锯齿)
                images = self.shader(fragments, gt_mesh, cameras=cameras, lights=lights)
                # [B, H, W, 4] -> [B, 3, H, W]
                outputs['rgb'] = images.permute(0, 3, 1, 2)[:, :3, :, :]

            # --- Depth 计算 ---
            if out_depth:
                # 直接取 Z-buffer 的最近面 (K=0)
                # [B, H, W] -> [B, 1, H, W]
                depth_map = fragments.zbuf[..., 0].unsqueeze(1)
                # 背景处的 zbuf 通常为 -1，这里可以通过 mask 过滤，或者保持原值
                # 为了保持张量纯净，这里直接返回 zbuf，用户可结合 mask 使用
                outputs['depth'] = depth_map

            # --- Point Cloud (XYZ Map) 计算 ---
            if out_points:
                # 将像素对应的重心坐标插值回世界坐标
                face_vertices = gt_mesh.verts_packed()[gt_mesh.faces_packed()] # (F, 3, 3)
                interpolated = interpolate_face_attributes(
                    fragments.pix_to_face, fragments.bary_coords, face_vertices
                ) # (B, H, W, K, 3)
                
                # 取最近面 (K=0) 的坐标
                point_map = interpolated[..., 0, :] 
                # [B, H, W, 3] -> [B, 3, H, W]
                outputs['points'] = point_map.permute(0, 3, 1, 2)

            return outputs
