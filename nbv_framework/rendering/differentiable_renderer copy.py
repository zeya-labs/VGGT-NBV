"""
可微分渲染器 (Fully Differentiable Optimized)

主要改进：
1. 全面移除硬截断 (Hard Selection)，改为软加权 (Soft Blending)。
2. Depth 和 Point Cloud 使用 Softmax Aggregation，支持遮挡梯度的反向传播。
3. Mask 使用 Sigmoid Probability，边缘可微。
"""

import torch
import torch.nn as nn
from typing import Dict
import warnings

from loguru import logger

try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        FoVPerspectiveCameras, RasterizationSettings, 
        MeshRasterizer, SoftPhongShader, PointLights, 
        BlendParams, Materials
    )
    from pytorch3d.renderer.blending import softmax_rgb_blend, sigmoid_alpha_blend
    from pytorch3d.renderer.mesh.utils import interpolate_face_attributes
    from pytorch3d.transforms import quaternion_to_matrix
    PYTORCH3D_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch3D not available.")
    PYTORCH3D_AVAILABLE = False


class DifferentiableRenderer(nn.Module):
    def __init__(self, 
                 image_size: int = 518,
                 device: str = "cuda",
                 fov: float = 60.0,
                 # 关键参数：控制可微性的平滑程度
                 blur_radius: float = 1e-4, 
                 faces_per_pixel: int = 15,  # 增加 K 值以获得更好的遮挡梯度
                 sigma: float = 1e-4, 
                 gamma: float = 1e-4):
        """
        初始化全可导渲染器
        
        Args:
            blur_radius: 光栅化时的模糊半径。非0值是产生梯度的关键。
            faces_per_pixel (K): 每个像素保留的面数。K越大，遮挡边缘的梯度越精确，但显存消耗增加。
            sigma: 混合函数的平滑参数。
            gamma: 混合函数的透明度参数。
        """
        super().__init__()
        
        if not PYTORCH3D_AVAILABLE:
            raise ImportError("PyTorch3D is required.")
        
        self.image_size = image_size
        self.device = torch.device(device)
        self.default_fov = fov
        
        # --- 1. 渲染设置 (Rasterization) ---
        # 必须设置 blur_radius > 0 才能在边缘产生梯度
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=blur_radius, 
            faces_per_pixel=faces_per_pixel, 
            perspective_correct=False,
            clip_barycentric_coords=True, # 建议开启，防止重心坐标越界
            z_clip_value=0.1
        )
        
        # --- 2. 混合参数 (Blending) ---
        # 统一管理混合参数，确保 RGB/Depth/Mask 的梯度行为一致
        self.blend_params = BlendParams(
            sigma=sigma, 
            gamma=gamma, 
            background_color=(0.0, 0.0, 0.0)
        )
        
        self.materials = Materials(device=self.device, shininess=128.0)
        
        # --- 3. 组件初始化 ---
        self.rasterizer = MeshRasterizer(raster_settings=self.raster_settings)
        
        # RGB 使用 SoftPhongShader (内部包含了光照计算和 Softmax Blending)
        self.rgb_shader = SoftPhongShader(
            device=self.device,
            materials=self.materials,
            blend_params=self.blend_params
        )

    def _get_cameras(self, camera_poses: torch.Tensor, pose_format: str, fov: float):
        """构建相机对象 (保持原逻辑)"""
        camera_poses = camera_poses.float()
        
        if pose_format == "spherical":
            spherical = camera_poses[:, :3]
            quats = camera_poses[:, 3:7]
            theta, phi, radius = spherical[:, 0], spherical[:, 1], spherical[:, 2]
            x = radius * torch.sin(phi) * torch.cos(theta)
            y = radius * torch.sin(phi) * torch.sin(theta)
            z = radius * torch.cos(phi)
            pos_xyz = torch.stack([x, y, z], dim=1)
            camera_poses = torch.cat([pos_xyz, quats], dim=1)

        positions = camera_poses[:, :3]
        quaternions = camera_poses[:, 3:7]
        
        # 归一化四元数
        quat_norm = quaternions.norm(dim=1, keepdim=True)
        quaternions = quaternions / quat_norm.clamp_min(1e-6)
        
        q_wxyz = quaternions[:, [3, 0, 1, 2]]
        R = quaternion_to_matrix(q_wxyz)
        T = -(positions.unsqueeze(1) @ R).squeeze(1)

        return FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=fov)

    def forward(self, 
                gt_mesh: Meshes,
                camera_poses: torch.Tensor,
                pose_format: str = "cartesian",
                fov: float = 60.0,
                out_rgb: bool = True,
                out_depth: bool = False,
                out_points: bool = False,
                out_mask: bool = False) -> Dict[str, torch.Tensor]:
        
        fov = fov if fov is not None else self.default_fov
        
        # 强制 Float32
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            gt_mesh = gt_mesh.to(self.device)
            camera_poses = camera_poses.to(self.device, dtype=torch.float32)
            
            # 1. 相机投影
            cameras = self._get_cameras(camera_poses, pose_format, fov)
            
            # 2. 光栅化 (Rasterization)
            fragments = self.rasterizer(gt_mesh, cameras=cameras)
            
            outputs = {}
            
            # --- Mask 计算 (Fully Differentiable) ---
            # 修复点：构造 dummy_colors 传给 sigmoid_alpha_blend
            if out_mask:
                # 获取维度信息 [B, H, W, K]
                N, H, W, K = fragments.pix_to_face.shape
                # 构造虚拟颜色 (全白), 形状需要是 [B, H, W, K, 3]
                dummy_colors = torch.ones((N, H, W, K, 3), device=self.device)
                
                # 计算 RGBA，取最后一个通道作为 Alpha
                # sigmoid_alpha_blend 会利用 fragments.dists 计算软边缘
                soft_rgba = sigmoid_alpha_blend(dummy_colors, fragments, self.blend_params)
                
                # soft_rgba 是 [B, H, W, 4], 取 alpha 通道 -> [B, H, W] -> [B, 1, H, W]
                outputs['mask'] = soft_rgba[..., 3].unsqueeze(1).permute(0, 1, 2, 3) > 0.5

            # --- RGB 计算 ---
            if out_rgb:
                lights = PointLights(device=self.device, ambient_color=((1.0, 1.0, 1.0),))
                rgba = self.rgb_shader(fragments, gt_mesh, cameras=cameras, lights=lights)
                outputs['rgb'] = rgba.permute(0, 3, 1, 2)[:, :3, :, :]

            # --- Point Cloud / XYZ Map 计算 ---
            if out_points:
                face_vertices = gt_mesh.verts_packed()[gt_mesh.faces_packed()]
                interpolated_xyz = interpolate_face_attributes(
                    fragments.pix_to_face, fragments.bary_coords, face_vertices
                )
                # softmax_rgb_blend 返回 [B, H, W, 4] (最后一位是 Alpha)
                soft_xyz = softmax_rgb_blend(
                    interpolated_xyz, fragments, self.blend_params
                )
                # 修复：取前3个通道 [:, :3, :, :]
                outputs['points'] = soft_xyz.permute(0, 3, 1, 2)[:, :3, :, :]

            # --- Depth 计算 ---
            if out_depth:
                z_values = fragments.zbuf
                # softmax_rgb_blend 返回 [B, H, W, 2] (Depth + Alpha)
                soft_depth = softmax_rgb_blend(
                    z_values.unsqueeze(-1), fragments, self.blend_params
                )
                # 修复：取第1个通道 [:, :1, :, :]
                outputs['depth'] = soft_depth.permute(0, 3, 1, 2)[:, :1, :, :]
            # print(outputs)
            return outputs