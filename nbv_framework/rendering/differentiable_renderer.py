"""
可微分渲染器

使用PyTorch3D实现可微分渲染，从给定的相机位姿和GT mesh生成新的观测图像。
该模块是训练框架的关键组件，用于模拟环境交互。
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Union
import numpy as np
import os
import math
import warnings
import torch.nn.functional as F

from nbv_framework.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

try:
    from pytorch3d.structures import Meshes, join_meshes_as_batch
    from pytorch3d.renderer import (
        FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRendererWithFragments,
        MeshRasterizer, SoftPhongShader, PointLights, BlendParams,
        look_at_view_transform, HardPhongShader, Materials
    )
    from pytorch3d.renderer.mesh import TexturesVertex
    from pytorch3d.renderer.mesh.utils import interpolate_face_attributes
    from pytorch3d.io import load_objs_as_meshes, load_ply
    from pytorch3d.transforms import quaternion_to_matrix
    PYTORCH3D_AVAILABLE = True
except ImportError:
    LOGGER.warning("PyTorch3D not available. Please install it for rendering functionality.")
    PYTORCH3D_AVAILABLE = False

class DifferentiableRenderer(nn.Module):
    """
    可微分渲染器
    
    该类封装了PyTorch3D的渲染功能，提供以下核心功能：
    1. 从相机位姿和GT mesh渲染新视图
    2. 支持批量渲染
    3. 可微分操作，支持梯度回传
    4. 支持两种位姿格式：
       - spherical: [B, 7] (theta, phi, radius, qx, qy, qz, qw)
       - cartesian: [B, 7] (x, y, z, qx, qy, qz, qw)
    """
    
    def __init__(self, 
                 image_size: int = 224,
                 device: str = "cuda",
                 quality: str = "high",
                 downsample_factor: int = 2):
        """
        初始化可微分渲染器
        
        Args:
            image_size: 渲染图像尺寸
            device: 计算设备
            quality: 渲染质量 ("low", "medium", "high")
            downsample_factor: high质量模式下的超采样倍数
        """
        super().__init__()
        
        if not PYTORCH3D_AVAILABLE:
            raise ImportError("PyTorch3D is required for DifferentiableRenderer")
        
        self.image_size = image_size
        self.device = torch.device(device)
        self.quality = quality
        self.downsample_factor = downsample_factor
        
        # 设置混合参数
        self.blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))
        
        # 设置材质
        self.materials = Materials(
            device=self.device,
            shininess=128.0
        )
        
        # 设置基础光照配置（将在渲染时动态创建）
        self.light_properties = None
        self.base_ambient_color = (0.2, 0.2, 0.2)
        
        # 根据质量设置渲染参数
        if quality == "low":
            LOGGER.info("使用 'low' 质量设置 (快速, 无抗锯齿)")
            self.render_image_size = image_size
            self.raster_settings = RasterizationSettings(
                image_size=self.render_image_size,
                blur_radius=0.0,
                faces_per_pixel=1,
            )
        elif quality == "medium":
            LOGGER.info("使用 'medium' 质量设置 (标准抗锯齿)")
            self.render_image_size = image_size
            self.raster_settings = RasterizationSettings(
                image_size=self.render_image_size,
                blur_radius=1e-4,
                faces_per_pixel=8,
            )
        else:  # high
            self.render_image_size = image_size * downsample_factor
            # print(f"🚀 使用 'high' 质量设置 (SSAA超采样抗锯齿 x{downsample_factor})")
            # print(f"   渲染分辨率: {self.render_image_size}x{self.render_image_size} -> 下采样到: {image_size}x{image_size}")
            self.raster_settings = RasterizationSettings(
                image_size=self.render_image_size,
                blur_radius=1e-5,
                faces_per_pixel=1,
            )
        
        # 创建渲染器 - 使用与render.py相同的架构
        self.shader = SoftPhongShader(
            device=self.device,
            materials=self.materials,
            blend_params=self.blend_params
        )
        
        self.renderer_with_frags = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(raster_settings=self.raster_settings),
            shader=self.shader
        )
    


    def pose_to_camera_matrices(self, camera_poses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从相机位姿（位置+四元数）计算PyTorch3D所需的旋转矩阵R和平移向量T。
        """
        # print(camera_poses)
        # 1. 分离位置和四元数，并修正可能的数值异常
        positions = camera_poses[:, :3]      # [B, 3] (x, y, z)
        quaternions = camera_poses[:, 3:7]  # [B, 4] (qx, qy, qz, qw)

        positions = torch.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
        quaternions = torch.nan_to_num(quaternions, nan=0.0, posinf=0.0, neginf=0.0)

        quat_norm = quaternions.norm(dim=1, keepdim=True)
        invalid_quat = quat_norm.squeeze(1) < 1e-6
        if invalid_quat.any():
            warnings.warn(
                "Received near-zero quaternion; substituting identity rotation to keep renderer stable.",
                RuntimeWarning,
                stacklevel=2,
            )
            fallback = quaternions.new_zeros(quaternions.shape)
            fallback[:, 3] = 1.0
            quaternions = torch.where(invalid_quat.unsqueeze(1), fallback, quaternions)
            quat_norm = quaternions.norm(dim=1, keepdim=True)

        quaternions = quaternions / quat_norm.clamp_min(1e-6)

        # 2. 转换四元数格式
        # PyTorch3D 的 quaternion_to_matrix 需要 (w, x, y, z) 格式
        # 我们的输入是 (x, y, z, w)，所以需要调整顺序
        q_wxyz = quaternions[:, [3, 0, 1, 2]]  # 从 [qx,qy,qz,qw] -> [qw,qx,qy,qz]
        R = quaternion_to_matrix(q_wxyz)  # 不需要 transpose

        with torch.no_grad():
            det_R = torch.linalg.det(R)
            near_singular = det_R.abs() < 1e-3
            if near_singular.any():
                warnings.warn(
                    "Detected near-singular rotation matrix; re-orthonormalizing via SVD.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                idx = near_singular.nonzero(as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    u, _, vh = torch.linalg.svd(R[idx])
                    # Enforce right-handed rotation; adjust if det is negative
                    det_uv = torch.det(torch.matmul(u, vh))
                    neg_det = det_uv < 0
                    if neg_det.any():
                        vh_adjusted = vh.clone()
                        vh_adjusted[neg_det, -1, :] *= -1.0
                        vh = vh_adjusted
                    R[idx] = torch.matmul(u, vh)

        # 关键：T = -C @ R（行向量约定，与PyTorch3D FoV相机接口一致）
        T = -(positions.unsqueeze(1) @ R).squeeze(1)

        return R, T

    def setup_lighting(self, lighting_type: str = "three_point"):
        """
        设置光照方案
        
        Args:
            lighting_type: 光照类型 ("three_point", "ambient", "reconstruction")
        """
        if lighting_type == "three_point":
            # 三点布光
            self.light_properties = [
                {
                    "location": [0.0, 2.0, 2.0],
                    "diffuse_color": (0.6, 0.6, 0.6),
                    "specular_color": (0.1, 0.1, 0.1),
                },
                {
                    "location": [0.0, -1.0, 1.0],
                    "diffuse_color": (0.3, 0.3, 0.3),
                    "specular_color": (0.0, 0.0, 0.0),
                },
                {
                    "location": [0.0, 0.0, -2.0],
                    "diffuse_color": (0.2, 0.2, 0.2),
                    "specular_color": (0.0, 0.0, 0.0),
                }
            ]
            self.base_ambient_color = (0.1, 0.1, 0.1)
            
        elif lighting_type == "ambient":
            # 纯环境光
            self.light_properties = []
            self.base_ambient_color = (1.0, 1.0, 1.0)
            
        elif lighting_type == "reconstruction":
            # 用于三维重建的理想光照
            self.light_properties = []
            # 添加多个顶部光源
            for i in range(8):
                angle = 2 * 3.14159 * i / 8
                x = 4.0 * math.cos(angle)
                y = 4.0 * math.sin(angle)
                self.light_properties.append({
                    "location": [x, y, 3.0],
                    "diffuse_color": (0.125, 0.125, 0.125),  # 1.0 / 8
                    "specular_color": (0.0, 0.0, 0.0),
                })
            # 添加顶部中心光源
            self.light_properties.append({
                "location": [0.0, 0.0, 5.0],
                "diffuse_color": (0.125, 0.125, 0.125),
                "specular_color": (0.0, 0.0, 0.0),
            })
            self.base_ambient_color = (0.5, 0.5, 0.5)
        else:
            raise ValueError(f"Unknown lighting_type: {lighting_type}")

    def spherical_to_cartesian(self, spherical_coords: torch.Tensor) -> torch.Tensor:
        """
        将球坐标转换为笛卡尔坐标
        
        Args:
            spherical_coords: [B, 3] (theta, phi, radius)
            
        Returns:
            cartesian_coords: [B, 3] (x, y, z)
        """
        theta, phi, radius = spherical_coords[:, 0], spherical_coords[:, 1], spherical_coords[:, 2]
        
        x = radius * torch.sin(phi) * torch.cos(theta)
        y = radius * torch.sin(phi) * torch.sin(theta)
        z = radius * torch.cos(phi)
        
        return torch.stack([x, y, z], dim=1)
    
    def create_cameras_from_poses(self, camera_poses: torch.Tensor, 
                                 pose_format: str = "cartesian",
                                 fov: float = 60.0) -> FoVPerspectiveCameras:
        """
        从相机位姿创建PyTorch3D相机对象
        
        Args:
            camera_poses: 相机位姿张量
                - spherical: [B, 7] (theta, phi, radius, qx, qy, qz, qw) - 球坐标+四元数旋转
                - cartesian: [B, 7] (x, y, z, qx, qy, qz, qw) - 笛卡尔坐标+四元数旋转
            pose_format: 位姿格式 "spherical" 或 "cartesian"
            fov: 视场角（度）
            
        Returns:
            cameras: PyTorch3D相机对象
            
        Notes:
            - spherical格式：相机位于球面坐标(theta, phi, radius)，朝向由四元数(qx,qy,qz,qw)决定
            - cartesian格式：相机位置和朝向完全由7DOF位姿决定，四元数格式为(qx,qy,qz,qw)
            - 相机朝向完全由四元数控制
        """
        # 确保使用 float32，避免 AMP 导致的半精度与渲染器 dtype 不匹配
        camera_poses = camera_poses.float()
        batch_size = camera_poses.shape[0]
        
        if pose_format == "spherical":
            # 球坐标格式：从球坐标和四元数计算位姿
            if camera_poses.shape[1] != 7:
                raise ValueError(f"spherical格式需要7个参数 [theta,phi,radius,qx,qy,qz,qw]，但得到了{camera_poses.shape[1]}个")
            
            # 将球坐标转换为笛卡尔坐标
            spherical_coords = camera_poses[:, :3]  # [B, 3] (theta, phi, radius)
            quaternions = camera_poses[:, 3:7]      # [B, 4] (qx, qy, qz, qw)
            
            # 转换球坐标为笛卡尔坐标
            camera_positions = self.spherical_to_cartesian(spherical_coords)
            
            # 构造完整的cartesian位姿
            cartesian_poses = torch.cat([camera_positions, quaternions], dim=1)  # [B, 7]
            
            # 使用和cartesian格式相同的处理方式
            R, T = self.pose_to_camera_matrices(cartesian_poses)
            
        elif pose_format == "cartesian":
            # 笛卡尔格式：直接从位姿和四元数计算R和T矩阵
            if camera_poses.shape[1] != 7:
                raise ValueError(f"cartesian格式需要7个参数 [x,y,z,qx,qy,qz,qw]，但得到了{camera_poses.shape[1]}个")
            
            # 从位姿和四元数直接计算R和T矩阵
            R, T = self.pose_to_camera_matrices(camera_poses)
            
        else:
            raise ValueError(f"Unknown pose_format: {pose_format}")
        
        # 创建相机对象
        cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            fov=fov
        )
        
        return cameras
    
    def render_views(self, 
                    meshes: Meshes,
                    cameras: FoVPerspectiveCameras,
                    lighting_type: str = "ambient",
                    return_point_maps: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        从已匹配的批次化网格和相机中渲染视图。
        
        这是一个纯粹的渲染函数，它假定输入的 'meshes' 和 'cameras' 
        对象的批次大小已经完全匹配。

        Args:
            meshes: PyTorch3D Meshes对象，批次大小为B。
            cameras: PyTorch3D Cameras对象，批次大小为B。
            lighting_type: 光照类型。

        Returns:
            rendered_images: 渲染的图像 [B, 3, H, W]。
            point_maps (可选): 当 `return_point_maps=True` 时返回对应的世界坐标 [B, 3, H, W]。
            valid_masks (可选): 对应点的有效掩码 [B, 1, H, W]。
        """
        # 1. 初始化光照
        self.setup_lighting(lighting_type)
        batch_size = len(meshes)

        # 2. 一次性完成光栅化，获取整个批次的 fragments
        _, fragments = self.renderer_with_frags(meshes, cameras=cameras)

        # 3. 向量化的光照和着色 (这部分逻辑不变)
        image_shape = (batch_size, self.raster_settings.image_size, self.raster_settings.image_size, 4)
        final_image = torch.zeros(image_shape, device=self.device, dtype=torch.float32)

        if self.light_properties:
            for i, properties in enumerate(self.light_properties):
                ambient_color = self.base_ambient_color if i == 0 else (0.0, 0.0, 0.0)
                lights_pass = PointLights(
                    device=self.device, location=[properties["location"]],
                    ambient_color=(ambient_color,), diffuse_color=(properties["diffuse_color"],),
                    specular_color=(properties["specular_color"],)
                )
                image_pass = self.shader(fragments, meshes, cameras=cameras, lights=lights_pass)
                final_image += image_pass
        else:
            lights_pass = PointLights(device=self.device, ambient_color=(self.base_ambient_color,))
            final_image = self.shader(fragments, meshes, cameras=cameras, lights=lights_pass)

        # 4. 后处理 (这部分逻辑不变)
        final_image = torch.clamp(final_image, 0.0, 1.0)
        rendered_images = final_image.permute(0, 3, 1, 2)[:, :3, :, :]

        point_maps: Optional[torch.Tensor] = None
        valid_masks: Optional[torch.Tensor] = None

        if return_point_maps:
            # 将每个像素插值得到的网格顶点位置转换为世界坐标点
            face_vertices = meshes.verts_packed()[meshes.faces_packed()]  # (F, 3, 3)
            interpolated = interpolate_face_attributes(
                fragments.pix_to_face, fragments.bary_coords, face_vertices
            )  # (B, H, W, K, 3)

            # 只保留 faces_per_pixel == 1 的第一个面
            interpolated = interpolated[..., 0, :]
            point_maps = interpolated.permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)

            # 有效像素掩码：pix_to_face >= 0 表示命中了一张面
            valid_masks = (fragments.pix_to_face[..., 0] >= 0).unsqueeze(1).float()

        if self.quality == "high" and self.render_image_size != self.image_size:
            try:
                rendered_images = F.interpolate(
                    rendered_images, size=(self.image_size, self.image_size),
                    mode='bilinear', align_corners=False, antialias=True
                )
            except TypeError:
                rendered_images = F.interpolate(
                    rendered_images, size=(self.image_size, self.image_size),
                    mode='bilinear', align_corners=False
                )

            if return_point_maps and point_maps is not None and valid_masks is not None:
                point_maps = F.interpolate(
                    point_maps, size=(self.image_size, self.image_size),
                    mode='bilinear', align_corners=False
                )
                valid_masks = F.interpolate(
                    valid_masks, size=(self.image_size, self.image_size),
                    mode='nearest'
                )

        if return_point_maps and point_maps is not None and valid_masks is not None:
            valid_masks = valid_masks > 0.5
            return rendered_images, point_maps, valid_masks

        return rendered_images

    def forward(self, 
               gt_mesh: Meshes,
               camera_poses: torch.Tensor,
               pose_format: str = "cartesian",
               fov: float = 60.0,
               lighting_type: str = "ambient",
               return_point_maps: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        前向传播：渲染新视图
        """
        # 1. 从位姿创建批次化的相机对象
        cameras = self.create_cameras_from_poses(
            camera_poses, 
            pose_format, 
            fov=fov
        )

        # print("camera_poses:",camera_poses)
        # 2. 调用纯粹的渲染函数
        #    这里隐含了一个假设：len(gt_mesh) == len(camera_poses)
        return self.render_views(
            gt_mesh,
            cameras,
            lighting_type=lighting_type,
            return_point_maps=return_point_maps
        )
