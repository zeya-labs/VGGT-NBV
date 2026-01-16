"""
合成训练数据生成
重构后的主模块，负责协调各个子模块生成合成训练数据
"""

import json
import os
from typing import Dict, List

from .mesh_generator import MeshGenerator, create_pytorch3d_mesh
from .camera_utils import CameraPoseGenerator, pose_dict_to_tensor
from ..rendering.differentiable_renderer import DifferentiableRenderer
from .textures import TextureGenerator
from .logging_utils import setup_logging


class SyntheticDataGenerator:
    """合成数据生成器主类"""
    
    def __init__(
        self,
        output_dir: str,
        image_size: int = 518,
        device: str = "cuda",
        quality: str = "high",
        downsample_factor: int = 2,
        up_axis: str = "Y"
    ):
        """
        初始化合成数据生成器
        
        Args:
            output_dir: 输出目录
            image_size: 图像尺寸
            device: 计算设备
            quality: 渲染质量
            downsample_factor: 下采样因子
            up_axis: 上朝向轴
        """
        self.output_dir = output_dir
        self.device = device
        self.up_axis = up_axis
        
        # 初始化各个组件
        self.mesh_generator = MeshGenerator(up_axis=up_axis)
        self.camera_generator = CameraPoseGenerator(up_axis=up_axis)
        self.renderer = DifferentiableRenderer(
            image_size=image_size,
            device=device,
            quality=quality,
            downsample_factor=downsample_factor
        )
        self.texture_generator = TextureGenerator(
            image_size=(1024, 1024), 
            num_squares=16
        )
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_dataset(
        self,
        num_objects: int = 100,
        num_views_per_object: int = 20,
        train_ratio: float = 0.8
    ):
        """
        生成完整的合成数据集
        
        Args:
            num_objects: 生成的对象数量
            num_views_per_object: 每个对象的视图数量
            train_ratio: 训练集比例
        """
        train_objects = int(num_objects * train_ratio)
        val_objects = num_objects - train_objects

        train_list = []
        val_list = []

        for i in range(num_objects):
            object_id = f"object_{i:04d}"
            logger.info("--- Generating %s ---", object_id)
            
            # 生成单个对象的数据
            data_item = self._generate_single_object(
                object_id, i, num_views_per_object
            )

            if i < train_objects:
                train_list.append(data_item)
            else:
                val_list.append(data_item)

        # 保存数据集列表
        self._save_dataset_lists(train_list, val_list)
        
        # 打印统计信息
        self._print_statistics(num_objects, len(train_list), len(val_list))
    
    def _generate_single_object(
        self, 
        object_id: str, 
        seed: int, 
        num_views: int
    ) -> Dict:
        """
        生成单个对象的数据
        
        Args:
            object_id: 对象ID
            seed: 随机种子
            num_views: 视图数量
            
        Returns:
            data_item: 对象数据字典
        """
        object_dir = os.path.join(self.output_dir, object_id)
        os.makedirs(object_dir, exist_ok=True)

        # 1. 生成网格
        mesh_data = self.mesh_generator.generate_textured_perlin_sphere(seed)
        mesh_path = os.path.join(object_dir, "mesh.obj")
        self.mesh_generator.save_mesh_with_uvs(mesh_data, mesh_path)

        # 2. 生成纹理
        texture_path = os.path.join(object_dir, "texture.png")
        texture_image = self.texture_generator.generate_unique_color_texture(
            texture_path, seed=seed, font_scale=0.5
        )

        # 3. 生成相机位姿
        camera_poses = self.camera_generator.generate_camera_poses(
            num_views, seed=seed
        )
        poses_path = os.path.join(object_dir, "camera_poses.json")
        self.camera_generator.save_camera_poses(camera_poses, poses_path)

        # 4. 创建PyTorch3D mesh
        mesh = create_pytorch3d_mesh(mesh_data, texture_image, self.device)

        # 5. 渲染图像
        images_dir = os.path.join(object_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        image_files = self._render_all_views(mesh, camera_poses, images_dir)

        # 6. 构建数据项
        return {
            "object_id": object_id,
            "mesh_path": os.path.relpath(mesh_path, self.output_dir),
            "texture_path": os.path.relpath(texture_path, self.output_dir),
            "poses_path": os.path.relpath(poses_path, self.output_dir),
            "images_dir": os.path.relpath(images_dir, self.output_dir),
            "available_images": image_files,
        }
    
    def _render_all_views(
        self, 
        mesh, 
        camera_poses: List[Dict], 
        images_dir: str
    ) -> List[str]:
        """
        渲染所有视图
        
        Args:
            mesh: PyTorch3D mesh对象
            camera_poses: 相机位姿列表
            images_dir: 图像保存目录
            
        Returns:
            image_files: 图像文件名列表
        """
        import torch
        import numpy as np
        from PIL import Image
        
        image_files = []
        
        for j, pose in enumerate(camera_poses):
            img_name = f"view_{j:03d}.png"
            img_path = os.path.join(images_dir, img_name)
            
            # 将相机位姿转换为张量格式
            pose_tensor = pose_dict_to_tensor(pose, self.device)
            
            # 使用可微分渲染器渲染图像
            with torch.no_grad():
                rendered_tensor = self.renderer.forward(
                    gt_mesh=mesh,
                    camera_poses=pose_tensor,
                    pose_format="cartesian",
                    fov=60.0,
                    lighting_type="ambient"
                )
            
            # 转换为PIL图像
            rendered_array = rendered_tensor[0].cpu().numpy()  # [3, H, W]
            rendered_array = np.transpose(rendered_array, (1, 2, 0))  # [H, W, 3]
            rendered_array = np.clip(rendered_array * 255, 0, 255).astype(np.uint8)
            
            synthetic_image = Image.fromarray(rendered_array)
            synthetic_image.save(img_path)
            image_files.append(img_name)
        
        return image_files
    
    def _save_dataset_lists(self, train_list: List[Dict], val_list: List[Dict]):
        """保存训练和验证数据集列表"""
        with open(os.path.join(self.output_dir, "train.json"), 'w') as f:
            json.dump(train_list, f, indent=2)

        with open(os.path.join(self.output_dir, "val.json"), 'w') as f:
            json.dump(val_list, f, indent=2)
    
    def _print_statistics(self, total_objects: int, train_count: int, val_count: int):
        """打印数据集统计信息"""
        logger.info(
            "Generated %d synthetic objects | Train: %d | Val: %d | Up axis: %s",
            total_objects,
            train_count,
            val_count,
            self.up_axis,
        )


def create_synthetic_training_data(
    output_dir: str,
    num_objects: int = 100,
    num_views_per_object: int = 20,
    image_size: int = 518,
    device: str = "cuda",
    quality: str = "high",
    downsample_factor: int = 2,
    up_axis: str = "Y",
):
    """
    创建带有特征纹理的合成训练数据。
    
    Args:
        output_dir: 输出目录
        num_objects: 生成的对象数量
        num_views_per_object: 每个对象的视图数量
        image_size: 图像尺寸
        device: 计算设备
        quality: 渲染质量
        downsample_factor: 下采样因子
        up_axis: 上朝向轴，可选 "X", "Y", "Z", "-X", "-Y", "-Z"
    """
    generator = SyntheticDataGenerator(
        output_dir=output_dir,
        image_size=image_size,
        device=device,
        quality=quality,
        downsample_factor=downsample_factor,
        up_axis=up_axis
    )
    
    generator.generate_dataset(
        num_objects=num_objects,
        num_views_per_object=num_views_per_object
    )


__all__ = ["create_synthetic_training_data", "SyntheticDataGenerator"]

