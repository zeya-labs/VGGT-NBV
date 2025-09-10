"""
基础数据集抽象类
定义所有数据集类的通用接口
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    NBV 数据集基类
    
    所有具体的数据集实现都应继承此类并实现抽象方法
    """
    
    def __init__(
        self,
        data_root: str,
        num_initial_views: int = 4,
        image_size: int = 518,
        split: str = "train",
        normalize_method: str = "quantile",
        num_samples: int = 10000,
        up_axis: str = "Y",
        **kwargs
    ):
        """
        初始化基础数据集
        
        Args:
            data_root: 数据根目录
            num_initial_views: 初始视图数量
            image_size: 图像尺寸
            split: 数据集分割 (train/val/test)
            normalize_method: 网格归一化方法
            num_samples: 采样点数量
            up_axis: 上方向轴 ('Y' 或 'Z')
            **kwargs: 其他参数，由子类处理
        """
        self.data_root = data_root
        self.num_initial_views = num_initial_views
        self.image_size = image_size
        self.split = split
        self.normalize_method = normalize_method
        self.num_samples = num_samples
        self.up_axis = up_axis.upper()  # 确保大写格式
        
        # 验证数据根目录
        if not os.path.exists(data_root):
            raise ValueError(f"Data root directory does not exist: {data_root}")
        
        # 由子类实现具体的数据加载逻辑
        self.data_list = self._load_data_list()
        
        print(f"[{self.__class__.__name__}] Loaded {len(self.data_list)} samples for {split} split")
        print(f"Mesh normalization: {normalize_method}, Sample points: {num_samples}")
    
    @abstractmethod # 由子类实现具体的数据加载逻辑
    def _load_data_list(self) -> List[Dict]:
        """
        加载数据列表
        
        每个数据集应根据自己的目录结构实现此方法
        返回包含数据项信息的字典列表
        """
        pass
    
    @abstractmethod
    def _get_mesh_path(self, data_item: Dict) -> str:
        """
        获取网格文件路径
        
        Args:
            data_item: 数据项字典
            
        Returns:
            网格文件的完整路径
        """
        pass
    
    @abstractmethod
    def _get_image_paths(self, data_item: Dict) -> List[str]:
        """
        获取可用图像路径列表
        
        Args:
            data_item: 数据项字典
            
        Returns:
            图像文件路径列表
        """
        pass
    
    @abstractmethod
    def _get_camera_poses_path(self, data_item: Dict) -> Optional[str]:
        """
        获取相机位姿文件路径

        Args:
            data_item: 数据项字典
            
        Returns:
            相机位姿文件路径，如果不存在则返回None
        """
        pass
    
    def _load_camera_poses(self, camera_poses_path: Optional[str], selected_indices: Optional[List[int]] = None) -> Optional[torch.Tensor]:
        """
        加载相机位姿数据
        
        通用实现，支持常见的相机位姿格式
        子类可以重写此方法以支持特定格式
        
        Args:
            camera_poses_path: 相机位姿文件路径，可以为None
            selected_indices: 选中的图像索引列表，如果为None则返回所有位姿
            
        Returns:
            相机位姿张量，形状为 [N, 7] (position + quaternion) 或 [N, 4, 4] (变换矩阵)
            如果加载失败则返回None
        """
        if camera_poses_path is None or not os.path.exists(camera_poses_path):
            return None
            
        try:
            import json
            with open(camera_poses_path, 'r') as f:
                poses_data = json.load(f)
            
            # 处理不同的数据格式并获取所有位姿
            all_poses = None
            
            if isinstance(poses_data, list):
                # 格式1: [{"position": [...], "quaternion": [...]}] 
                if len(poses_data) > 0 and isinstance(poses_data[0], dict):
                    if "position" in poses_data[0] and "quaternion" in poses_data[0]:
                        all_poses = self._parse_position_quaternion_format(poses_data)
                    elif "camera_matrix" in poses_data[0] or "transform" in poses_data[0]:
                        all_poses = self._parse_matrix_format(poses_data)
                # 格式2: [[x, y, z, qx, qy, qz, qw], ...]
                elif len(poses_data) > 0 and isinstance(poses_data[0], list) and len(poses_data[0]) == 7:
                    all_poses = torch.tensor(poses_data, dtype=torch.float32)
            
            if all_poses is None:
                print(f"Warning: Unsupported camera poses format in {camera_poses_path}")
                return None
            
            # 如果指定了选中的索引，则只返回对应的位姿
            if selected_indices is not None:
                try:
                    return all_poses[selected_indices]
                except IndexError as e:
                    print(f"Warning: Index out of range when selecting poses: {e}")
                    return None
            
            return all_poses
            
        except Exception as e:
            print(f"Error loading camera poses from {camera_poses_path}: {e}")
            return None
    
    def _parse_position_quaternion_format(self, poses_data: List[Dict]) -> torch.Tensor:
        """
        解析 position + quaternion 格式的相机位姿
        
        Args:
            poses_data: [{"position": [x,y,z], "quaternion": [x,y,z,w]}]
            
        Returns:
            形状为 [N, 7] 的张量 (position + quaternion)
        """
        poses = []
        for pose in poses_data:
            position = pose["position"]
            quaternion = pose["quaternion"]
            # 合并position和quaternion: [x, y, z, qx, qy, qz, qw]
            poses.append(position + quaternion)
        
        return torch.tensor(poses, dtype=torch.float32)
    
    def _parse_matrix_format(self, poses_data: List[Dict]) -> torch.Tensor:
        """
        解析变换矩阵格式的相机位姿
        
        Args:
            poses_data: [{"camera_matrix": [[...], [...], [...], [...]]}, ...]
            
        Returns:
            形状为 [N, 4, 4] 的张量
        """
        poses = []
        for pose in poses_data:
            if "camera_matrix" in pose:
                matrix = pose["camera_matrix"]
            elif "transform" in pose:
                matrix = pose["transform"]
            else:
                continue
            poses.append(matrix)
        
        return torch.tensor(poses, dtype=torch.float32)
    
    def _parse_cameras_format(self, cameras_data: List[Dict]) -> torch.Tensor:
        """
        解析cameras格式的相机位姿
        
        Args:
            cameras_data: [{"R": [...], "t": [...]}, ...]
            
        Returns:
            形状为 [N, 4, 4] 的张量
        """
        poses = []
        for camera in cameras_data:
            if "R" in camera and "t" in camera:
                # 构建4x4变换矩阵
                R = torch.tensor(camera["R"], dtype=torch.float32)
                t = torch.tensor(camera["t"], dtype=torch.float32)
                
                # 构建变换矩阵
                transform = torch.eye(4)
                transform[:3, :3] = R
                transform[:3, 3] = t
                poses.append(transform)
        
        return torch.stack(poses) if poses else None
    
    def _load_mesh_data(
        self,
        mesh_path: str,
        normalize_method: str = "quantile",
        num_samples: int = 10000,
    ) -> Dict[str, torch.Tensor]:
        """
        加载网格数据
        
        这是一个通用的网格加载方法，子类可以重写以支持特定格式
        """
        from ..utils.mesh_utils import load_and_normalize_mesh
        
        try:
            mesh_data = load_and_normalize_mesh(
                mesh_path=mesh_path,
                normalize_method=normalize_method,
                num_samples=num_samples,
            )
            return mesh_data
        except Exception as e:
            print(f"Error loading mesh {mesh_path}: {e}")
            # 返回默认数据以避免训练中断
            return {
                "mesh_path": mesh_path,
                "vertices": torch.randn(1000, 3),
                "faces": torch.randint(0, 1000, (1800, 3)),
                "gt_points": torch.randn(num_samples, 3),
                "normalize_method": normalize_method,
                "num_samples": num_samples,
            }
    
    def _select_initial_images(self, available_images: List[str]) -> Tuple[List[str], List[int]]:
        """
        选择初始视图图像
        
        默认随机选择，子类可以重写以实现特定的选择策略
        
        Returns:
            Tuple[List[str], List[int]]: (选中的图像路径列表, 对应的索引列表)
        """
        import random
        
        if len(available_images) < self.num_initial_views:
            raise ValueError(
                f"Not enough images available. Required: {self.num_initial_views}, "
                f"Available: {len(available_images)}"
            )
        
        # 使用局部随机数生成器，基于可用图像的哈希值作为种子确保可重现性
        seed = abs(hash(tuple(sorted(available_images)))) % (2**32 - 1)
        rng = random.Random(seed)
        selected_paths = rng.sample(available_images, self.num_initial_views)
        
        # 提取对应的索引
        selected_indices = []
        for path in selected_paths:
            index = self._extract_image_index(path)
            if index is not None:
                selected_indices.append(index)
            else:
                # 如果无法从文件名提取索引，使用在available_images中的位置作为索引
                try:
                    index = available_images.index(path)
                    selected_indices.append(index)
                except ValueError:
                    print(f"Warning: Cannot determine index for image {path}")
        
        # 按索引升序排列，保持路径与索引一一对应
        pairs = sorted(zip(selected_indices, selected_paths), key=lambda x: x[0])
        selected_indices = [p[0] for p in pairs]
        selected_paths = [p[1] for p in pairs]
        
        return selected_paths, selected_indices
    
    def _extract_image_index(self, image_path: str) -> Optional[int]:
        """
        从图像路径中提取索引
        
        默认实现假设图像文件名包含数字索引，如 view_000.png, image_001.jpg 等
        子类可以重写此方法以适应不同的命名规则
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像索引，如果无法提取则返回None
        """
        import re
        import os
        
        # 获取文件名（不包含扩展名）
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # 尝试提取数字
        # 匹配常见模式：view_000, image_001, 000, 001 等
        patterns = [
            r'view_(\d+)',      # view_000
            r'image_(\d+)',     # image_001  
            r'(\d+)$',          # 纯数字结尾
            r'_(\d+)$',         # 下划线+数字结尾
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return int(match.group(1))
        
        # 如果无法提取索引，尝试从available_images列表中找到对应位置
        return None
    
    def _load_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        加载和预处理图像
        """
        import sys
        sys.path.append("vggt/")
        from vggt.utils.load_fn import load_and_preprocess_images  # type: ignore
        
        return load_and_preprocess_images(
            image_paths, 
            mode="crop", 
            image_size=self.image_size
        )
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据样本
        
        这是通用实现，大部分子类不需要重写
        """
        data_item = self.data_list[idx]
        
        # 获取可用图像路径
        available_image_paths = self._get_image_paths(data_item)
        
        # 选择初始视图并获取对应索引
        selected_image_paths, selected_indices = self._select_initial_images(available_image_paths)
        
        # 加载图像
        initial_images = self._load_images(selected_image_paths)
        
        # 获取网格路径并加载网格数据
        mesh_path = self._get_mesh_path(data_item)
        gt_mesh_data = self._load_mesh_data(
            mesh_path,
            normalize_method=self.normalize_method,
            num_samples=self.num_samples,
        )
        
        # 加载相机位姿数据，只获取选中图像对应的位姿
        camera_poses_path = self._get_camera_poses_path(data_item)
        camera_poses = self._load_camera_poses(camera_poses_path, selected_indices)
        result = {
            "initial_images": initial_images,
            "gt_mesh_data": gt_mesh_data,
            "camera_poses": camera_poses,
            "mesh_path": mesh_path,
            "dataset_type": self.__class__.__name__,
        }
        
        return result
    
    @property # @是装饰器，用于将一个方法转换为属性，可以像属性一样访问，而不是像方法一样调用，通常用于返回一些元数据或配置信息。
    def dataset_info(self) -> Dict:
        """返回数据集信息"""
        return {
            "dataset_type": self.__class__.__name__,
            "data_root": self.data_root,
            "split": self.split,
            "num_samples": len(self.data_list),
            "num_initial_views": self.num_initial_views,
            "image_size": self.image_size,
            "normalize_method": self.normalize_method,
            "num_mesh_samples": self.num_samples,
        }
