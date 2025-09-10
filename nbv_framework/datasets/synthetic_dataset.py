"""
合成数据集加载器
适用于您当前的合成数据格式：object_XXXX/mesh.obj + images/
"""

import os
import json
from typing import List, Dict
from .base_dataset import BaseDataset


class SyntheticDataset(BaseDataset):
    """
    合成数据集加载器
    
    数据结构：
    data_root/
    ├── object_0000/
    │   ├── mesh.obj
    │   └── images/
    │       ├── view_000.png
    │       ├── view_001.png
    │       └── ...
    ├── object_0001/
    └── ...
    """
    
    def __init__(self, **kwargs):
        """
        初始化合成数据集
        """
        super().__init__(**kwargs)
    
    def _load_data_list(self) -> List[Dict]:
        """
        加载合成数据集列表
        
        优先从 split.json 文件加载，如果不存在则扫描目录
        """
        split_file = os.path.join(self.data_root, f"{self.split}.json")
        
        if os.path.exists(split_file):
            return self._load_from_split_file(split_file)
    
    def _load_from_split_file(self, split_file: str) -> List[Dict]:
        """
        从分割文件（如 split.json）加载数据列表。

        该方法会读取指定的分割文件（通常为JSON格式），并将其中每个数据项的相对路径（如 mesh_path、images_dir）
        自动补全为绝对路径，确保后续数据加载的路径正确。

        Args:
            split_file (str): 分割文件的完整路径。

        Returns:
            List[Dict]: 处理后的数据项列表，每个数据项为字典，包含 mesh_path、images_dir 等字段，且路径为绝对路径。
        """
        with open(split_file, 'r') as f:
            data_list = json.load(f)
        
        # 补全JSON中的相对路径为绝对路径
        normalized_list = []
        for item in data_list:
            new_item = dict(item)
            
            # 处理网格路径
            mesh_path = new_item.get("mesh_path")
            if isinstance(mesh_path, str) and not os.path.isabs(mesh_path):
                new_item["mesh_path"] = os.path.join(self.data_root, mesh_path)
            
            # 处理图像目录路径
            images_dir = new_item.get("images_dir")
            if isinstance(images_dir, str) and not os.path.isabs(images_dir):
                new_item["images_dir"] = os.path.join(self.data_root, images_dir)
            
            # 处理相机位姿路径
            poses_path = new_item.get("poses_path")
            if isinstance(poses_path, str) and not os.path.isabs(poses_path):
                new_item["poses_path"] = os.path.join(self.data_root, poses_path)
            
            normalized_list.append(new_item)
        
        return normalized_list
    
    def _get_mesh_path(self, data_item: Dict) -> str:
        """获取网格文件路径"""
        return data_item["mesh_path"]
    
    def _get_image_paths(self, data_item: Dict) -> List[str]:
        """获取可用图像路径列表"""
        images_dir = data_item["images_dir"]
        available_images = data_item["available_images"]
        
        return [os.path.join(images_dir, img_name) for img_name in available_images]
    
    def _get_camera_poses_path(self, data_item: Dict) -> str:
        """
        获取相机位姿文件路径
        """
        return data_item["poses_path"]