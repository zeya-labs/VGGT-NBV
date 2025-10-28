"""
House3K数据集加载器
用于加载House3K_obj数据集，支持从3D网格动态生成多视角图像
"""

import os
import glob
import random
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from .base_dataset import BaseDataset
from ..utils.camera_utils import (
    pose_dict_to_tensor,
    world_points_to_camera_depth,
    normalize_depth_for_visualization,
)
from ..utils.render_utils import render_gt_point_maps


class House3KDataset(BaseDataset):
    """
    House3K数据集加载器
    
    数据结构：
    data_root/
    ├── BATCH_1/
    │   ├── Set_A/
    │   │   ├── BAT1_SETA_HOUSE1.obj
    │   │   ├── BAT1_SETA_HOUSE1.mtl
    │   │   ├── BAT1_SETA_HOUSE2.obj
    │   │   └── ...
    │   └── Set_B/
    │       └── ...
    ├── BATCH_2/
    └── ...
    
    特点：
    - 只有3D网格文件，没有预渲染图像
    - 需要动态生成相机位姿和渲染图像
    - 支持训练/验证/测试分割
    """
    
    def __init__(
        self,
        data_root: str,
        num_initial_views: int = 4,
        image_size: int = 518,
        split: str = "train",
        normalize_method: str = "quantile",
        num_samples: int = 10000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        max_meshes: int = None,
        use_cache: bool = True,
        **kwargs
    ):
        """
        初始化House3K数据集
        
        Args:
            data_root: House3K_obj数据根目录
            num_initial_views: 初始视图数量
            image_size: 图像尺寸
            split: 数据集分割 (train/val/test)
            normalize_method: 网格归一化方法
            num_samples: 采样点数量
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            max_meshes: 全局最大mesh数量限制（用于控制训练规模）
            use_cache: 是否使用缓存加速数据加载
            **kwargs: 其他参数
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.max_meshes = max_meshes
        self.use_cache = use_cache
        
        # 验证分割比例
        if self.test_ratio < 0:
            raise ValueError(f"Invalid split ratios: train={train_ratio}, val={val_ratio}")
        
        # 初始化渲染器（延迟加载）
        self._renderer = None
        self._camera_generator = None
        
        # 图像缓存目录
        self.cache_dir = os.path.join(data_root, ".cache", split) if use_cache else None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        super().__init__(
            data_root=data_root,
            num_initial_views=num_initial_views,
            image_size=image_size,
            split=split,
            normalize_method=normalize_method,
            num_samples=num_samples,
            **kwargs
        )
    
    def _load_data_list(self) -> List[Dict]:
        """
        加载House3K数据集列表
        
        扫描所有BATCH目录，找到所有.obj文件，过滤掉没有纹理的模型，然后按比例分割
        """
        print(f"正在扫描House3K数据集: {self.data_root}")
        
        all_objects = []
        total_scanned = 0
        total_with_texture = 0
        
        # 查找所有批次目录
        batch_dirs = []
        for item in os.listdir(self.data_root):
            item_path = os.path.join(self.data_root, item)
            if os.path.isdir(item_path) and ('BATCH' in item.upper() or 'Batch' in item):
                batch_dirs.append(item)
        
        batch_dirs.sort()  # 确保顺序一致
        print(f"找到 {len(batch_dirs)} 个批次目录: {batch_dirs}")
        
        for batch_name in batch_dirs:
            batch_path = os.path.join(self.data_root, batch_name)
            batch_objects = self._scan_batch_directory(batch_path, batch_name)
            
            # 统计信息
            batch_scanned = len(batch_objects)
            batch_with_texture = len([obj for obj in batch_objects if obj.get('has_texture', True)])
            total_scanned += batch_scanned
            total_with_texture += batch_with_texture
            
            # 只添加有纹理的模型
            valid_objects = [obj for obj in batch_objects if obj.get('has_texture', True)]
            all_objects.extend(valid_objects)
            
            # print(f"批次 {batch_name}: 扫描 {batch_scanned} 个模型，有纹理 {batch_with_texture} 个")
            

        
        print(f"[House3K数据集] 总共扫描 {total_scanned} 个3D模型，其中 {total_with_texture} 个有完整纹理")
        print(f"[House3K数据集] 最终加载 {len(all_objects)} 个有效3D模型")
        
        # 全局mesh数量限制
        if self.max_meshes and len(all_objects) > self.max_meshes:
            original_count = len(all_objects)
            # 使用固定种子确保可重复性
            rng = random.Random(42)
            rng.shuffle(all_objects)
            all_objects = all_objects[:self.max_meshes]
            # print(all_objects)
            print(f"[House3K数据集] 应用全局mesh限制，从 {original_count} 个减少到 {self.max_meshes} 个")
        
        # 按分割比例划分数据集
        split_data = self._split_dataset(all_objects)
        
        return split_data
    
    def _scan_batch_directory(self, batch_path: str, batch_name: str) -> List[Dict]:
        """
        扫描单个批次目录，找到所有房屋模型
        
        Args:
            batch_path: 批次目录路径
            batch_name: 批次名称
            
        Returns:
            该批次中的所有模型信息列表
        """
        batch_objects = []
        
        # 查找所有Set目录（处理不同的命名格式）
        set_dirs = []
        for item in os.listdir(batch_path):
            item_path = os.path.join(batch_path, item)
            if os.path.isdir(item_path) and 'SET' in item.upper():
                set_dirs.append(item)
        
        set_dirs.sort()
        
        for set_name in set_dirs:
            set_path = os.path.join(batch_path, set_name)
            
            # 查找该Set中的所有.obj文件
            obj_pattern = os.path.join(set_path, "*.obj")
            obj_files = glob.glob(obj_pattern)
            
            for obj_file in obj_files:
                # 提取模型信息
                obj_basename = os.path.basename(obj_file)
                model_name = os.path.splitext(obj_basename)[0]
                
                # 查找对应的.mtl文件
                mtl_file = os.path.join(set_path, model_name + ".mtl")
                
                # 检查纹理文件是否存在
                has_valid_textures = self._check_texture_files(mtl_file, set_path)
                
                # 添加所有模型，但标记是否有纹理
                batch_objects.append({
                    "batch_name": batch_name,
                    "set_name": set_name,
                    "model_name": model_name,
                    "obj_path": obj_file,
                    "mtl_path": mtl_file,
                    "set_path": set_path,
                    "has_texture": has_valid_textures,
                })
                
                # if not has_valid_textures:
                #     print(f"模型 {model_name}: 纹理文件不完整或缺失，将被过滤")
        
        print(f"批次 {batch_name}: 找到 {len(batch_objects)} 个模型")
        return batch_objects
    
    def _check_texture_files(self, mtl_file: str, set_path: str) -> bool:
        """
        检查MTL文件中引用的纹理文件是否存在
        
        Args:
            mtl_file: MTL文件路径
            set_path: Set目录路径
            
        Returns:
            bool: 如果所有纹理文件都存在则返回True，否则返回False
        """
        if not os.path.exists(mtl_file):
            return False
        
        try:
            with open(mtl_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # 查找纹理文件引用
            texture_files = []
            for line in lines:
                line = line.strip()
                # 检查常见的纹理映射关键字
                if line.startswith(('map_Kd', 'map_Ka', 'map_Ks', 'map_Ns', 'map_d', 'map_bump', 'bump')):
                    # 提取纹理文件名
                    parts = line.split()
                    if len(parts) >= 2:
                        texture_filename = parts[-1]  # 通常纹理文件名是最后一个参数
                        texture_files.append(texture_filename)
            
            # 检查所有纹理文件是否存在
            for texture_file in texture_files:
                texture_path = os.path.join(set_path, texture_file)
                if not os.path.exists(texture_path):
                    return False
            
            # 如果没有找到任何纹理文件引用，也认为是无效的
            return len(texture_files) > 0
            
        except Exception as e:
            print(f"读取MTL文件失败 {mtl_file}: {e}")
            return False
    
    def _split_dataset(self, all_objects: List[Dict]) -> List[Dict]:
        """
        将所有对象按比例分割成训练/验证/测试集
        """
        # 使用局部随机数生成器确保可重复性和多进程安全
        shuffled_objects = all_objects.copy()
        rng = random.Random(42)  # 使用固定种子确保数据集分割的一致性
        rng.shuffle(shuffled_objects)
        
        total_count = len(shuffled_objects)
        
        # 当数据量很少时，确保每个split都有数据
        if total_count <= 3:
            split_objects = shuffled_objects  # 所有split共享数据
            train_count = total_count
            val_count = total_count
            test_count = total_count
        else:
            train_count = max(1, int(total_count * self.train_ratio))
            val_count = max(1, int(total_count * self.val_ratio))
            test_count = total_count - train_count - val_count
            
            if self.split == "train":
                split_objects = shuffled_objects[:train_count]
            elif self.split == "val":
                split_objects = shuffled_objects[train_count:train_count + val_count]
            elif self.split == "test":
                split_objects = shuffled_objects[train_count + val_count:]
            else:
                raise ValueError(f"Unknown split: {self.split}")
        
        print(f"数据集分割 - 总计: {total_count}, "
              f"训练: {train_count}, 验证: {val_count}, "
              f"测试: {test_count}")
        print(f"当前分割 '{self.split}': {len(split_objects)} 个样本")
        
        return split_objects
    
    def _get_mesh_path(self, data_item: Dict) -> str:
        """获取网格文件路径"""
        return data_item["obj_path"]
    
    def _get_image_paths(self, data_item: Dict) -> List[str]:
        """
        获取可用图像路径列表
        
        由于House3K数据集没有预渲染图像，这里返回虚拟的图像路径
        实际图像将在运行时动态生成
        """
        # 生成虚拟图像路径，用于后续的视图选择
        model_name = data_item["model_name"]
        num_virtual_views = max(20, self.num_initial_views * 2)  # 确保有足够的虚拟视图
        
        virtual_paths = []
        for i in range(num_virtual_views):
            virtual_path = f"virtual://{model_name}/view_{i:03d}.png"
            virtual_paths.append(virtual_path)
        
        return virtual_paths
    
    def _get_camera_poses_path(self, data_item: Dict) -> Optional[str]:
        """
        获取相机位姿文件路径
        
        House3K数据集没有相机位姿文件，返回None
        相机位姿将在运行时动态生成
        """
        return None
    
    def _extract_image_index(self, image_path: str) -> Optional[int]:
        """从虚拟图像路径中提取索引"""
        if image_path.startswith("virtual://"):
            # 从 "virtual://model_name/view_XXX.png" 中提取索引
            try:
                filename = image_path.split("/")[-1]  # view_XXX.png
                index_str = filename.split("_")[1].split(".")[0]  # XXX
                return int(index_str)
            except (IndexError, ValueError):
                return None
        return super()._extract_image_index(image_path)
    
    def _generate_camera_poses(self, num_views: int, seed: int = None) -> List[Dict]:
        """
        生成相机位姿
        
        Args:
            num_views: 视图数量
            seed: 随机种子
            
        Returns:
            相机位姿列表
        """
        if self._camera_generator is None:
            # 延迟导入和初始化
            from ..utils.camera_utils import CameraPoseGenerator
            self._camera_generator = CameraPoseGenerator(up_axis=self.up_axis)
        
        return self._camera_generator.generate_camera_poses(num_views, seed=seed, hemisphere='upper')
    
    def _get_renderer(self):
        """获取渲染器（延迟初始化），失败时立即抛出异常。"""
        if self._renderer is None:
            try:
                from ..rendering.differentiable_renderer import DifferentiableRenderer
            except ImportError as exc:
                raise RuntimeError(
                    "无法导入 DifferentiableRenderer，请确认 PyTorch3D 依赖已正确安装。"
                ) from exc

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._renderer = DifferentiableRenderer(
                image_size=self.image_size,
                device=device
            )

        return self._renderer
    
    # def _render_images_from_mesh(
    #     self,
    #     mesh_path: str,
    #     camera_poses: List[Dict],
    #     selected_indices: List[int]
    # ) -> torch.Tensor:
    #     """
    #     从3D网格渲染图像
        
    #     Args:
    #         mesh_path: 网格文件路径
    #         camera_poses: 相机位姿列表
    #         selected_indices: 选中的视图索引
            
    #     Returns:
    #         渲染的图像张量 [N, 3, H, W]
    #     """
    #     renderer = self._get_renderer()
        
    #     if renderer is None:
    #         # 如果渲染器不可用，返回随机图像作为占位符
    #         print(f"警告：渲染器不可用，使用随机图像 {mesh_path}")
    #         num_views = len(selected_indices)
    #         return torch.rand(num_views, 3, self.image_size, self.image_size)
        
    #     try:
    #         # 加载3D网格
    #         from ..utils.mesh_utils import load_mesh_as_pytorch3d
    #         mesh = load_mesh_as_pytorch3d(mesh_path)
            
    #         # 选择对应的相机位姿
    #         selected_poses = [camera_poses[i] for i in selected_indices]
            
    #         # 为每个相机位姿复制网格
    #         device = renderer.device
            
    #         # 转换位姿格式
    #         pose_tensors = [pose_dict_to_tensor(pose, device=device) for pose in selected_poses]
    #         camera_poses_tensor = torch.cat(pose_tensors, dim=0)
    #         mesh = mesh.to(device)
            
    #         # 创建批次化的网格，每个相机位姿对应一个网格副本
    #         num_views = len(selected_poses)
    #         meshes_batch = mesh.extend(num_views)
            
    #         # 渲染图像
    #         with torch.no_grad():
    #             rendered_images = renderer.forward(
    #                 gt_mesh=meshes_batch,
    #                 camera_poses=camera_poses_tensor,
    #                 pose_format="cartesian",
    #                 fov=60.0
    #             )
            
    #         # 确保返回CPU张量，避免pin_memory问题
    #         return rendered_images.cpu()
            
    #     except Exception as e:
    #         print(f"渲染失败 {mesh_path}: {e}")
    #         # 返回随机图像作为后备
    #         num_views = len(selected_indices)
    #         return torch.rand(num_views, 3, self.image_size, self.image_size)
    
    def _render_images_from_mesh_data(
        self,
        gt_mesh_data: Dict,
        camera_poses: List[Dict],
        selected_indices: List[int]
    ) -> torch.Tensor:
        """
        从已加载的网格数据渲染图像
        
        Args:
            gt_mesh_data: 已加载的网格数据字典，包含normalized_mesh
            camera_poses: 相机位姿列表
            selected_indices: 选中的视图索引
            
        Returns:
            渲染的图像张量 [N, 3, H, W]
        """
        renderer = self._get_renderer()

        try:
            # 使用已经归一化的网格
            mesh = gt_mesh_data['normalized_mesh']

            # 选择对应的相机位姿
            selected_poses = [camera_poses[i] for i in selected_indices]

            # 为每个相机位姿复制网格
            device = renderer.device

            # 转换位姿格式
            pose_tensors = [pose_dict_to_tensor(pose, device=device) for pose in selected_poses]
            camera_poses_tensor = torch.cat(pose_tensors, dim=0)
            mesh = mesh.to(device)

            # 创建批次化的网格，每个相机位姿对应一个网格副本
            num_views = len(selected_poses)
            meshes_batch = mesh.extend(num_views)

            # 渲染图像
            with torch.no_grad():
                rendered_images = renderer.forward(
                    gt_mesh=meshes_batch,
                    camera_poses=camera_poses_tensor,
                    pose_format="cartesian",
                    fov=60.0
                )

            # 确保返回CPU张量，避免pin_memory问题
            return rendered_images.cpu()

        except Exception as exc:
            raise RuntimeError("渲染 House3K 样本失败，请检查网格或相机位姿数据。") from exc
    
    def _load_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        加载和预处理图像
        
        重写此方法以支持动态渲染
        """
        # 检查是否为虚拟图像路径
        if all(path.startswith("virtual://") for path in image_paths):
            # 需要动态渲染图像
            return self._render_virtual_images(image_paths)
        else:
            # 使用基类方法加载实际图像文件
            return super()._load_images(image_paths)
    
    def _render_virtual_images(self, virtual_paths: List[str]) -> torch.Tensor:
        """
        渲染虚拟图像路径对应的图像
        
        这个方法在_load_images中被调用，此时我们有当前数据项的上下文
        使用与__getitem__中相同的相机位姿生成逻辑，确保一致性
        """
        # 这里需要获取当前数据项的信息
        # 由于这是在_load_images中调用的，我们需要从虚拟路径中提取信息

        # 提取模型名称和视图索引
        model_name = virtual_paths[0].split("//")[1].split("/")[0]
        view_indices = []

        for path in virtual_paths:
            index = self._extract_image_index(path)
            if index is not None:
                view_indices.append(index)

        # 从当前处理的数据项中获取已加载的网格数据
        # 这需要在__getitem__中设置上下文
        if hasattr(self, '_current_data_item') and hasattr(self, '_current_gt_mesh_data'):
            if hasattr(self, '_current_camera_poses_list'):
                camera_poses = self._current_camera_poses_list
            else:
                max_view_index = max(view_indices) if view_indices else -1
                seed = abs(hash(model_name)) % (2**32 - 1)
                camera_poses = self._generate_camera_poses(max_view_index + 1, seed=seed) if max_view_index >= 0 else []

            # 使用已加载的网格数据进行渲染
            rendered_images = self._render_images_from_mesh_data(
                self._current_gt_mesh_data, camera_poses, view_indices
            )

            return rendered_images

        raise RuntimeError(
            "无法获取当前数据项或网格上下文，House3KDataset 无法渲染虚拟图像。"
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据样本
        
        重写以支持动态渲染和相机位姿生成
        """
        # print(idx,"worker:",torch.utils.data.get_worker_info(),self.split)
        data_item = self.data_list[idx]
        
        # 设置当前数据项上下文，供_load_images使用
        self._current_data_item = data_item
        self._current_gt_mesh_data = None  # 将在加载后设置
        
        try:
            # 获取网格路径并加载网格数据（先加载，供渲染使用）
            mesh_path = self._get_mesh_path(data_item)
            gt_mesh_data = self._load_mesh_data(
                mesh_path,
                normalize_method=self.normalize_method,
                num_samples=self.num_samples,
            )
            
            # 设置网格数据上下文，供渲染使用
            self._current_gt_mesh_data = gt_mesh_data
            
            # 获取可用图像路径
            available_image_paths = self._get_image_paths(data_item)
            
            # 选择初始视图并获取对应索引
            selected_image_paths, selected_indices = self._select_initial_images(available_image_paths)

            # 预先生成相机位姿，避免在渲染时重复计算
            if selected_indices:
                max_view_index = max(selected_indices)
                seed = abs(hash(data_item["model_name"])) % (2**32 - 1)
                camera_poses_list = self._generate_camera_poses(max_view_index + 1, seed=seed)
            else:
                camera_poses_list = []

            self._current_camera_poses_list = camera_poses_list

            # 加载图像（此时gt_mesh_data已可用于渲染）
            initial_images = self._load_images(selected_image_paths)

            # 只选择对应的相机位姿
            selected_camera_poses = [camera_poses_list[i] for i in selected_indices]
            
            # 将相机位姿转换为张量格式，与渲染时使用的格式一致
            camera_poses_tensor = []
            for pose in selected_camera_poses:
                # 转换为 [x, y, z, qx, qy, qz, qw] 格式
                # print(pose)
                pose_tensor = torch.tensor(
                    pose["position"] + pose["quaternion"], 
                    dtype=torch.float32
                )
                # print(pose_tensor)
                camera_poses_tensor.append(pose_tensor)
            
            camera_poses = torch.stack(camera_poses_tensor) if camera_poses_tensor else torch.empty(0, 7)
            # print(camera_poses)
            result = {
                "initial_images": initial_images,
                "gt_mesh_data": gt_mesh_data,
                "camera_poses": camera_poses,
                "mesh_path": mesh_path,
                # "dataset_type": self.__class__.__name__,
                "batch_name": data_item["batch_name"],
                "set_name": data_item["set_name"],
                "model_name": data_item["model_name"],
            }

            metadata = {
                "data_item": data_item,
                "selected_indices": selected_indices,
                "camera_poses_list": camera_poses_list,
            }
            gt_targets = self._build_gt_targets(gt_mesh_data, camera_poses, metadata)
            if gt_targets:
                gt_mesh_data.update(gt_targets)
                depth_z_value = gt_targets.get("depth_z")
                if depth_z_value is not None:
                    result["depth_z"] = depth_z_value
                depth_viz_value = gt_targets.get("depth_z_viz")
                if depth_viz_value is not None:
                    result["depth_z_viz"] = depth_viz_value

            return result
            
        finally:
            # 清理上下文
            if hasattr(self, '_current_data_item'):
                delattr(self, '_current_data_item')
            if hasattr(self, '_current_gt_mesh_data'):
                delattr(self, '_current_gt_mesh_data')
            if hasattr(self, '_current_camera_poses_list'):
                delattr(self, '_current_camera_poses_list')

    def _build_gt_targets(
        self,
        gt_mesh_data: Dict[str, torch.Tensor],
        camera_poses: Optional[torch.Tensor],
        metadata: Dict,
    ) -> Dict[str, torch.Tensor]:
        if camera_poses is None or camera_poses.numel() == 0:
            return {}

        renderer = self._get_renderer()
        if renderer is None:
            return {}

        normalized_mesh = gt_mesh_data.get("normalized_mesh")
        if normalized_mesh is None:
            return {}

        # Ensure poses include batch dimension expected by renderer helper.
        poses_with_batch = camera_poses.unsqueeze(0) if camera_poses.dim() == 2 else camera_poses
        point_maps, valid_masks = render_gt_point_maps(
            renderer=renderer,
            mesh_batch=normalized_mesh,
            camera_poses=poses_with_batch,
            output_device=torch.device("cpu"),
        )

        # Remove batch dimension for dataset sample.
        point_maps = point_maps.squeeze(0).contiguous()
        valid_masks = valid_masks.squeeze(0).contiguous()
        depth_z = world_points_to_camera_depth(
            point_maps,
            camera_poses,
            valid_masks=valid_masks,
        )
        depth_z_viz = normalize_depth_for_visualization(depth_z, valid_masks)

        return {
            "gt_point_maps": point_maps.to(dtype=torch.float32),
            "gt_valid_masks": valid_masks.to(dtype=torch.bool),
            "depth_z": depth_z,
            "depth_z_viz": depth_z_viz,
        }

    @property
    def dataset_info(self) -> Dict:
        """返回数据集信息"""
        base_info = super().dataset_info
        base_info.update({
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "use_cache": self.use_cache,
            "dynamic_rendering": True,
            "has_prerendered_images": False,
        })
        return base_info
