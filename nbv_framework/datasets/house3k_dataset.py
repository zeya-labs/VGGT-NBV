"""
House3K 数据集加载器。

负责扫描网格文件、生成相机位姿并返回训练所需的结构化样本。
渲染逻辑已移至训练阶段的 GPU 路径执行。
"""

import hashlib
import random
import torch
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
from .base_dataset import BaseDataset
from ..cache.render_cache import RenderCache
from ..utils.camera_utils import (
    position_to_pose_tensor,
)
from .house3k_utils import (
    find_batch_directories,
    scan_house3k_batches,
    split_house3k_dataset,
)
from loguru import logger


class House3KDataset(BaseDataset):
    """
    House3K 数据集加载器。

    数据结构:
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

    特点:
    - 只有 3D 网格文件，没有预渲染图像
    - 相机位姿在加载时动态生成
    - 训练/验证/测试分割
    """
    
    def __init__(
        self,
        data_root: str,
        num_initial_views: int = 4,
        image_size: int = 518,
        split: str = "train",
        normalize_method: str = "quantile",
        num_samples: int = 10000,
        camera_radius: float = 2.6,
        camera_radius_variation: float = 0.0,
        camera_radius_mode: str = "random",
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        max_meshes: int = None,
        view_sampling_mode: str = "deterministic_per_call",
        render_cache_enabled: bool = True,
        render_cache_root: Optional[str] = None,
        render_cache_version: int = 1,
        render_cache_signature: Optional[str] = None,
        manual_camera_position: Optional[
            Union[
                Sequence[float],
                Sequence[Sequence[float]],
                Dict[Union[str, int], Sequence[Sequence[float]]],
            ]
        ] = None,
        manual_camera_look_at: Optional[
            Union[
                Sequence[float],
                Sequence[Sequence[float]],
                Dict[Union[str, int], Sequence[Sequence[float]]],
            ]
        ] = None,
        use_manual_camera: bool = False,
        up_axis: str = "Y",
        seed: int = 42,
        **kwargs,
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
            camera_radius: 相机半径基准
            camera_radius_variation: 相机半径随机偏移范围
            camera_radius_mode: 相机半径采样模式 ("random"/"constant")
                - random: 在 [base_radius - variation, base_radius + variation] 内随机采样
                - constant: 固定为 base_radius（忽略 variation）
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            max_meshes: 全局最大mesh数量限制（用于控制训练规模）
            view_sampling_mode: 视角采样模式，支持 fixed / deterministic_per_call / fully_random
            manual_camera_position: 手动指定的相机位置，支持单个位置、列表或按模型名称/索引映射
            manual_camera_look_at: 手动指定的相机朝向目标点，格式同上，默认为原点
            use_manual_camera: 是否启用手动相机逻辑
            up_axis: 相机向上轴，支持 "Y" (默认) 或 "Z"
            seed: 随机种子，用于相机位姿生成
            **kwargs: 其他参数
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.max_meshes = max_meshes
        self.use_manual_camera = use_manual_camera
        self.manual_camera_position = manual_camera_position
        self.manual_camera_look_at = manual_camera_look_at
        self.view_sampling_mode = str(view_sampling_mode).lower()
        self.camera_radius = float(camera_radius)
        self.camera_radius_variation = float(camera_radius_variation)
        self.camera_radius_mode = str(camera_radius_mode).lower()
        self._render_cache_enabled = bool(render_cache_enabled)
        self._render_cache_root = Path(render_cache_root) if render_cache_root else None
        self._render_cache_version = int(render_cache_version)
        self._render_cache_signature = render_cache_signature
        self._render_cache = None

        # 验证分割比例，允许浮点运算带来的微小负数
        if self.test_ratio < 0:
            if self.test_ratio < -1e-6:
                raise ValueError(f"Invalid split ratios: train={train_ratio}, val={val_ratio}")
            logger.warning(
                f"Split ratios sum to ~1 within tolerance; clamping test ratio to 0 "
                f"(train={train_ratio}, val={val_ratio}, computed_test={self.test_ratio})"
            )
            self.test_ratio = 0.0
        
        self._camera_generator = None
        
        super().__init__(
            data_root=data_root,
            num_initial_views=num_initial_views,
            image_size=image_size,
            split=split,
            normalize_method=normalize_method,
            num_samples=num_samples,
            up_axis=up_axis,
            seed=seed,
            **kwargs
        )

        if self._render_cache_enabled:
            if self._render_cache_signature is None:
                self._render_cache_signature = RenderCache.build_signature(
                    version=self._render_cache_version,
                    image_size=self.image_size,
                    fov=60.0,
                    faces_per_pixel=4,
                    blur_radius=1e-5,
                    perspective_correct=False,
                    cull_backfaces=False,
                )
            self._render_cache = RenderCache(
                renderer=None,
                root=self._render_cache_root,
                version=self._render_cache_version,
                render_signature=self._render_cache_signature,
            )
    
    def _load_data_list(self) -> List[Dict]:
        """
        加载House3K数据集列表

        扫描所有批次目录，找到可用 .obj 文件，按比例分割。
        """
        logger.info(f"正在扫描House3K数据集: {self.data_root}，seed={self.seed}")
        data_root_path = Path(self.data_root)

        # 查找所有批次目录
        batch_dirs = find_batch_directories(data_root_path)
        logger.info(f"找到 {len(batch_dirs)} 个批次目录: {[d.name for d in batch_dirs]}")  

        all_objects, total_scanned = scan_house3k_batches(batch_dirs, logger=logger)
        logger.info(f"[House3K数据集] 总共扫描 {total_scanned} 个3D模型")
        logger.info(f"[House3K数据集] 加载 {len(all_objects)} 个有效3D模型")
        
        # 全局 mesh 数量限制
        if self.max_meshes and len(all_objects) > self.max_meshes:
            rng = random.Random(self.seed)
            rng.shuffle(all_objects)
            all_objects = all_objects[:self.max_meshes]
            logger.info(
                f"[House3K数据集] 应用全局mesh限制，从 {len(all_objects)} 个减少到 {self.max_meshes} 个"
            )
        
        # 按分割比例划分数据集
        split_objects, split_stats = split_house3k_dataset(
            all_objects,
            split=self.split,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
        )

        logger.info(
            f"数据集分割 - 总计: {split_stats['total']}, 训练: {split_stats['train']}, "
            f"验证: {split_stats['val']}, 测试: {split_stats['test']}"
        )
        logger.info(f"当前分割 '{self.split}': 加载了 {split_stats['current_split']} 个样本")

        return split_objects
    
    def _get_mesh_path(self, data_item: Dict) -> str:
        """获取网格文件路径"""
        return data_item["obj_path"]
    
    def _get_image_paths(self, data_item: Dict) -> List[str]:
        """
        House3K 不提供预渲染图像路径，保留该接口以满足基类约束。
        """
        return []

    def set_epoch(self, epoch: int) -> None:
        """更新 epoch，保持 BaseDataset 的行为。"""
        super().set_epoch(epoch)
    
    def _get_camera_poses_path(self, data_item: Dict) -> Optional[str]:
        """
        获取相机位姿文件路径
        
        House3K数据集没有相机位姿文件，返回None
        相机位姿将在运行时动态生成
        """
        return None
    
    def _resolve_view_seed(self, model_name: str, idx: int) -> Optional[int]:
        mode = self.view_sampling_mode
        base_seed = self.seed

        if mode == "fixed":
            seed_material = f"{model_name}|{base_seed}"
        elif mode == "deterministic_per_call":
            seed_material = f"{model_name}|{idx}|{base_seed}"
        elif mode == "fully_random":
            return None
        else:
            raise ValueError(f"Unknown view_sampling_mode: {self.view_sampling_mode}")

        digest = hashlib.md5(seed_material.encode("utf-8")).hexdigest()
        return int(digest, 16) % (2 ** 32 - 1)

    def _generate_camera_poses(self, num_views: int, seed: Optional[int] = None) -> List[Dict[str, List[float]]]:
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

        return self._camera_generator.generate_random_camera_poses(
            num_views,
            seed=seed,
            hemisphere="upper",
            base_radius=self.camera_radius,
            radius_variation=self.camera_radius_variation,
            radius_mode=self.camera_radius_mode,
        )

    def _poses_tensor_to_list(self, camera_poses: torch.Tensor) -> List[Dict[str, List[float]]]:
        return [
            {
                "position": camera_poses[i, :3].tolist(),
                "quaternion": camera_poses[i, 3:].tolist(),
            }
            for i in range(camera_poses.shape[0])
        ]

    def _build_camera_poses(
        self,
        idx: int,
        data_item: Dict,
        model_name: str,
        num_views: int,
    ) -> Tuple[torch.Tensor, List[Dict[str, List[float]]]]:
        if self.use_manual_camera:
            manual_positions = self._resolve_manual_camera_positions(idx, data_item)
            if manual_positions is not None:
                manual_look_at = self._resolve_manual_camera_look_at(idx, data_item)
                manual_camera_pose = position_to_pose_tensor(
                    manual_positions,
                    up_axis=self.up_axis,
                    look_at=manual_look_at,
                )
                return (
                    manual_camera_pose.detach(),
                    self._poses_tensor_to_list(manual_camera_pose),
                )

        seed = self._resolve_view_seed(model_name, idx)
        camera_poses_list = self._generate_camera_poses(num_views, seed=seed)
        camera_pose_rows = [
            torch.tensor(pose["position"] + pose["quaternion"], dtype=torch.float32)
            for pose in camera_poses_list
        ]
        camera_poses_tensor = torch.stack(camera_pose_rows, dim=0)
        return camera_poses_tensor, camera_poses_list

    def _resolve_manual_config(self, config, idx: int, data_item: Dict):
        if config is None:
            return None
        if callable(config):
            return config(data_item, idx)
        if isinstance(config, dict):
            keys_to_try = [
                data_item.get("model_name"),
                (data_item.get("batch_name"), data_item.get("set_name"), data_item.get("model_name")),
                idx,
            ]
            for key in keys_to_try:
                if key in config:
                    return config[key]
            return None
        return config

    def _resolve_manual_camera_positions(
        self,
        idx: int,
        data_item: Dict,
    ) -> Optional[torch.Tensor]:
        value = self._resolve_manual_config(self.manual_camera_position, idx, data_item)
        if value is None:
            return None

        positions = torch.as_tensor(value, dtype=torch.float32)
        if positions.ndim == 1:
            if positions.numel() != 3:
                raise ValueError(
                    f"manual camera position expects 3 values, but received shape {tuple(positions.shape)}"
                )
            positions = positions.unsqueeze(0)
        elif positions.ndim == 2 and positions.shape[1] == 3:
            pass
        else:
            raise ValueError(
                f"manual camera position must have shape [N, 3] or [3], got {tuple(positions.shape)}"
            )

        return positions

    def _resolve_manual_camera_look_at(
        self,
        idx: int,
        data_item: Dict,
    ) -> Optional[torch.Tensor]:
        value = self._resolve_manual_config(self.manual_camera_look_at, idx, data_item)
        if value is None:
            return None

        look_at = torch.as_tensor(value, dtype=torch.float32)
        if look_at.ndim == 1:
            if look_at.numel() != 3:
                raise ValueError(
                    f"manual camera look_at expects 3 values, but received shape {tuple(look_at.shape)}"
                )
            look_at = look_at.unsqueeze(0)
        elif look_at.ndim == 2 and look_at.shape[1] == 3:
            pass
        else:
            raise ValueError(
                f"manual camera look_at must have shape [N, 3] or [3], got {tuple(look_at.shape)}"
            )

        return look_at
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回结构化样本：inputs/targets/mesh/meta。"""
        data_item = self.data_list[idx]
        '''
        from icecream import ic
        ic(data_item)
        {
            'batch_name': 'BATCH_3',
            'has_texture': True,
            'model_name': 'BAT3_SETD_HOUSE33',
            'mtl_path': '/mnt/sdb/chenmohan/VGGT-NBV/models/House3K_obj/BATCH_3/SetD/BAT3_SETD_HOUSE33.mtl',
            'obj_path': '/mnt/sdb/chenmohan/VGGT-NBV/models/House3K_obj/BATCH_3/SetD/BAT3_SETD_HOUSE33.obj',
            'set_name': 'SetD',
            'set_path': '/mnt/sdb/chenmohan/VGGT-NBV/models/House3K_obj/BATCH_3/SetD'
        }
        '''
        mesh_path = self._get_mesh_path(data_item)
        # logger.info(f"Loading mesh from {mesh_path}")
        gt_mesh_data = self._load_mesh_data(
            mesh_path,
            normalize_method=self.normalize_method,
            num_samples=self.num_samples,
        )
        
        model_name = data_item["model_name"]
        camera_poses_tensor, camera_poses_list = self._build_camera_poses(
            idx,
            data_item,
            model_name,
            self.num_initial_views,
        )

        gt_supervision = dict(gt_mesh_data)
        gt_supervision.pop("original_mesh", None)
        gt_supervision.pop("normalized_mesh", None)
        gt_supervision.pop("mesh_path", None)

        cache_hit = False
        if self._render_cache is not None:
            cache_paths = self._render_cache.build_paths(
                mesh_paths=[mesh_path],
                normalize_methods=[self.normalize_method],
                camera_poses_batch=camera_poses_tensor.unsqueeze(0),
            )
            if cache_paths:
                cache_payload = self._render_cache.load_item(
                    cache_path=cache_paths[0],
                    base_gt_mesh_data=gt_supervision,
                )
                if cache_payload is not None:
                    _, cached_images, cached_gt_mesh_data = cache_payload
                    gt_supervision = cached_gt_mesh_data
                    cache_hit = True
        metadata = {
            "data_item": data_item,
            # "camera_poses_list": camera_poses_list,
            "mesh_path": mesh_path,
            "batch_name": data_item["batch_name"],
            "set_name": data_item["set_name"],
            "model_name": model_name,
            "normalize_method": gt_supervision.get("normalize_method"),
            "num_samples": gt_supervision.get("num_samples"),
            "cache_hit": cache_hit,
        }

        inputs = {
            "camera_poses": camera_poses_tensor,
        }
        if cache_hit:
            inputs["images"] = cached_images

        sample = {
            "inputs": inputs,
            "targets": {
                "gt_mesh_data": gt_supervision,
            },
            "mesh": {
                "normalized": gt_mesh_data.get("normalized_mesh"),
            },
            # "mesh": {
            #     "original": original_mesh,
            #     "normalized": normalized_mesh,
            # },
            "meta": metadata,
        }

        return sample

    @property
    def dataset_info(self) -> Dict:
        """返回数据集信息"""
        base_info = super().dataset_info
        base_info.update({
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "dynamic_rendering": True,
            "has_prerendered_images": False,
            "manual_camera_enabled": self.use_manual_camera,
            "camera_radius": self.camera_radius,
            "camera_radius_variation": self.camera_radius_variation,
            "camera_radius_mode": self.camera_radius_mode,
        })
        return base_info
