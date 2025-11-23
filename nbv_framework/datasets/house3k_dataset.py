"""
House3K数据集加载器
用于加载House3K_obj数据集，支持从3D网格动态生成多视角图像
"""

import os
import hashlib
import glob
import random
import torch
from pathlib import Path
from typing import List, Dict, Optional, Sequence, Union
from torch.utils.data import get_worker_info
from .base_dataset import BaseDataset
from ..utils.camera_utils import (
    pose_dict_to_tensor,
    position_to_pose_tensor,
    world_points_to_camera_depth,
    normalize_depth_for_visualization,
)
from ..utils.render_utils import render_gt_point_maps
from .house3k_utils import (
    find_batch_directories,
    scan_house3k_batches,
    split_house3k_dataset,
)
from nbv_framework.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


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
        val_ratio: float = 0.2,
        max_meshes: int = None,
        use_cache: bool = True,
        view_sampling_mode: str = "deterministic_per_call",
        view_sampling_seed: Optional[int] = None,
        process_rank: int = 0,
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
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            max_meshes: 全局最大mesh数量限制（用于控制训练规模）
            use_cache: 是否使用缓存加速数据加载
            view_sampling_mode: 视角采样模式，支持 fixed / deterministic_per_call / fully_random
            view_sampling_seed: 视角采样的基础种子（可选）；未提供则回退到 dataset seed
            process_rank: 当前进程的全局rank，用于分布式场景下区分不同GPU
            manual_camera_position: 手动指定的相机位置，支持单个位置、列表或按模型名称/索引映射
            manual_camera_look_at: 手动指定的相机朝向目标点，格式同上，默认为原点
            use_manual_camera: 是否启用手动相机逻辑
            **kwargs: 其他参数
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.max_meshes = max_meshes
        self.use_cache = use_cache
        self.use_manual_camera = use_manual_camera
        self.manual_camera_position = manual_camera_position
        self.manual_camera_look_at = manual_camera_look_at
        self.view_sampling_mode = str(view_sampling_mode).lower()
        self.view_sampling_seed = view_sampling_seed
        self.process_rank = int(process_rank)

        # 验证分割比例，允许浮点运算带来的微小负数
        if self.test_ratio < 0:
            if self.test_ratio < -1e-6:
                raise ValueError(f"Invalid split ratios: train={train_ratio}, val={val_ratio}")
            LOGGER.warning(
                "Split ratios sum to ~1 within tolerance; clamping test ratio to 0 "
                "(train=%s, val=%s, computed_test=%s)",
                train_ratio,
                val_ratio,
                self.test_ratio,
            )
            self.test_ratio = 0.0
        
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
        LOGGER.info("正在扫描House3K数据集: %s", self.data_root)
        data_root_path = Path(self.data_root)

        # 1. 查找所有批次目录
        batch_dirs = find_batch_directories(data_root_path)
        LOGGER.info("找到 %d 个批次目录: %s", len(batch_dirs), [d.name for d in batch_dirs])

        all_objects, total_scanned = scan_house3k_batches(batch_dirs, logger=LOGGER)
        LOGGER.info(
            "[House3K数据集] 总共扫描 %d 个3D模型，其中 %d 个有完整纹理",
            total_scanned,
            len(all_objects),
        )
        LOGGER.info("[House3K数据集] 最终加载 %d 个有效3D模型", len(all_objects))
        
        # 4. 全局mesh数量限制
        if self.max_meshes and len(all_objects) > self.max_meshes:
            original_count = len(all_objects)
            rng = random.Random(42)
            rng.shuffle(all_objects)
            all_objects = all_objects[:self.max_meshes]
            LOGGER.info(
                "[House3K数据集] 应用全局mesh限制，从 %d 个减少到 %d 个",
                original_count,
                self.max_meshes,
            )
        
        # 5. 按分割比例划分数据集
        split_objects, split_stats = split_house3k_dataset(
            all_objects,
            split=self.split,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
        )

        LOGGER.info(
            "数据集分割 - 总计: %d, 训练: %d, 验证: %d, 测试: %d",
            split_stats["total"],
            split_stats["train"],
            split_stats["val"],
            split_stats["test"],
        )
        LOGGER.info("当前分割 '%s': 加载了 %d 个样本", self.split, split_stats["current_split"])

        return split_objects
    
    def _get_mesh_path(self, data_item: Dict) -> str:
        """获取网格文件路径"""
        return data_item["obj_path"]
    
    def _get_image_paths(self, data_item: Dict) -> List[str]:
        """
        返回可用视图的符号标识列表。

        House3K 样本的图像在加载时即时渲染，因此这里不返回真实文件路径，
        而是生成用于确定性采样的稳定标识。
        """
        model_name = data_item["model_name"]
        return [self._view_token(model_name, i) for i in range(self._num_candidate_views())]

    def _view_token(self, model_name: str, index: int) -> str:
        """为指定模型的视图生成稳定且可解析的标识符。"""
        return f"{model_name}#view_{index:03d}"

    def _num_candidate_views(self) -> int:
        """返回每个网格可用的候选视图数量。"""
        return max(20, self.num_initial_views * 2)

    def _sample_view_indices(self, model_name: str, num_candidate_views: int) -> List[int]:
        """
        根据模型名称和当前 epoch 均匀采样初始视图索引，保持与基类相同的确定性行为。
        """
        if num_candidate_views < self.num_initial_views:
            raise ValueError(
                f"Not enough candidate views ({num_candidate_views}) for initial selection "
                f"({self.num_initial_views})."
            )

        mode = self.view_sampling_mode
        base_seed = (
            self.view_sampling_seed
            if self.view_sampling_seed is not None
            else (self._base_seed if self._base_seed is not None else 0)
        )

        if mode == "fixed":
            # 完全固定：同一模型始终返回同一视角；不随 epoch/进程/worker 变化
            seed_material = f"{model_name}|{base_seed}"
            digest = hashlib.md5(seed_material.encode("utf-8")).hexdigest()
            seed = int(digest, 16) % (2 ** 32 - 1)
            rng = random.Random(seed)
            return sorted(rng.sample(range(num_candidate_views), self.num_initial_views))

        if mode == "deterministic_per_call":
            # 可重现的“每次调用随机”：跨 epoch 数据一致，不同 worker/进程打散
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            seed_material = f"{model_name}|{worker_id}|{self.process_rank}|{base_seed}"
            digest = hashlib.md5(seed_material.encode("utf-8")).hexdigest()
            seed = int(digest, 16) % (2 ** 32 - 1)
            rng = random.Random(seed)
            return sorted(rng.sample(range(num_candidate_views), self.num_initial_views))

        if mode == "fully_random":
            # 完全随机：不可重现
            rng = random.SystemRandom()
            return sorted(rng.sample(range(num_candidate_views), self.num_initial_views))

        raise ValueError(f"Unknown view_sampling_mode: {self.view_sampling_mode}")

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
    
    def _extract_image_index(self, image_path: str) -> Optional[int]:
        """从视图标识符中提取索引。"""
        marker = "view_"
        if marker in image_path:
            try:
                suffix = image_path.split(marker, 1)[1]
                digits = ""
                for char in suffix:
                    if char.isdigit():
                        digits += char
                    else:
                        break
                if digits:
                    return int(digits)
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

            if torch.cuda.is_available():
                device_index = min(
                    max(int(self.process_rank), 0),
                    torch.cuda.device_count() - 1,
                )
                device = torch.device(f"cuda:{device_index}")
            else:
                device = torch.device("cpu")
            self._renderer = DifferentiableRenderer(
                image_size=self.image_size,
                device=device
            )

        return self._renderer

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
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        value = self._resolve_manual_config(self.manual_camera_position, idx, data_item)
        if value is None:
            return None

        positions = torch.as_tensor(value, dtype=torch.float32, device=device)
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
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        value = self._resolve_manual_config(self.manual_camera_look_at, idx, data_item)
        if value is None:
            return None

        look_at = torch.as_tensor(value, dtype=torch.float32, device=device)
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
    
    def render_custom_view(
        self,
        idx: int,
        position: Union[Sequence[float], torch.Tensor],
        *,
        look_at: Optional[Union[Sequence[float], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        根据自定义相机位置渲染图像，并自动计算朝向。

        Args:
            idx: 数据集中样本的索引。
            position: 相机位置 (x, y, z)，可以是长度为3的序列或张量。
            look_at: 可选的目标点 (x, y, z)，默认为原点。

        Returns:
            包含渲染图像、相机位姿以及基础元数据的字典。
        """
        data_item = self.data_list[idx]
        mesh_path = self._get_mesh_path(data_item)
        gt_mesh_data = self._load_mesh_data(
            mesh_path,
            normalize_method=self.normalize_method,
            num_samples=self.num_samples,
        )

        renderer = self._get_renderer()
        device = renderer.device

        position_tensor = torch.as_tensor(position, dtype=torch.float32)
        if position_tensor.ndim == 1:
            if position_tensor.numel() != 3:
                raise ValueError(
                    f"position expects 3 values, but received shape {tuple(position_tensor.shape)}"
                )
            position_tensor = position_tensor.unsqueeze(0)
        elif position_tensor.ndim != 2 or position_tensor.shape[1] != 3:
            raise ValueError(
                f"position must have shape [N, 3] or [3], but received {tuple(position_tensor.shape)}"
            )
        position_tensor = position_tensor.to(device)

        look_at_tensor: Optional[torch.Tensor] = None
        if look_at is not None:
            look_at_tensor = torch.as_tensor(look_at, dtype=torch.float32)
            if look_at_tensor.ndim == 1:
                if look_at_tensor.numel() != 3:
                    raise ValueError(
                        f"look_at expects 3 values, but received shape {tuple(look_at_tensor.shape)}"
                    )
                look_at_tensor = look_at_tensor.unsqueeze(0)
            elif look_at_tensor.ndim != 2 or look_at_tensor.shape[1] != 3:
                raise ValueError(
                    f"look_at must have shape [N, 3] or [3], but received {tuple(look_at_tensor.shape)}"
                )
            look_at_tensor = look_at_tensor.to(device)

        camera_pose_tensor = position_to_pose_tensor(
            position_tensor,
            up_axis=self.up_axis,
            look_at=look_at_tensor,
        )

        mesh = gt_mesh_data["normalized_mesh"].to(device)
        num_views = camera_pose_tensor.shape[0]
        meshes_batch = mesh.extend(num_views)

        with torch.no_grad():
            rendered_images = renderer.forward(
                gt_mesh=meshes_batch,
                camera_poses=camera_pose_tensor,
                pose_format="cartesian",
                fov=60.0,
            )

        rendered_images = rendered_images.detach().cpu()
        camera_pose_cpu = camera_pose_tensor.detach().cpu()

        pose_dicts = [
            {
                "position": camera_pose_cpu[i, :3].tolist(),
                "quaternion": camera_pose_cpu[i, 3:].tolist(),
            }
            for i in range(camera_pose_cpu.shape[0])
        ]

        result = {
            "rendered_images": rendered_images,
            "camera_poses": camera_pose_cpu,
            "gt_mesh_data": gt_mesh_data,
            "mesh_path": mesh_path,
            "batch_name": data_item["batch_name"],
            "set_name": data_item["set_name"],
            "model_name": data_item["model_name"],
        }
        result["camera_pose_dicts"] = pose_dicts

        if look_at_tensor is not None:
            result["look_at"] = look_at_tensor.detach().cpu()

        return result
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回结构化样本：inputs/targets/mesh/meta。"""
        data_item = self.data_list[idx]
        mesh_path = self._get_mesh_path(data_item)
        gt_mesh_data = self._load_mesh_data(
            mesh_path,
            normalize_method=self.normalize_method,
            num_samples=self.num_samples,
        )

        model_name = data_item["model_name"]
        num_candidate_views = self._num_candidate_views()
        selected_indices = self._sample_view_indices(model_name, num_candidate_views)

        camera_poses_list: List[Dict] = []
        initial_images: torch.Tensor
        camera_poses_tensor: torch.Tensor

        manual_mode = False
        if self.use_manual_camera:
            renderer = self._get_renderer()
            manual_positions = self._resolve_manual_camera_positions(
                idx, data_item, renderer.device
            )
            if manual_positions is not None:
                manual_mode = True
                manual_look_at = self._resolve_manual_camera_look_at(
                    idx, data_item, renderer.device
                )
                manual_camera_pose = position_to_pose_tensor(
                    manual_positions,
                    up_axis=self.up_axis,
                    look_at=manual_look_at,
                )
                mesh = gt_mesh_data["normalized_mesh"].to(renderer.device)
                meshes_batch = mesh.extend(manual_camera_pose.shape[0])
                with torch.no_grad():
                    rendered = renderer.forward(
                        gt_mesh=meshes_batch,
                        camera_poses=manual_camera_pose,
                        pose_format="cartesian",
                        fov=60.0,
                    )
                initial_images = rendered.detach().cpu()
                camera_poses_tensor = manual_camera_pose.detach().cpu()
                selected_indices = list(range(camera_poses_tensor.shape[0]))
                camera_poses_list = [
                    {
                        "position": camera_poses_tensor[i, :3].tolist(),
                        "quaternion": camera_poses_tensor[i, 3:].tolist(),
                    }
                    for i in range(camera_poses_tensor.shape[0])
                ]

        if not manual_mode:
            if selected_indices:
                seed = abs(hash(model_name)) % (2 ** 32 - 1)
                camera_poses_list = self._generate_camera_poses(
                    num_candidate_views, seed=seed
                )

                initial_images = self._render_images_from_mesh_data(
                    gt_mesh_data=gt_mesh_data,
                    camera_poses=camera_poses_list,
                    selected_indices=selected_indices,
                )

                selected_camera_poses = [camera_poses_list[i] for i in selected_indices]
                camera_pose_rows = [
                    torch.tensor(pose["position"] + pose["quaternion"], dtype=torch.float32)
                    for pose in selected_camera_poses
                ]
                camera_poses_tensor = torch.stack(camera_pose_rows, dim=0)
            else:
                LOGGER.warning(
                    "未选择任何相机位姿，模型名称: %s",
                    model_name,
                )

        gt_supervision = dict(gt_mesh_data)
        original_mesh = gt_supervision.pop("original_mesh", None)
        normalized_mesh = gt_supervision.pop("normalized_mesh", None)
        gt_supervision.pop("mesh_path", None)

        metadata = {
            "data_item": data_item,
            "selected_indices": selected_indices,
            "camera_poses_list": camera_poses_list,
            "mesh_path": mesh_path,
            "batch_name": data_item["batch_name"],
            "set_name": data_item["set_name"],
            "model_name": model_name,
            "normalize_method": gt_supervision.get("normalize_method"),
            "num_samples": gt_supervision.get("num_samples"),
        }

        gt_targets = self._build_gt_targets(gt_mesh_data, camera_poses_tensor, metadata)
        if gt_targets:
            gt_supervision.update(gt_targets)

        inputs = {
            "images": initial_images,
            "camera_poses": camera_poses_tensor,
        }
        depth_z = gt_supervision.get("depth_z")
        if depth_z is not None:
            inputs["depth_z"] = depth_z

        sample = {
            "inputs": inputs,
            "targets": {
                "gt_mesh_data": gt_supervision,
            },
            "mesh": {
                "original": original_mesh,
                "normalized": normalized_mesh,
            },
            "meta": metadata,
        }

        return sample

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
            "manual_camera_enabled": self.use_manual_camera,
        })
        return base_info
