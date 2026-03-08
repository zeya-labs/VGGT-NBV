"""
House3K 数据集加载器。

负责扫描网格文件、生成相机位姿并返回训练所需的结构化样本。
渲染逻辑已移至训练阶段的 GPU 路径执行。
"""

import torch
from typing import Dict, List
from .base_dataset import BaseDataset
from .house3k_camera import House3KCameraConfig, House3KCameraPlanner, ManualCameraValue
from .house3k_index_builder import House3KIndexConfig, build_house3k_split_objects
from .house3k_sample_builder import build_house3k_sample
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
        manual_camera_position: ManualCameraValue = None,
        manual_camera_look_at: ManualCameraValue = None,
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

        # 验证分割比例，允许浮点运算带来的微小负数
        if self.test_ratio < 0:
            if self.test_ratio < -1e-6:
                raise ValueError(f"Invalid split ratios: train={train_ratio}, val={val_ratio}")
            logger.warning(
                f"Split ratios sum to ~1 within tolerance; clamping test ratio to 0 "
                f"(train={train_ratio}, val={val_ratio}, computed_test={self.test_ratio})"
            )
            self.test_ratio = 0.0
        
        self._camera_planner = House3KCameraPlanner(
            House3KCameraConfig(
                up_axis=str(up_axis).upper(),
                seed=int(seed),
                view_sampling_mode=self.view_sampling_mode,
                camera_radius=self.camera_radius,
                camera_radius_variation=self.camera_radius_variation,
                camera_radius_mode=self.camera_radius_mode,
                use_manual_camera=self.use_manual_camera,
                manual_camera_position=self.manual_camera_position,
                manual_camera_look_at=self.manual_camera_look_at,
            )
        )
        self._index_config = House3KIndexConfig(
            data_root=data_root,
            seed=int(seed),
            split=str(split),
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            max_meshes=max_meshes,
        )
        
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
    
    def _load_data_list(self) -> List[Dict]:
        """
        加载House3K数据集列表

        扫描所有批次目录，找到可用 .obj 文件，按比例分割。
        """
        return build_house3k_split_objects(self._index_config)
    
    def _get_mesh_path(self, data_item: Dict) -> str:
        """获取网格文件路径"""
        return data_item["obj_path"]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回结构化样本：inputs/targets/mesh/meta。"""
        data_item = self.data_list[idx]
        mesh_path = self._get_mesh_path(data_item)
        # logger.info(f"Loading mesh from {mesh_path}")
        gt_mesh_data = self._load_mesh_data(
            mesh_path,
            normalize_method=self.normalize_method,
            num_samples=self.num_samples,
        )
        
        model_name = data_item["model_name"]
        camera_poses_tensor, _ = self._camera_planner.build_camera_poses(
            idx=idx,
            data_item=data_item,
            model_name=model_name,
            num_views=self.num_initial_views,
        )
        return build_house3k_sample(
            data_item=data_item,
            mesh_path=mesh_path,
            gt_mesh_data=gt_mesh_data,
            camera_poses_tensor=camera_poses_tensor,
        )

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
