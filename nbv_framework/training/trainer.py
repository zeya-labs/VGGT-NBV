"""
NBV策略训练器
实现端到端的目标驱动策略学习训练流程
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, NamedTuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import WandbLogger
from lightning_fabric.utilities.apply_func import apply_to_collection
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_to_matrix

from mapanything.utils.geometry import (
    normalize_pose_translations,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    transform_pose_using_quats_and_trans_2_to_1,
)

if TYPE_CHECKING:
    from ..models import MapAnythingWrapper, BaseNBVPolicy
from ..rendering import DifferentiableRenderer
from .loss import ReconstructionLoss, ChamferDistance
from ..utils.camera_utils import (
    position_to_pose_tensor,
    world_points_to_camera_depth,
    camera_depth_z_to_world_points,
    infer_depth_backprojection_xy_signs,
)
from ..utils.mapanything_views import _compute_pose_quats_and_trans_for_across_views_in_ref_view
from ..utils.render_utils import render_gt_point_maps
from ..models.direct_reconstruction import build_recon_from_point_maps


logger = logging.getLogger(__name__)


class PoseEvaluationResult(NamedTuple):
    total_loss: torch.Tensor
    loss_components: Dict[str, float]
    new_images: torch.Tensor
    gt_mesh_data: Dict[str, torch.Tensor]
    depth_z: Optional[torch.Tensor]


class NBVTrainer(LightningModule):
    """
    NBV策略训练器

    实现完整的目标驱动策略学习训练流程。
    """

    def __init__(self,
                 vggt_wrapper: MapAnythingWrapper,
                 policy_network: BaseNBVPolicy,
                 renderer: DifferentiableRenderer,
                 loss_fn: ReconstructionLoss,
                 min_initial_views: Optional[int] = None,
                 max_initial_views: Optional[int] = None,
                 randomize_initial_views: bool = False,
                 max_epochs: int = 1000,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 log_dir: str = "runs/nbv_experiment",
                 use_epoch_seed: bool = False,
                 enable_random_baseline: bool = True):
        """
        初始化训练器

        Args:
            vggt_wrapper: 冻结的 MapAnything 基础模型
            policy_network: 可训练的NBV策略网络
            renderer: 可微分渲染器
            loss_fn: 重建质量损失函数
            min_initial_views: 训练时可用的最小初始视图数量（None 表示由 max_initial_views 或输入大小决定）
            max_initial_views: 训练时可用的最大初始视图数量（None 表示使用批次实际视图数）
            randomize_initial_views: 是否在训练步骤中随机采样初始视图数量
            learning_rate: 学习率
            weight_decay: 权重衰减
            log_dir: 训练日志/可视化输出目录
            enable_random_baseline: 是否计算随机基线视角的 Chamfer 统计
        """
        super().__init__()

        self.vggt_wrapper = vggt_wrapper
        self.policy_network = policy_network
        self.renderer = renderer
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs
        self.save_hyperparameters(
            ignore=[
                "vggt_wrapper",
                "policy_network",
                "renderer",
                "loss_fn",
            ]
        )
        self.log_dir = log_dir
        self.use_epoch_seed = use_epoch_seed
        self.enable_random_baseline = enable_random_baseline
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.min_initial_views = min_initial_views
        self.max_initial_views = max_initial_views
        self.randomize_initial_views = randomize_initial_views
        self._last_initial_view_count = 0
        self._last_initial_view_indices: Optional[torch.Tensor] = None

        # 深度反投影坐标轴符号约定（与渲染器一致）
        self._depth_backproject_xy_signs: Optional[Tuple[int, int]] = (-1,-1)

        # Debug tensors for gradient attribution (filled during training_step).
        self._last_predicted_relative_position: Optional[torch.Tensor] = None
        self._last_next_camera_pose: Optional[torch.Tensor] = None
        self._last_new_point_maps_render: Optional[torch.Tensor] = None

    # def configure_model(self):
    #     """
    #     在 Trainer setup 阶段调用，用于对模型进行 torch.compile 编译加速。
    #     注意：不要编译 renderer，PyTorch3D 的渲染器通常不兼容 inductor 编译器。
    #     """
    #     if self.policy_network is not None:
    #         logger.info("Compiling Policy Network...")
    #         # mode="reduce-overhead" 适合小网络（策略网络通常不大），可以减少 Python 调用开销
    #         # 如果策略网络很大（如 ResNet50+），改用 mode="default"
    #         self.policy_network = torch.compile(self.policy_network, mode="default")

        # 可选：尝试编译 VGGT Wrapper
        # MapAnything/VGGT 结构通常很复杂，编译可能会失败或导致启动极慢。
        # 如果 vggt_wrapper 内部是标准的 Transformer/CNN，可以尝试取消下面的注释：
        # if self.vggt_wrapper is not None:
        #     logger.info("Compiling VGGT Wrapper...")
        #     # fullgraph=False 允许在无法完全捕获图时回退到 Python
        #     self.vggt_wrapper = torch.compile(self.vggt_wrapper, mode="default", fullgraph=False)

    def _extract_scene_features(
        self,
        initial_images: torch.Tensor,
        camera_poses_batch: torch.Tensor,
        depth_z_batch: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """封装 MapAnything scene feature 提取，便于复用和测试。"""
        return self.vggt_wrapper.extract_scene_features(
            initial_images,
            camera_poses_batch,
            is_metric_scale=False,
            depth_z=depth_z_batch,
        )

    def _compute_pose_for_across_views_in_ref_view(
            self,
            views: List[Dict[str, Any]],
        ) -> torch.Tensor:
            """
            计算跨视角策略评估时的相机位姿，均转换到参考视角坐标系下。
            
            Args:
                views: 包含视图信息的字典列表。
                
            Returns:
                torch.Tensor: 形状为 (B, S, 7) 的张量。
                            S 为视图数量 (num_views)。
                            最后一维 7 为 [tx, ty, tz, qx, qy, qz, qw]。
            """
            # 1. 获取基础维度信息
            num_views = len(views)
            batch_size_per_view = views[0]["img"].shape[0]
            device = views[0]["img"].device
            dtype = views[0]["img"].dtype

            # 2. 构造全为 True 的掩码
            # 我们希望计算所有 View 的位姿，不进行随机 Dropout
            per_sample_cam_input_mask = torch.ones(
                batch_size_per_view * num_views, 
                dtype=torch.bool, 
                device=device
            )

            # 3. 调用核心函数计算相对位姿
            # pose_quats_flat: (B*S, 4)
            # pose_trans_flat: (B*S, 3)
            # 这些位姿都已经转换到了 View 0 的坐标系下（View 0 为 Identity）
            (
                pose_quats_flat, 
                pose_trans_flat, 
                _
            ) = _compute_pose_quats_and_trans_for_across_views_in_ref_view(
                views=views,
                num_views=num_views,
                device=device,
                dtype=dtype,
                batch_size_per_view=batch_size_per_view,
                per_sample_cam_input_mask=per_sample_cam_input_mask,
            )

            # 4. 拼接平移和四元数 -> (B*S, 7)
            # 顺序：[Translation (3), Quaternion (4)] -> [tx, ty, tz, qx, qy, qz, qw]
            pose_flat_7d = torch.cat([pose_trans_flat, pose_quats_flat], dim=-1)

            # 5. 重塑维度
            # 原始数据排列顺序是 view-major: [View0_Batch..., View1_Batch..., ...]
            # 先变为 (S, B, 7)
            pose_sb7 = pose_flat_7d.view(num_views, batch_size_per_view, 7)
            
            # 转置为 batch-major: (B, S, 7)
            pose_bs7 = pose_sb7.transpose(0, 1).contiguous()

            return pose_bs7

    def _compute_policy_pose(
        self,
        policy_output: torch.Tensor,
        camera_poses_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据策略输出计算绝对位姿及相关中间量。"""
        reference_position = camera_poses_batch[:, 0, :3]
        # 使用 MapAnything 同步的位姿归一化尺度，将策略输出从归一化坐标恢复到原始尺度
        # scale_factor = self._compute_pose_scale_factor(camera_poses_batch).unsqueeze(-1)
        predicted_relative_position = policy_output[:, :3]
        # absolute_position = reference_position + predicted_relative_position
        # 策略网络看到的 across-view extrinsics 是以 view0 为参考系的相对位姿；
        # 因此策略输出的相对平移更自然地处在 view0(相机0)坐标系下，而不是世界坐标系。
        # 将其旋转回世界坐标，再与世界系下的 reference_position 相加，避免“加在错误坐标系”
        # 导致视角跑飞（常见表现：渲染纯黑、Chamfer 爆炸、梯度尖峰）。
        if camera_poses_batch.ndim != 3 or camera_poses_batch.shape[-1] != 7:
            raise ValueError(
                f"camera_poses_batch expected shape [B, S, 7], got {tuple(camera_poses_batch.shape)}"
            )
        ref_quat_xyzw = camera_poses_batch[:, 0, 3:]
        quat_wxyz = ref_quat_xyzw[:, [3, 0, 1, 2]]
        rotation_w2c_row = quaternion_to_matrix(quat_wxyz)  # [B, 3, 3]
        rotation_c2w_row = rotation_w2c_row.transpose(1, 2)
        predicted_relative_position_world = torch.bmm(
            predicted_relative_position.unsqueeze(1),
            rotation_c2w_row,
        ).squeeze(1)
        absolute_position = reference_position + predicted_relative_position_world
        next_camera_pose = position_to_pose_tensor(absolute_position)
        return next_camera_pose, predicted_relative_position, absolute_position

    def _compute_pose_scale_factor(self, camera_poses_batch: torch.Tensor) -> torch.Tensor:
        """模仿 MapAnything：将所有视角变换到 view0 坐标系后，按跨视角平均范数归一化平移，返回归一化因子。

        输入 camera_poses_batch 采用本项目通用的 world->camera 约定 [x,y,z,qx,qy,qz,qw]，
        先转换到 cam2world (OpenCV, scalar-last) 再复用 MapAnything 的相对位姿计算。
        """
        if camera_poses_batch.ndim != 3 or camera_poses_batch.shape[-1] != 7:
            raise ValueError(
                f"camera_poses_batch expected shape [B, S, 7], got {tuple(camera_poses_batch.shape)}"
            )
        positions_world = camera_poses_batch[..., :3]  # camera centers in world
        quats_world_to_cam = camera_poses_batch[..., 3:]  # world->cam rotation, xyzw

        B, S, _ = positions_world.shape
        # world->cam rotation matrix
        R_wc = quaternion_to_rotation_matrix(quats_world_to_cam.reshape(-1, 4)).view(B, S, 3, 3)
        # cam2world rotation
        R_cw = R_wc.transpose(-1, -2)
        quats_cam2world = rotation_matrix_to_quaternion(R_cw.reshape(-1, 3, 3)).view(B, S, 4)
        trans_cam2world = positions_world  # cam center in world frame

        ref_quat = quats_cam2world[:, 0]  # [B, 4]
        ref_trans = trans_cam2world[:, 0]  # [B, 3]

        ref_quat_exp = ref_quat.unsqueeze(1).expand(-1, S, -1).reshape(-1, 4)
        ref_trans_exp = ref_trans.unsqueeze(1).expand(-1, S, -1).reshape(-1, 3)
        quats_flat = quats_cam2world.reshape(-1, 4)
        trans_flat = trans_cam2world.reshape(-1, 3)

        _, rel_trans_flat = transform_pose_using_quats_and_trans_2_to_1(
            ref_quat_exp,
            ref_trans_exp,
            quats_flat,
            trans_flat,
        )
        rel_trans = rel_trans_flat.view(B, S, 3)

        _, norm_factor = normalize_pose_translations(rel_trans, return_norm_factor=True)
        logger.info(
            "Pose scale factor stats — mean: %.4f, min: %.4f, max: %.4f",
            norm_factor.mean().item(),
            norm_factor.min().item(),
            norm_factor.max().item(),
        )
        return norm_factor

    def _log_camera_pose_stats(
        self,
        next_camera_pose: torch.Tensor,
        predicted_relative_position: torch.Tensor,
        step_index: Optional[int],
    ) -> None:
        """记录相机位姿统计信息。"""
        if step_index is None:
            return

        positions = next_camera_pose[:, :3]
        quaternions = next_camera_pose[:, 3:]

        position_norms = torch.norm(positions, dim=1)
        self.log(
            "Camera_pose/position_norm_mean",
            position_norms.mean(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=self.world_size > 1,
        )
        if position_norms.numel() > 1:
            self.log(
                "Camera_pose/position_norm_std",
                position_norms.std(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=self.world_size > 1,
            )

        # quaternion_norms = torch.norm(quaternions, dim=1)
        # if quaternion_norms.numel() > 1:
        #     self.log(
        #         "Camera_pose/quaternion_norm_std",
        #         quaternion_norms.std(),
        #         on_step=True,
        #         on_epoch=False,
        #         prog_bar=False,
        #         sync_dist=self.world_size > 1,
        #     )
        if quaternions.numel() > 0:
            qw_abs = quaternions[:, 3].abs()
            self.log(
                "Camera_pose/quaternion_w_abs_mean",
                qw_abs.mean(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=self.world_size > 1,
            )
            self.log(
                "Camera_pose/quaternion_w_abs_min",
                qw_abs.min(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=self.world_size > 1,
            )

        relative_position_norms = torch.norm(predicted_relative_position, dim=1)
        self.log(
            "Camera_pose/relative_position_norm_mean",
            relative_position_norms.mean(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=self.world_size > 1,
        )
        if relative_position_norms.numel() > 1:
            self.log(
                "Camera_pose/relative_position_norm_std",
                relative_position_norms.std(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=self.world_size > 1,
            )

    def _trim_gt_mesh_data(
        self,
        gt_mesh_data: Dict[str, torch.Tensor],
        selection: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """根据采样到的初始视图索引截取 GT 数据，避免后续函数重复实现。"""
        trimmed = dict(gt_mesh_data)
        if selection is None:
            return trimmed

        selection_device = selection
        for key in ("gt_point_maps", "gt_valid_masks", "depth_z"):
            value = trimmed.get(key)
            if value is None:
                continue
            selection_device = selection.to(value.device)
            trimmed[key] = value.index_select(1, selection_device).contiguous()
        return trimmed

    def _evaluate_candidate_pose(
        self,
        *,
        pose: torch.Tensor,
        initial_images: torch.Tensor,
        camera_poses_batch: torch.Tensor,
        gt_mesh_data: Dict[str, torch.Tensor],
        mesh_batch,
        point_cloud_dir: Optional[str],
        mesh_paths: Optional[Sequence[Optional[str]]] = None,
    ) -> PoseEvaluationResult:
        """统一执行渲染、重建与损失计算，便于策略预测与随机基线共用。"""

        gt_point_maps = gt_mesh_data.get("gt_point_maps")
        gt_valid_masks = gt_mesh_data.get("gt_valid_masks")

        new_render = self.renderer(
            gt_mesh=mesh_batch,
            camera_poses=pose,
            out_depth=True,
            out_points=True,
            out_mask=True,
        )
        new_images = new_render["rgb"]
        new_depth_z = new_render["depth"]
        new_point_maps_render = new_render["points"].permute(0, 2, 3, 1).unsqueeze(1)
        new_valid_masks = new_render["mask"]
        
        # print(new_images, new_depth_z)
        # print(new_point_maps_render,new_valid_masks)
        # tensor[4, 3, 518, 518] n=3219888 (12Mb) x∈[0., 1.276] μ=0.088 σ=0.213 grad SliceBackward0 cuda:0 tensor[4, 1, 518, 518] n=1073296 (4.1Mb) x∈[0., 5.213] μ=0.661 σ=1.383 grad SliceBackward0 cuda:0
        # tensor[4, 1, 518, 518, 3] n=3219888 (12Mb) x∈[-1.595, 1.793] μ=0.025 σ=0.299 grad UnsqueezeBackward0 cuda:0 tensor[4, 1, 518, 518] bool n=1073296 (1.0Mb) x∈[False, True] μ=0.190 σ=0.392 cuda:0

        # 用 depth_z + 可微 pose + 固定内参反投影得到 new world point maps
        # fov_degrees = float(getattr(self.renderer, "default_fov_degrees", 60.0))
        # if self._depth_backproject_xy_signs is None:
        #     self._depth_backproject_xy_signs = infer_depth_backprojection_xy_signs(
        #         depth_z=new_depth_z,
        #         camera_poses=pose.unsqueeze(1).detach(),
        #         reference_world_points=new_point_maps_render,
        #         fov_degrees=fov_degrees,
        #         valid_masks=new_valid_masks,
        #     )
        #     logger.info(
        #         "Inferred depth backprojection xy_signs=%s (fov=%.2f)",
        #         self._depth_backproject_xy_signs,
        #         fov_degrees,
        #     )

        # new_point_maps = camera_depth_z_to_world_points(
        #     depth_z=new_depth_z.detach(),
        #     camera_poses=pose.unsqueeze(1),
        #     fov_degrees=fov_degrees,
        #     valid_masks=new_valid_masks,
        #     xy_signs=self._depth_backproject_xy_signs,
        # )

        # if self.training and self.trainer.is_global_zero:
        #     with torch.no_grad():
        #         diff_l2 = (new_point_maps - new_point_maps_render).norm(dim=-1)  # [B, S, H, W]
        #         if new_valid_masks.any():
        #             diff_valid = diff_l2[new_valid_masks]
        #             self.log(
        #                 "Backprojection/new_view_l2_mean",
        #                 diff_valid.mean(),
        #                 on_step=True,
        #                 on_epoch=False,
        #                 prog_bar=False,
        #                 sync_dist=False,
        #             )
        #             self.log(
        #                 "Backprojection/new_view_l2_max",
        #                 diff_valid.max(),
        #                 on_step=True,
        #                 on_epoch=False,
        #                 prog_bar=False,
        #                 sync_dist=False,
        #             )
        
        try:
            new_point_maps_render.retain_grad()
            self._last_new_point_maps_render = new_point_maps_render
        except RuntimeError:
            self._last_new_point_maps_render = None

        updated_point_maps = torch.cat([gt_point_maps, new_point_maps_render], dim=1).contiguous()
        updated_valid_masks = torch.cat([gt_valid_masks, new_valid_masks], dim=1).contiguous()

        updated_gt_mesh_data = dict(gt_mesh_data)
        updated_gt_mesh_data["gt_point_maps"] = torch.cat(
            [gt_point_maps, new_point_maps_render], dim=1
        ).contiguous()
        updated_gt_mesh_data["gt_valid_masks"] = updated_valid_masks

        depth_z_batch = gt_mesh_data.get("depth_z")
        updated_depth_z = None
        if depth_z_batch is not None:
            depth_z_batch_local = depth_z_batch
            updated_depth_z = torch.cat([depth_z_batch_local, new_depth_z.unsqueeze(-1)], dim=1).contiguous()
            updated_gt_mesh_data["depth_z"] = updated_depth_z

        combined_images_batch = torch.cat([initial_images, new_images.unsqueeze(1)], dim=1)
        combined_camera_poses = torch.cat([camera_poses_batch, pose.unsqueeze(1)], dim=1)

        recon_data = build_recon_from_point_maps(
            point_maps=updated_point_maps,
            camera_poses=combined_camera_poses,
            valid_masks=updated_valid_masks,
            depth_z=updated_depth_z,
        )

        total_loss, loss_components = self.loss_fn(
            recon_data,
            updated_gt_mesh_data,
            combined_images_batch,
            combined_camera_poses,
            return_components=True,
            point_cloud_dir=point_cloud_dir,
        )

        return PoseEvaluationResult(
            total_loss=total_loss,
            loss_components=loss_components,
            new_images=new_images,
            gt_mesh_data=updated_gt_mesh_data,
            depth_z=updated_depth_z,
        )

    def _sample_random_positions(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """在姿态约束内随机采样相机位置，保证 pose_penalty_loss=0。"""
        inner_radius = float(getattr(self.loss_fn, "pose_inner_radius", 1.5))
        outer_radius = float(getattr(self.loss_fn, "pose_outer_radius", inner_radius + 1.0))

        floor_margin = float(getattr(self.loss_fn, "pose_floor_margin", 1.0))
        up_axis = getattr(self.loss_fn, "pose_up_axis", "Y").upper()
        axis_index = {"X": 0, "Y": 1, "Z": 2}.get(up_axis, 1)
        min_height = -floor_margin

        dtype = self.dtype
        positions = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        filled = 0
        attempts = 0
        while filled < batch_size and attempts < 20:
            remaining = batch_size - filled
            sample_count = max(remaining * 2, 4)
            directions = torch.randn(sample_count, 3, device=device, dtype=dtype)
            directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            radii = torch.rand(sample_count, 1, device=device, dtype=dtype)
            radii = radii * (outer_radius - inner_radius) + inner_radius
            samples = directions * radii
            valid_mask = samples[:, axis_index] >= min_height
            valid_samples = samples[valid_mask]
            if valid_samples.numel() == 0:
                attempts += 1
                continue
            take = min(valid_samples.size(0), remaining)
            positions[filled:filled + take] = valid_samples[:take]
            filled += take
            attempts += 1

        if filled < batch_size:
            fallback = torch.randn(batch_size - filled, 3, device=device, dtype=dtype)
            fallback = fallback / fallback.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            radius = (inner_radius + outer_radius) * 0.5
            fallback = fallback * radius
            fallback[:, axis_index] = torch.clamp(fallback[:, axis_index], min=min_height + 1e-4)
            positions[filled:] = fallback

        return positions

    def _compute_random_baseline(
        self,
        *,
        initial_images: torch.Tensor,
        camera_poses_batch: torch.Tensor,
        gt_mesh_data: Dict[str, torch.Tensor],
        mesh_batch,
        mesh_paths: Optional[Sequence[Optional[str]]] = None,
    ) -> Tuple[float, torch.Tensor, float]:
        """生成符合姿态约束的随机位姿并计算其 Chamfer 损失。"""
        device = initial_images.device
        random_positions = self._sample_random_positions(initial_images.shape[0], device=device)
        random_pose = position_to_pose_tensor(random_positions)
        position_norm_mean = random_positions.norm(dim=1).mean().item()

        with torch.no_grad():
            result = self._evaluate_candidate_pose(
                pose=random_pose,
                initial_images=initial_images,
                camera_poses_batch=camera_poses_batch,
                gt_mesh_data=gt_mesh_data,
                mesh_batch=mesh_batch,
                point_cloud_dir=None,
                mesh_paths=mesh_paths,
            )

        return (
            float(result.loss_components.get("chamfer_loss", 0.0)),
            result.new_images,
            position_norm_mean,
        )

    def _select_initial_views(
        self,
        initial_images: torch.Tensor,
        camera_poses: torch.Tensor,
        *,
        depth_z: Optional[torch.Tensor] = None,
        randomize: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, int]:
        """选取训练/验证需要的初始视图子集。"""

        min_views = max(self.min_initial_views, 1)
        max_views = min(self.max_initial_views, initial_images.shape[1])

        total_views = initial_images.shape[1]
        should_randomize = randomize and self.randomize_initial_views

        if should_randomize:
            sampled = torch.randint(
                low=min_views,
                high=max_views + 1,
                size=(1,),
                device=initial_images.device,
            )
            num_views = int(sampled.item())
        else:
            num_views = max_views

        if should_randomize:
            perm = torch.randperm(total_views, device=initial_images.device, dtype=torch.long)
        else:
            perm = torch.arange(total_views, device=initial_images.device, dtype=torch.long)
        selection = perm[:num_views]
        selection, _ = torch.sort(selection)
        initial_images = initial_images.index_select(1, selection)
        camera_poses = camera_poses.index_select(1, selection)
        if depth_z is not None:
            depth_z = depth_z.index_select(1, selection)

        self._last_initial_view_count = num_views
        self._last_initial_view_indices = selection.detach().cpu()

        return initial_images, camera_poses, depth_z, selection, num_views

    def _process_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        单个训练步骤

        Args:
            batch: 训练批次数据
                - initial_images: 初始N个视图 [B, N, 3, H, W]
                - gt_mesh_data: GT mesh数据

        Returns:
            loss_dict: 损失字典
            new_images: 渲染的新视图
            initial_images: 初始视图
        """
        inputs = batch.get("inputs", {})
        targets = batch.get("targets", {})
        mesh_data = batch.get("mesh", {})
        meta = batch.get("meta")
        mesh_paths: Optional[List[Optional[str]]] = None
        if isinstance(meta, list):
            mesh_paths = []
            for entry in meta:
                if isinstance(entry, dict):
                    mesh_paths.append(entry.get("mesh_path"))
                else:
                    mesh_paths.append(None)

        initial_images = inputs["images"]
        camera_poses_batch = inputs["camera_poses"]
        gt_mesh_data = targets["gt_mesh_data"]
        mesh_batch = mesh_data.get("normalized")
        depth_z_batch = gt_mesh_data.get("depth_z")
        initial_images, camera_poses_batch, depth_z_batch, selection, active_view_count = self._select_initial_views(
            initial_images,
            camera_poses_batch,
            depth_z=depth_z_batch,
            randomize=self.trainer.training,
        )
        if mesh_paths is not None and len(mesh_paths) != initial_images.shape[0]:
            logger.warning(
                "mesh_paths length (%d) does not match batch size (%d); disable mesh logging for this batch.",
                len(mesh_paths),
                initial_images.shape[0],
            )
            mesh_paths = None

        trimmed_gt_mesh_data = self._trim_gt_mesh_data(gt_mesh_data, selection)

        scene_features, views = self._extract_scene_features(
            initial_images,
            camera_poses_batch,
            depth_z_batch,
        )

        # print(views[0].keys())
        # dict_keys(['img', 'data_norm_type', 'is_metric_scale', 'depth_along_ray', 'camera_pose_quats', 'camera_pose_trans', 'ray_directions_cam'])
        camera_poses_batch_across_views = self._compute_pose_for_across_views_in_ref_view(views)

        policy_output = self.policy_network(scene_features, camera_poses_batch_across_views)

        if policy_output.shape[-1] < 3:
            raise ValueError(
                f"policy_network 输出维度需至少包含位置 (3)，实际为 {policy_output.shape[-1]}"
            )

        next_camera_pose, predicted_relative_position, _ = self._compute_policy_pose(
            policy_output,
            camera_poses_batch,
        )

        pose_log_step = self.global_step if self.trainer.training else None
        self._log_camera_pose_stats(next_camera_pose, predicted_relative_position, pose_log_step)
        if self.trainer.training:
            try:
                predicted_relative_position.retain_grad()
                next_camera_pose.retain_grad()
                self._last_predicted_relative_position = predicted_relative_position
                self._last_next_camera_pose = next_camera_pose
            except RuntimeError:
                self._last_predicted_relative_position = None
                self._last_next_camera_pose = None

        step_output_dir = None
        if self.trainer.training:
            step_output_dir = os.path.join(
                self.log_dir,
                "images",
                f"step_{self.global_step:06d}",
                f"rank_{self.global_rank:02d}",
            )

        policy_eval = self._evaluate_candidate_pose(
            pose=next_camera_pose,
            initial_images=initial_images,
            camera_poses_batch=camera_poses_batch,
            gt_mesh_data=trimmed_gt_mesh_data,
            mesh_batch=mesh_batch,
            point_cloud_dir=step_output_dir,
        )

        total_loss = policy_eval.total_loss
        loss_components = policy_eval.loss_components
        new_images = policy_eval.new_images
        self._log_new_view_diagnostics(
            new_images=new_images,
            new_depth_z=policy_eval.depth_z,
        )

        if self.trainer.training and step_output_dir is not None:
            self._save_pre_images_grid(
                initial_images=initial_images,
                new_images=new_images,
                step_output_dir=step_output_dir,
            )

        random_chamfer: Optional[float] = None
        random_images = None
        random_position_norm_mean = None
        if self.trainer.training and self.enable_random_baseline:
            random_chamfer, random_images, random_position_norm_mean = self._compute_random_baseline(
                initial_images=initial_images,
                camera_poses_batch=camera_poses_batch,
                gt_mesh_data=trimmed_gt_mesh_data,
                mesh_batch=mesh_batch,
                mesh_paths=mesh_paths,
            )

        logged_loss_keys = (
            "total_loss",
            "chamfer_loss",
            "weighted_chamfer_loss",
            "chamfer_pred_points_mean",
            "chamfer_pred_points_min",
            "chamfer_pred_points_zero_frac",
            "chamfer_pred_points_last_view_mean",
            "chamfer_pred_points_last_view_min",
            "chamfer_pred_points_last_view_zero_frac",
            "confidence_loss",
            "weighted_confidence_loss",
            "viewpoint_loss",
            "weighted_viewpoint_loss",
            "pose_penalty_loss",
            "weighted_pose_penalty_loss",
        )
        loss_dict = {
            key: float(loss_components[key]) for key in logged_loss_keys if key in loss_components
        }
        if random_chamfer is not None:
            loss_dict["random_chamfer_loss"] = float(random_chamfer)
        loss_dict["num_initial_views"] = float(active_view_count)

        if self.trainer.training:
            self._log_training_metrics(loss_dict, active_view_count)
            self._log_random_baseline(
                loss_dict=loss_dict,
                random_chamfer=random_chamfer,
                random_position_norm_mean=random_position_norm_mean,
                random_images=random_images,
                step_output_dir=step_output_dir,
            )

        return total_loss, loss_dict, new_images, initial_images

    def _log_training_metrics(self, loss_dict: Dict[str, float], active_view_count: int) -> None:
        """Record scalar metrics for training."""
        metrics: Dict[str, float] = {
            "train/num_initial_views": float(active_view_count),
        }
        if "chamfer_loss" in loss_dict:
            metrics["train/chamfer_loss"] = loss_dict["chamfer_loss"]
        # for key in (
        #     "chamfer_pred_points_mean",
        #     "chamfer_pred_points_min",
        #     "chamfer_pred_points_zero_frac",
        #     "chamfer_pred_points_last_view_mean",
        #     "chamfer_pred_points_last_view_min",
        #     "chamfer_pred_points_last_view_zero_frac",
        # ):
        #     if key in loss_dict:
        #         metrics[f"train/{key}"] = loss_dict[key]
        if "pose_penalty_loss" in loss_dict:
            metrics["train/pose_penalty_loss"] = loss_dict["pose_penalty_loss"]
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=self.world_size > 1,
        )

    def _log_random_baseline(
        self,
        loss_dict: Dict[str, float],
        random_chamfer: Optional[float],
        random_position_norm_mean: Optional[float],
        random_images: Optional[torch.Tensor],
        step_output_dir: Optional[str],
    ) -> None:
        """Log random baseline diagnostics."""
        if random_chamfer is None:
            return

        self.log(
            "train/random_baseline_chamfer_loss",
            float(random_chamfer),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=self.world_size > 1,
        )
        if random_position_norm_mean is not None:
            self.log(
                "train/random_baseline_position_norm_mean",
                float(random_position_norm_mean),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=self.world_size > 1,
            )
        if not self.trainer.is_global_zero:
            return
        if random_images is None or step_output_dir is None:
            return

        random_image_dir = os.path.join(step_output_dir, "random_baseline")
        os.makedirs(random_image_dir, exist_ok=True)
        save_path = os.path.join(random_image_dir, "random_view.png")
        random_images_cpu = random_images.detach().cpu()
        torchvision.utils.save_image(random_images_cpu, save_path)
        # if random_images_cpu.dim() == 4 and random_images_cpu.size(0) > 0:
        #     first_image = random_images_cpu[0]
        # else:
        #     first_image = random_images_cpu
        # self._log_image("train/random_baseline_new_view", first_image, step)

    def _save_pre_images_grid(
        self,
        *,
        initial_images: torch.Tensor,
        new_images: torch.Tensor,
        step_output_dir: str,
    ) -> None:
        """Save a stitched grid of initial views + NBV view into step_output_dir/pre_images/pre_images.png.

        Layout:
            - Rows: samples in the batch (B)
            - Cols: N initial views, then the generated (N+1)-th NBV view
        """
        if not self.trainer.is_global_zero:
            return
        if step_output_dir is None:
            return
        if initial_images.ndim != 5:
            logger.warning(
                "Skip pre_images grid: initial_images expected [B, N, C, H, W], got %s",
                tuple(initial_images.shape),
            )
            return
        if new_images.ndim != 4:
            logger.warning(
                "Skip pre_images grid: new_images expected [B, C, H, W], got %s",
                tuple(new_images.shape),
            )
            return

        batch_size, num_views, channels, height, width = initial_images.shape
        if new_images.shape[0] != batch_size:
            logger.warning(
                "Skip pre_images grid: batch size mismatch initial_images=%d vs new_images=%d",
                batch_size,
                new_images.shape[0],
            )
            return
        if tuple(new_images.shape[1:]) != (channels, height, width):
            logger.warning(
                "Skip pre_images grid: new_images shape %s does not match expected %s",
                tuple(new_images.shape),
                (batch_size, channels, height, width),
            )
            return

        initial_cpu = initial_images.detach().float().cpu().clamp(0.0, 1.0)
        new_cpu = new_images.detach().float().cpu().clamp(0.0, 1.0)
        images_for_grid = []
        for sample_idx in range(batch_size):
            for view_idx in range(num_views):
                images_for_grid.append(initial_cpu[sample_idx, view_idx])
            images_for_grid.append(new_cpu[sample_idx])

        grid_tensor = torch.stack(images_for_grid, dim=0)
        grid = torchvision.utils.make_grid(
            grid_tensor,
            nrow=num_views + 1,
            padding=2,
        )

        pre_images_dir = os.path.join(step_output_dir, "pre_images")
        os.makedirs(pre_images_dir, exist_ok=True)
        save_path = os.path.join(pre_images_dir, "pre_images.png")
        torchvision.utils.save_image(grid, save_path)

    def _log_new_view_diagnostics(
        self,
        *,
        new_images: torch.Tensor,
        new_depth_z: Optional[torch.Tensor],
    ) -> None:
        """记录新视图渲染质量诊断，帮助定位纯黑/空视角导致的梯度尖峰。"""
        if not self.trainer.training:
            return
        if new_images.numel() == 0:
            return

        with torch.no_grad():
            mean_intensity = new_images.mean()
            min_val = new_images.min()
            max_val = new_images.max()
            gray = new_images.mean(dim=1) if new_images.dim() == 4 else None
            black_frac = None
            if gray is not None:
                black_frac = (gray < 0.05).float().mean()

            metrics = {
                "render/new_view_intensity_mean": mean_intensity,
                "render/new_view_intensity_min": min_val,
                "render/new_view_intensity_max": max_val,
            }
            if black_frac is not None:
                metrics["render/new_view_black_frac"] = black_frac

            if new_depth_z is not None and torch.is_tensor(new_depth_z) and new_depth_z.numel() > 0:
                depth_nonzero = (new_depth_z.abs() > 1e-6).float()
                metrics["render/new_view_valid_frac"] = depth_nonzero.mean()

            self.log_dict(
                metrics,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=self.world_size > 1,
            )

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.policy_network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-7
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        }

    def setup(self, stage=None):
        self.world_size = self.trainer.world_size

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, loss_dict, _, _ = self._process_batch(batch)
        self.log(
            "train/total_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=self.world_size > 1,
        )
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, loss_dict, _, _ = self._process_batch(batch)
        self.log(
            "val/total_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.world_size > 1,
        )
        for key, value in loss_dict.items():
            if key == "total_loss":
                continue
            self.log(
                f"val/{key}",
                float(value),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=self.world_size > 1,
            )
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        """
        在优化器更新参数之前执行。
        用于监控梯度范数，检测梯度消失或爆炸。
        """
        clip_val = None
        clip_algo = "norm"
        if self.trainer is not None:
            clip_val = getattr(self.trainer, "gradient_clip_val", None)
            clip_algo = getattr(self.trainer, "gradient_clip_algorithm", "norm") or "norm"

        parameters = list(self.policy_network.parameters())
        num_params = 0
        for p in parameters:
            if p.grad is not None:
                num_params += p.grad.numel()
        if clip_val is None or float(clip_val) <= 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=float("inf"))
            post_clip_norm = grad_norm
        else:
            clip_val_f = float(clip_val)
            if str(clip_algo).lower() == "value":
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=float("inf"))
                torch.nn.utils.clip_grad_value_(parameters, clip_value=clip_val_f)
                post_clip_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=float("inf"))
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=clip_val_f)
                post_clip_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=float("inf"))

        if num_params > 0:
            grad_rms = grad_norm / math.sqrt(num_params)
        else:
            grad_rms = torch.tensor(0.0, device=grad_norm.device)

        self.log(
            "gradients/global_norm",
            grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "gradients/grad_rms",
            grad_rms,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        if clip_val is not None and float(clip_val) > 0:
            self.log(
                "gradients/global_norm_post_clip",
                post_clip_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

    def on_after_backward(self) -> None:
        """Log gradients w.r.t. pose tensors to pinpoint spike sources."""
        if not self.trainer.training:
            self._last_predicted_relative_position = None
            self._last_next_camera_pose = None
            return

        # relative_position = self._last_predicted_relative_position
        next_camera_pose = self._last_next_camera_pose
        new_point_maps_render = self._last_new_point_maps_render

        # if relative_position is not None and relative_position.grad is not None:
        #     rel_grad_norm = relative_position.grad.norm(dim=-1).mean()
        #     self.log(
        #         "gradients/relative_position_grad_norm",
        #         rel_grad_norm,
        #         on_step=True,
        #         on_epoch=False,
        #         prog_bar=False,
        #         logger=True,
        #     )

        if new_point_maps_render is not None and new_point_maps_render.grad is not None:
            points_grad = new_point_maps_render.grad
            points_grad_norm = points_grad.norm(dim=-1).mean()
            self.log(
                "gradients/new_point_maps_render_grad_norm",
                points_grad_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

        if next_camera_pose is not None and next_camera_pose.grad is not None:
            pose_grad = next_camera_pose.grad
            pos_grad_norm = pose_grad[:, :3].norm(dim=-1).mean()
            quat_grad_norm = pose_grad[:, 3:].norm(dim=-1).mean()
            self.log(
                "gradients/next_pose_position_grad_norm",
                pos_grad_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "gradients/next_pose_quaternion_grad_norm",
                quat_grad_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

        self._last_predicted_relative_position = None
        self._last_next_camera_pose = None

    def transfer_batch_to_device(self, batch: Dict[str, torch.Tensor], device: torch.device, dataloader_idx: int):
        """递归迁移 inputs/targets 到设备，float 张量统一 float32。meta 保持在 CPU。"""

        dtype = self.dtype

        def move_item(x):
            if isinstance(x, torch.Tensor):
                moved = x.to(device)
                return moved.to(dtype=dtype) if moved.is_floating_point() else moved
            if isinstance(x, Meshes):
                return x.to(device)
            return x

        moved: Dict[str, Any] = {}
        data_keys = {"inputs", "targets", "mesh"}
        for key, value in batch.items():
            if key in data_keys:
                moved[key] = apply_to_collection(
                    value,
                    dtype=(torch.Tensor, Meshes),
                    function=move_item,
                )
            else:
                moved[key] = value
        return moved

    def _log_image(self, tag: str, img_tensor: torch.Tensor, step: int) -> None:
        if not self.trainer.is_global_zero:
            return
        if not isinstance(self.logger, WandbLogger):
            return
        try:
            import wandb  # type: ignore
        except ModuleNotFoundError:
            return

        image_cpu = img_tensor.detach().float().cpu()
        if image_cpu.ndim == 3 and image_cpu.shape[0] in (1, 3):
            image_cpu = image_cpu.permute(1, 2, 0).contiguous()
        elif image_cpu.ndim != 2 and image_cpu.ndim != 3:
            return

        run = self.logger.experiment
        current_step = getattr(run, "step", None)
        if current_step is None:
            run.log({tag: wandb.Image(image_cpu.numpy())})
            return

        safe_step = max(int(step), int(current_step))
        run.log({tag: wandb.Image(image_cpu.numpy())}, step=safe_step)
