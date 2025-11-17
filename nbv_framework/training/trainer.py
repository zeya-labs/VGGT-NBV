"""
NBV策略训练器

实现端到端的目标驱动策略学习训练流程：
1. 状态编码：MapAnything 提取场景特征
2. 动作提议：策略网络输出相机位姿
3. 环境交互：可微分渲染生成新视图
4. 质量评估：VGGT重建并计算质量损失
5. 策略更新：反向传播更新策略网络
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, NamedTuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_fabric.utilities.apply_func import apply_to_collection

if TYPE_CHECKING:
    from ..models import MapAnythingWrapper, BaseNBVPolicy
from ..rendering import DifferentiableRenderer
from .loss import ReconstructionLoss, ChamferDistance
from ..utils.camera_utils import (
    position_to_pose_tensor,
    world_points_to_camera_depth,
)
from ..utils.render_utils import render_gt_point_maps


LOGGER = logging.getLogger(__name__)


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
                 device: str = "cuda",
                 use_epoch_seed: bool = False,
                 enable_random_baseline: bool = True,
                 distributed: bool = False,
                 world_size: int = 1,
                 rank: int = 0):
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
            log_dir: TensorBoard日志目录
            device: 计算设备
            enable_random_baseline: 是否计算随机基线视角的 Chamfer 统计
            distributed: 是否启用分布式训练
            world_size: 全局进程数
            rank: 当前进程的全局rank
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
        self.enable_random_baseline = bool(enable_random_baseline)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.is_main_process = True
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.min_initial_views = min_initial_views
        self.max_initial_views = max_initial_views
        self.randomize_initial_views = bool(randomize_initial_views)
        self._last_initial_view_count = 0
        self._last_initial_view_indices: Optional[torch.Tensor] = None

        # 启用VGGT梯度捕获，便于调试NBV梯度链路
        self._vggt_grad_keys = ("world_points", "world_points_conf")
        self.vggt_wrapper.configure_gradient_capture(
            enable=True,
            keys=self._vggt_grad_keys,
            capture_input=False
        )

        # 初始化TensorBoard Writer（仅主进程写入）
        self.val_image_step = 0
        self.tb_writer = None

    def _configure_policy_mode(self, backprop: bool) -> None:
        """根据是否反向传播设置策略网络模式。"""
        if backprop:
            self.policy_network.train()
        else:
            self.policy_network.eval()

    def _extract_scene_features(
        self,
        initial_images: torch.Tensor,
        camera_poses_batch: torch.Tensor,
        depth_z_batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """封装 MapAnything scene feature 提取，便于复用和测试。"""
        return self.vggt_wrapper.extract_scene_features(
            initial_images,
            camera_poses_batch,
            is_metric_scale=False,
            depth_z=depth_z_batch,
        )

    def _compute_policy_pose(
        self,
        policy_output: torch.Tensor,
        camera_poses_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据策略输出计算绝对位姿及相关中间量。"""
        reference_position = camera_poses_batch[:, 0, :3]
        predicted_relative_position = policy_output[:, :3]
        absolute_position = reference_position + predicted_relative_position
        next_camera_pose = position_to_pose_tensor(absolute_position)
        return next_camera_pose, predicted_relative_position, absolute_position

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
        self._add_scalar('Camera_pose/position_norm_mean', position_norms.mean(), step_index)
        if position_norms.numel() > 1:
            self._add_scalar('Camera_pose/position_norm_std', position_norms.std(), step_index)

        quaternion_norms = torch.norm(quaternions, dim=1)
        if quaternion_norms.numel() > 1:
            self._add_scalar('Camera_pose/quaternion_norm_std', quaternion_norms.std(), step_index)

        relative_position_norms = torch.norm(predicted_relative_position, dim=1)
        self._add_scalar('Camera_pose/relative_position_norm_mean', relative_position_norms.mean(), step_index)
        if relative_position_norms.numel() > 1:
            self._add_scalar('Camera_pose/relative_position_norm_std', relative_position_norms.std(), step_index)

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
        render_step: Optional[int],
        train_flag: bool,
        point_cloud_dir: Optional[str],
        log_to_tensorboard: bool,
        retain_gradients: bool,
    ) -> PoseEvaluationResult:
        """统一执行渲染、重建与损失计算，便于策略预测与随机基线共用。"""
        batched_mesh = gt_mesh_data['normalized_mesh']
        batched_mesh = batched_mesh.to(self.renderer.device)

        new_images = self.renderer(
            gt_mesh=batched_mesh,
            camera_poses=pose,
        )
        if new_images.device != initial_images.device:
            new_images = new_images.to(initial_images.device)
        if retain_gradients:
            if new_images.requires_grad:
                new_images.retain_grad()
        else:
            new_images = new_images.detach()

        gt_point_maps = gt_mesh_data.get("gt_point_maps")
        gt_valid_masks = gt_mesh_data.get("gt_valid_masks")
        if gt_point_maps is None or gt_valid_masks is None:
            raise KeyError(
                "gt_mesh_data must contain 'gt_point_maps' and 'gt_valid_masks' for pose evaluation."
            )

        gt_point_maps = gt_point_maps.to(device=initial_images.device, dtype=torch.float32)
        gt_valid_masks = gt_valid_masks.to(device=initial_images.device, dtype=torch.bool)

        if camera_poses_batch.dim() == 2:
            base_camera_poses = camera_poses_batch.unsqueeze(1)
        else:
            base_camera_poses = camera_poses_batch

        tb_writer = self.tb_writer if log_to_tensorboard and hasattr(self, "tb_writer") else None
        step_arg = render_step if log_to_tensorboard else None

        new_point_maps, new_valid_masks = render_gt_point_maps(
            renderer=self.renderer,
            mesh_batch=batched_mesh,
            camera_poses=pose,
            output_device=initial_images.device,
            writer=tb_writer,
            step=step_arg,
            train_flag=train_flag,
        )

        updated_point_maps = torch.cat([gt_point_maps, new_point_maps], dim=1).contiguous()
        updated_valid_masks = torch.cat([gt_valid_masks, new_valid_masks], dim=1).contiguous().to(dtype=torch.bool)

        updated_gt_mesh_data = dict(gt_mesh_data)
        updated_gt_mesh_data["gt_point_maps"] = updated_point_maps
        updated_gt_mesh_data["gt_valid_masks"] = updated_valid_masks

        depth_z_batch = gt_mesh_data.get("depth_z")
        updated_depth_z = None
        if depth_z_batch is not None:
            depth_z_batch_local = depth_z_batch.to(initial_images.device, dtype=torch.float32)
            new_depth_z = world_points_to_camera_depth(
                new_point_maps,
                pose.unsqueeze(1),
                valid_masks=new_valid_masks,
                writer=tb_writer,
                step=step_arg,
                log_prefix="DepthZ/NewView",
                train_flag=train_flag,
            ).detach()
            if depth_z_batch_local.device != new_depth_z.device:
                depth_z_batch_local = depth_z_batch_local.to(new_depth_z.device)
            if depth_z_batch_local.dtype != new_depth_z.dtype:
                depth_z_batch_local = depth_z_batch_local.to(new_depth_z.dtype)
            updated_depth_z = torch.cat([depth_z_batch_local, new_depth_z], dim=1).contiguous()
            updated_gt_mesh_data["depth_z"] = updated_depth_z

        combined_images_batch = torch.cat([initial_images, new_images.unsqueeze(1)], dim=1)
        combined_camera_poses = torch.cat([base_camera_poses, pose.unsqueeze(1)], dim=1)

        recon_data = self.vggt_wrapper.reconstruct_and_evaluate(
            combined_images_batch,
            combined_camera_poses,
            depth_z=updated_depth_z,
            is_metric_scale=False,
            view_save_dir=point_cloud_dir,
        )

        total_loss, loss_components = self.loss_fn(
            recon_data,
            updated_gt_mesh_data,
            combined_images_batch,
            combined_camera_poses,
            return_components=True,
            writer=tb_writer,
            step=step_arg,
            train_flag=train_flag,
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

        positions = torch.zeros(batch_size, 3, device=device, dtype=torch.float32)
        filled = 0
        attempts = 0
        while filled < batch_size and attempts < 20:
            remaining = batch_size - filled
            sample_count = max(remaining * 2, 4)
            directions = torch.randn(sample_count, 3, device=device, dtype=torch.float32)
            directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            radii = torch.rand(sample_count, 1, device=device, dtype=torch.float32)
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
            fallback = torch.randn(batch_size - filled, 3, device=device, dtype=torch.float32)
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
        render_step: Optional[int],
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
                render_step=render_step,
                train_flag=False,
                point_cloud_dir=None,
                log_to_tensorboard=False,
                retain_gradients=False,
            )

        return (
            float(result.loss_components.get("chamfer_loss", 0.0)),
            result.new_images.detach(),
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
        backprop: bool = True,
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
        self._configure_policy_mode(backprop)

        initial_images = batch["initial_images"]
        gt_mesh_data = batch["gt_mesh_data"]
        camera_poses_batch = batch["camera_poses"]
        depth_z_batch = gt_mesh_data.get("depth_z")
        # print("keys",batch.keys())
        # gt_mesh_data_keys dict_keys(['mesh_path', 'gt_points', 'normalize_method', 'num_samples', 'gt_point_maps', 'gt_valid_masks', 'depth_z', 'depth_z_viz', 'original_mesh', 'normalized_mesh'])
        # batch_keys dict_keys(['initial_images', 'camera_poses', 'mesh_path', 'batch_name', 'set_name', 'model_name', 'depth_z', 'depth_z_viz', 'source_dataset', 'source_dataset_idx', 'source_dataset_sample_idx', 'gt_mesh_data'])
        initial_images, camera_poses_batch, depth_z_batch, selection, active_view_count = self._select_initial_views(
            initial_images,
            camera_poses_batch,
            depth_z=depth_z_batch,
            randomize=backprop,
        )

        trimmed_gt_mesh_data = self._trim_gt_mesh_data(gt_mesh_data, selection)

        scene_features = self._extract_scene_features(
            initial_images,
            camera_poses_batch,
            depth_z_batch,
        )

        policy_output = self.policy_network(scene_features)

        if policy_output.shape[-1] < 3:
            raise ValueError(
                f"policy_network 输出维度需至少包含位置 (3)，实际为 {policy_output.shape[-1]}"
            )

        next_camera_pose, predicted_relative_position, _ = self._compute_policy_pose(
            policy_output,
            camera_poses_batch,
        )

        if backprop and next_camera_pose.requires_grad:
            next_camera_pose.retain_grad()

        pose_log_step = self.global_step if backprop else None
        self._log_camera_pose_stats(next_camera_pose, predicted_relative_position, pose_log_step)

        render_step = self.global_step if backprop else getattr(self, "val_image_step", None)
        step_output_dir = None
        if backprop and self.is_main_process:
            step_output_dir = os.path.join(
                self.log_dir,
                "images",
                f"step_{self.global_step:06d}",
                f"rank_{self.rank:02d}",
            )

        policy_eval = self._evaluate_candidate_pose(
            pose=next_camera_pose,
            initial_images=initial_images,
            camera_poses_batch=camera_poses_batch,
            gt_mesh_data=trimmed_gt_mesh_data,
            render_step=render_step,
            train_flag=backprop,
            point_cloud_dir=step_output_dir,
            log_to_tensorboard=backprop,
            retain_gradients=False,
        )

        total_loss = policy_eval.total_loss
        loss_components = policy_eval.loss_components
        new_images = policy_eval.new_images

        random_chamfer: Optional[float] = None
        random_images = None
        random_position_norm_mean = None
        if backprop and self.enable_random_baseline:
            random_chamfer, random_images, random_position_norm_mean = self._compute_random_baseline(
                initial_images=initial_images,
                camera_poses_batch=camera_poses_batch,
                gt_mesh_data=trimmed_gt_mesh_data,
                render_step=render_step,
            )

        logged_loss_keys = (
            "total_loss",
            "chamfer_loss",
            "weighted_chamfer_loss",
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

        if backprop:
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
        step = self.global_step
        self._add_scalar("Train_losses/total_loss", loss_dict["total_loss"], step)
        self._add_scalar("Train/num_initial_views", active_view_count, step)

        if "chamfer_loss" in loss_dict:
            self._add_scalar("Train_losses/chamfer_loss", loss_dict["chamfer_loss"], step)
        if "pose_penalty_loss" in loss_dict:
            self._add_scalar("Train_losses/pose_penalty_loss", loss_dict["pose_penalty_loss"], step)

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

        step = self.global_step
        if "chamfer_loss" in loss_dict:
            self._add_scalars(
                "Train_losses/chamfer_loss",
                {"policy": loss_dict["chamfer_loss"], "random": random_chamfer},
                step,
            )
        self._add_scalar("Random_baseline/random_chamfer_loss", random_chamfer, step)
        if random_position_norm_mean is not None:
            self._add_scalar("Random_baseline/position_norm_mean", random_position_norm_mean, step)
        if not self.is_main_process:
            return
        if random_images is None or step_output_dir is None:
            return

        random_image_dir = os.path.join(step_output_dir, "random_baseline")
        os.makedirs(random_image_dir, exist_ok=True)
        save_path = os.path.join(random_image_dir, "random_view.png")
        random_images_cpu = random_images.detach().cpu()
        torchvision.utils.save_image(random_images_cpu, save_path)
        if random_images_cpu.dim() == 4 and random_images_cpu.size(0) > 0:
            first_image = random_images_cpu[0]
        else:
            first_image = random_images_cpu
        self._add_image("Random_baseline/new_view", first_image, step)
    
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
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }

    def on_fit_start(self) -> None:
        if self.trainer is not None:
            self.rank = getattr(self.trainer, "global_rank", 0)
            world_size = getattr(self.trainer, "world_size", None)
            if world_size is None:
                world_size = max(1, getattr(self.trainer, "num_devices", 1))
            self.world_size = max(1, int(world_size))
            self.is_main_process = bool(getattr(self.trainer, "is_global_zero", True))

        if self.is_main_process and isinstance(self.logger, TensorBoardLogger):
            self.tb_writer = self.logger.experiment

        device = torch.device(self.device)
        self.vggt_wrapper.to(device)
        self.renderer.to(device)
        self.loss_fn.to(device)
        if hasattr(self.vggt_wrapper, "device"):
            self.vggt_wrapper.device = device
        if hasattr(self.renderer, "device"):
            self.renderer.device = device

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, loss_dict, _, _ = self._process_batch(batch, backprop=True)
        for key, value in loss_dict.items():
            prog = key == "total_loss"
            self.log(f"train/{key}", value, on_step=True, on_epoch=False, prog_bar=prog, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, loss_dict, _, _ = self._process_batch(batch, backprop=False)
        for key, value in loss_dict.items():
            prog = key == "total_loss"
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=prog, sync_dist=True)
        return loss

    def transfer_batch_to_device(self, batch: Dict[str, torch.Tensor], device: torch.device, dataloader_idx: int):
        """将batch数据递归移到训练设备，并将浮点张量统一为float32（支持嵌套的dict/list/tuple结构）"""
        # 定义一个处理函数
        def fn(t):
            return t.to(device).to(dtype=torch.float32) if t.is_floating_point() else t.to(device)

        return apply_to_collection(batch, dtype=torch.Tensor, function=fn)
    
    def _add_scalar(self, tag: str, value: float, step: int) -> None:
        if not self.is_main_process:
            return
        writer = getattr(self, "tb_writer", None)
        if writer is None:
            return
        writer.add_scalar(tag, value, step)

    def _add_scalars(self, tag: str, scalar_dict: Dict[str, float], step: int) -> None:
        if not self.is_main_process:
            return
        writer = getattr(self, "tb_writer", None)
        if writer is None:
            return
        writer.add_scalars(tag, scalar_dict, step)

    def _add_image(self, tag: str, img_tensor: torch.Tensor, step: int) -> None:
        if not self.is_main_process:
            return
        writer = getattr(self, "tb_writer", None)
        if writer is None:
            return
        writer.add_image(tag, img_tensor, step)
    
