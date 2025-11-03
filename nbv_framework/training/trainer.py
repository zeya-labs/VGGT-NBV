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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, NamedTuple
import numpy as np
import os
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision

if TYPE_CHECKING:
    from ..models import MapAnythingWrapper, BaseNBVPolicy
from ..rendering import DifferentiableRenderer
from .loss import ReconstructionLoss, ChamferDistance
from ..utils.camera_utils import (
    position_to_pose_tensor,
    world_points_to_camera_depth,
)
from ..utils.render_utils import render_gt_point_maps


class PoseEvaluationResult(NamedTuple):
    total_loss: torch.Tensor
    loss_components: Dict[str, float]
    new_images: torch.Tensor
    gt_mesh_data: Dict[str, torch.Tensor]
    depth_z: Optional[torch.Tensor]


class NBVTrainer:
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
                 num_epochs: int = 1000,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 log_dir: str = "runs/nbv_experiment",
                 device: str = "cuda",
                 enable_validation: bool = False,
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
            log_dir: TensorBoard日志目录
            device: 计算设备
            enable_validation: 是否在训练过程中执行验证流程
            enable_random_baseline: 是否计算随机基线视角的 Chamfer 统计
        """
        self.vggt_wrapper = vggt_wrapper
        self.policy_network = policy_network
        self.renderer = renderer
        self.loss_fn = loss_fn
        self.device = device
        self.num_epochs = num_epochs
        self.log_dir = log_dir
        self.enable_validation = enable_validation
        self.use_epoch_seed = use_epoch_seed
        self.enable_random_baseline = bool(enable_random_baseline)

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

        # 初始化TensorBoard Writer
        self.writer = SummaryWriter(self.log_dir)
        
        # 优化器（只优化策略网络）
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs, eta_min=1e-7
        )
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.val_image_step = 0
        
        # 日志
        self.setup_logging()

    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('nbv_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _configure_policy_mode(self, backprop: bool) -> None:
        """根据是否反向传播设置策略网络模式并清空梯度。"""
        if backprop:
            self.policy_network.train()
            self.optimizer.zero_grad()
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
        if step_index is None or not hasattr(self, "writer"):
            return

        positions = next_camera_pose[:, :3]
        quaternions = next_camera_pose[:, 3:]

        position_norms = torch.norm(positions, dim=1)
        self.writer.add_scalar('camera_pose/position_norm_mean', position_norms.mean(), step_index)
        if position_norms.numel() > 1:
            self.writer.add_scalar('camera_pose/position_norm_std', position_norms.std(), step_index)

        quaternion_norms = torch.norm(quaternions, dim=1)
        if quaternion_norms.numel() > 1:
            self.writer.add_scalar('camera_pose/quaternion_norm_std', quaternion_norms.std(), step_index)

        relative_position_norms = torch.norm(predicted_relative_position, dim=1)
        self.writer.add_scalar('camera_pose/relative_position_norm_mean', relative_position_norms.mean(), step_index)
        if relative_position_norms.numel() > 1:
            self.writer.add_scalar('camera_pose/relative_position_norm_std', relative_position_norms.std(), step_index)

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

        tb_writer = self.writer if log_to_tensorboard and hasattr(self, "writer") else None
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
    ) -> float:
        """生成符合姿态约束的随机位姿并计算其 Chamfer 损失。"""
        device = initial_images.device
        random_positions = self._sample_random_positions(initial_images.shape[0], device=device)
        random_pose = position_to_pose_tensor(random_positions)

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
            )

        return float(result.loss_components.get("chamfer_loss", 0.0))

    def _log_vggt_gradient_stats(self, new_images: torch.Tensor) -> None:
        """记录VGGT相关的梯度统计信息到TensorBoard。"""
        grad_stats = self.vggt_wrapper.collect_gradient_stats()

        for key in getattr(self, "_vggt_grad_keys", ("depth", "world_points_from_depth")):
            norm_key = f"{key}/grad_norm"
            mean_key = f"{key}/grad_mean_abs"

            norm_val = grad_stats.get(norm_key, 0.0)
            mean_val = grad_stats.get(mean_key, 0.0)
            has_grad = 1.0 if norm_key in grad_stats else 0.0

            self.writer.add_scalar(f'train/gradients/vggt/{key}_grad_norm', norm_val, self.global_step) if has_grad else None
            self.writer.add_scalar(f'train/gradients/vggt/{key}_grad_mean_abs', mean_val, self.global_step) if has_grad else None
            self.writer.add_scalar(f'train/gradients/vggt/{key}_has_grad', has_grad, self.global_step) if has_grad else None

        has_new_grad = 1.0 if new_images.grad is not None else 0.0
        self.writer.add_scalar('train/gradients/new_view_has_grad', has_new_grad, self.global_step)

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
    
    def training_step(self, 
                     batch: Dict[str, torch.Tensor],
                     backprop: bool = True) -> Tuple[Dict[str, float], Optional[torch.Tensor], Optional[torch.Tensor]]:
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
        if backprop:
            step_output_dir = os.path.join(self.log_dir, "images", f"step_{self.global_step:06d}")

        policy_eval = self._evaluate_candidate_pose(
            pose=next_camera_pose,
            initial_images=initial_images,
            camera_poses_batch=camera_poses_batch,
            gt_mesh_data=trimmed_gt_mesh_data,
            render_step=render_step,
            train_flag=backprop,
            point_cloud_dir=step_output_dir,
            log_to_tensorboard=backprop
        )

        total_loss = policy_eval.total_loss
        loss_components = policy_eval.loss_components
        new_images = policy_eval.new_images

        random_chamfer = None
        if backprop and self.enable_random_baseline:
            random_chamfer = self._compute_random_baseline(
                initial_images=initial_images,
                camera_poses_batch=camera_poses_batch,
                gt_mesh_data=trimmed_gt_mesh_data,
                render_step=render_step,
            )

        if backprop:
            total_loss.backward()

            pose_grad = next_camera_pose.grad
            pose_grad_norm = pose_grad.norm().detach().item() if pose_grad is not None else 0.0
            pose_grad_mean = pose_grad.abs().mean().detach().item() if pose_grad is not None else 0.0

            if pose_grad is None:
                self.logger.warning("next_camera_pose grad is None")

            self.writer.add_scalar('next_camera_pose_grad/norm', pose_grad_norm, self.global_step)
            self.writer.add_scalar('next_camera_pose_grad/mean_abs', pose_grad_mean, self.global_step)
            self.writer.add_scalar('next_camera_pose_grad/has_grad', 1.0 if pose_grad is not None else 0.0, self.global_step)

            self._log_vggt_gradient_stats(new_images)

            total_policy_norm = torch.nn.utils.clip_grad_norm_(
                self.policy_network.parameters(), max_norm=1.0
            )
            self.writer.add_scalar('policy_net_gradients/total_norm', total_policy_norm.item(), self.global_step)
            if total_policy_norm.item() > 1000.0:
                print(f"🚨 检测到梯度爆炸！当前全局step: {self.global_step}, 范数: {total_policy_norm.item()}")
                problematic_pose = next_camera_pose.detach().cpu().numpy()
                print(f"   引发问题的位姿: {problematic_pose}")

            self.optimizer.step()
        
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
        loss_dict = {key: loss_components[key] for key in logged_loss_keys if key in loss_components}
        if random_chamfer is not None:
            loss_dict["random_chamfer_loss"] = random_chamfer
        loss_dict["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        loss_dict["num_initial_views"] = float(active_view_count)

        if backprop:
            self.writer.add_scalar('train/total_loss', loss_dict['total_loss'], self.global_step)
            self.writer.add_scalar('train/learning_rate', loss_dict['learning_rate'], self.global_step)
            self.writer.add_scalar('train/num_initial_views', active_view_count, self.global_step)

            if random_chamfer is not None and 'chamfer_loss' in loss_dict:
                self.writer.add_scalars(
                    'train/losses/chamfer_loss',
                    {'policy': loss_dict['chamfer_loss'], 'random': random_chamfer},
                    self.global_step,
                )
                self.writer.add_scalar('train/losses/random_chamfer_loss', random_chamfer, self.global_step)
            elif 'chamfer_loss' in loss_dict:
                self.writer.add_scalar('train/losses/chamfer_loss', loss_dict['chamfer_loss'], self.global_step)

            if 'pose_penalty_loss' in loss_dict:
                self.writer.add_scalar('train/losses/pose_penalty_loss', loss_dict['pose_penalty_loss'], self.global_step)

            self.global_step += 1
        
        return loss_dict, new_images, initial_images
    
    def validation_step(self, 
                       batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """验证步骤"""
        self.policy_network.eval()
        with torch.no_grad():
            loss_dict, new_images, initial_images = self.training_step(batch, backprop=False)
        
        return loss_dict, new_images, initial_images
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        epoch_losses = []

        if self.use_epoch_seed and hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(self.current_epoch)

        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in progress_bar:
            # 将数据移到设备
            batch = self._move_batch_to_device(batch)
            
            # 训练步骤
            loss_dict, new_images, initial_images = self.training_step(batch)
            if new_images is not None and initial_images is not None:
                # 将 initial_images 的 B, N, C, H, W 格式转换为 B*N, C, H, W
                b, n, c, h, w = initial_images.shape
                initial_images_flat = initial_images.view(b * n, c, h, w)
                
                initial_grid = torchvision.utils.make_grid(initial_images_flat, nrow=n)
                # print(self.global_step)
                self.writer.add_image('train/initial_views', initial_grid, self.global_step)

                new_grid = torchvision.utils.make_grid(new_images, nrow=1)
                self.writer.add_image('train/next_best_view', new_grid, self.global_step)
            epoch_losses.append(loss_dict)
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{loss_dict['total_loss']:.4f}",
                "lr": f"{loss_dict['learning_rate']:.2e}",
                "views": int(loss_dict.get("num_initial_views", 0)),
            })
        
        # 计算epoch平均损失
        avg_loss_dict = self._average_loss_dicts(epoch_losses)
        
        return avg_loss_dict
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        epoch_losses = []

        if hasattr(val_loader, "dataset") and hasattr(val_loader.dataset, "set_epoch"):
            val_loader.dataset.set_epoch(0)

        progress_bar = tqdm(val_loader, desc="Validation")

        for i, batch in enumerate(progress_bar):
            batch = self._move_batch_to_device(batch)
            loss_dict, new_images, initial_images = self.validation_step(batch)
            epoch_losses.append(loss_dict)

            # 在第一个batch上记录图像
            if i == 0 and new_images is not None and initial_images is not None:
                # 将 initial_images 的 B, N, C, H, W 格式转换为 B*N, C, H, W
                b, n, c, h, w = initial_images.shape
                initial_images_flat = initial_images.view(b * n, c, h, w)
                
                initial_grid = torchvision.utils.make_grid(initial_images_flat, nrow=n)
                self.writer.add_image('val/initial_views', initial_grid, self.val_image_step)

                new_grid = torchvision.utils.make_grid(new_images, nrow=1)
                self.writer.add_image('val/next_best_view', new_grid, self.val_image_step)
                self.val_image_step += 1

            progress_bar.set_postfix({
                "val_loss": f"{loss_dict['total_loss']:.4f}",
                "views": int(loss_dict.get("num_initial_views", 0)),
            })

        avg_loss_dict = self._average_loss_dicts(epoch_losses)

        # 记录验证损失（以 epoch 作为 step）
        self.writer.add_scalar('val/total_loss', avg_loss_dict['total_loss'], self.current_epoch)
        if 'num_initial_views' in avg_loss_dict:
            self.writer.add_scalar('val/num_initial_views', avg_loss_dict['num_initial_views'], self.current_epoch)
        
        # 记录各个验证损失组件（原始值）
        self.writer.add_scalar('val/losses/chamfer_loss', avg_loss_dict['chamfer_loss'], self.current_epoch)
        self.writer.add_scalar('val/losses/confidence_loss', avg_loss_dict['confidence_loss'], self.current_epoch)
        self.writer.add_scalar('val/losses/viewpoint_loss', avg_loss_dict['viewpoint_loss'], self.current_epoch)
        
        # 记录加权后的验证损失组件
        self.writer.add_scalar('val/weighted_losses/chamfer_loss', avg_loss_dict['weighted_chamfer_loss'], self.current_epoch)
        self.writer.add_scalar('val/weighted_losses/confidence_loss', avg_loss_dict['weighted_confidence_loss'], self.current_epoch)
        self.writer.add_scalar('val/weighted_losses/viewpoint_loss', avg_loss_dict['weighted_viewpoint_loss'], self.current_epoch)

        return avg_loss_dict
    
    def train(self, 
             train_loader: DataLoader,
             val_loader: Optional[DataLoader] = None,
             save_dir: str = "checkpoints"):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            save_dir: 模型保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        self.logger.info(f"Policy network parameters: {sum(p.numel() for p in self.policy_network.parameters())}")

        if self.enable_validation:
            if val_loader is None:
                self.logger.warning("Validation is enabled but no val_loader is provided; validation will be skipped.")
        elif val_loader is not None:
            self.logger.info("Validation loader supplied but validation is disabled; it will be ignored.")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss_dict = self.train_epoch(train_loader)
            
            # 验证
            val_loss_dict = None
            if self.enable_validation and val_loader is not None:
                val_loss_dict = self.validate_epoch(val_loader)

            # 学习率调度
            if self.global_step == 0:
                self.logger.warning(
                    "Skipping lr_scheduler.step() at epoch %d because optimizer.step() has not been called yet.",
                    self.current_epoch,
                )
            else:
                self.scheduler.step()
            
            # 日志记录
            self._log_epoch_results(train_loss_dict, val_loss_dict)

            # 模型保存
            if val_loss_dict is not None:
                val_loss = val_loss_dict["total_loss"]
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self._save_checkpoint(save_dir, "best_model.pth")
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
        
        self.writer.close()
        self.logger.info("Training completed!")
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将batch数据递归移到训练设备，并将浮点张量统一为float32（支持嵌套的dict/list/tuple结构）"""
        def _to_device(x):
            if isinstance(x, torch.Tensor):
                tensor = x.to(self.device)
                if tensor.is_floating_point() and tensor.dtype != torch.float32:
                    tensor = tensor.to(dtype=torch.float32)
                return tensor
            if isinstance(x, dict):
                return {k: _to_device(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_to_device(v) for v in x]
            if isinstance(x, tuple):
                return tuple(_to_device(v) for v in x)
            return x
        return _to_device(batch)
    
    def _average_loss_dicts(self, loss_dicts: List[Dict[str, float]]) -> Dict[str, float]:
        """计算损失字典的平均值"""
        if not loss_dicts:
            return {}
        
        avg_dict = {}
        for key in loss_dicts[0].keys():
            avg_dict[key] = np.mean([d[key] for d in loss_dicts])
        
        return avg_dict
    
    def _log_epoch_results(self, 
                          train_loss_dict: Dict[str, float],
                          val_loss_dict: Optional[Dict[str, float]]):
        """记录epoch结果"""
        train_loss = train_loss_dict["total_loss"]
        log_msg = f"Epoch {self.current_epoch}: train_loss={train_loss:.4f}"
        
        if val_loss_dict is not None:
            val_loss = val_loss_dict["total_loss"]
            log_msg += f", val_loss={val_loss:.4f}"
        
        self.logger.info(log_msg)
    
    def _save_checkpoint(self, save_dir: str, filename: str):
        """保存检查点"""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "policy_network_state_dict": self.policy_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss
        }
        
        save_path = os.path.join(save_dir, filename)
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
