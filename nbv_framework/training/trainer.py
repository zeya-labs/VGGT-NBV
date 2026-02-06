"""
NBV策略训练器
实现端到端的目标驱动策略学习训练流程
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch
import torch.optim as optim
from lightning.pytorch import LightningModule
from lightning_fabric.utilities.apply_func import apply_to_collection
from pytorch3d.structures import Meshes

if TYPE_CHECKING:
    from ..models import MapAnythingWrapper, BaseNBVPolicy
from ..rendering import DifferentiableRenderer
from .logging import log_step_outputs, resolve_step_output_dir
from .loss import ReconstructionLoss
from ..pipeline.types import (
    PoseEvaluationResult,
    PolicyInferenceOutput,
    PreparedBatch,
    RandomBaselineOutput,
)
from ..utils.mesh_utils import load_meshes_as_batch
from ..data.batch_utils import parse_mesh_metadata, trim_gt_mesh_data
from ..cache.render_cache import RenderCache
from ..geometry.pose_ops import (
    compute_pose_for_across_views_in_ref_view,
    compute_policy_pose,
    compute_pose_scale_factor,
)
from ..pipeline.step_ops import (
    compute_random_baseline,
    evaluate_candidate_pose,
    render_inputs,
    select_initial_views,
)


logger = logging.getLogger(__name__)


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
                 enable_random_baseline: bool = True,
                 mesh_load_workers: int = 4,
                 render_cache_enabled: bool = True,
                 render_cache_root: Optional[str] = None):
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
        self.mesh_load_workers = mesh_load_workers
        self.render_cache_enabled = bool(render_cache_enabled)

        self.min_initial_views = min_initial_views
        self.max_initial_views = max_initial_views
        self.randomize_initial_views = randomize_initial_views
        self._last_initial_view_count = 0
        self._last_initial_view_indices: Optional[torch.Tensor] = None
        self._last_batch_size: Optional[int] = None

        # Grad stats captured via hooks (avoid keeping autograd graph alive).
        self._last_predicted_relative_position_grad_norm: Optional[torch.Tensor] = None
        self._last_next_pose_position_grad_norm: Optional[torch.Tensor] = None
        self._last_next_pose_quaternion_grad_norm: Optional[torch.Tensor] = None
        self._last_new_point_maps_grad_norm: Optional[torch.Tensor] = None

        # 深度反投影坐标轴符号约定（与渲染器一致）
        self._depth_backproject_xy_signs: Optional[Tuple[int, int]] = (-1,-1)

        if self.render_cache_enabled:
            root = Path(render_cache_root) if render_cache_root else None
            self.render_cache = RenderCache(renderer=self.renderer, root=root)
        else:
            self.render_cache = None
        
        self.log_freq = 5

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
        return compute_pose_for_across_views_in_ref_view(views)

    def _compute_policy_pose(
        self,
        policy_output: torch.Tensor,
        camera_poses_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return compute_policy_pose(policy_output, camera_poses_batch)

    def _compute_pose_scale_factor(self, camera_poses_batch: torch.Tensor) -> torch.Tensor:
        return compute_pose_scale_factor(camera_poses_batch)

    def _render_inputs(
        self,
        initial_images: Optional[torch.Tensor],
        camera_poses_batch: torch.Tensor,
        gt_mesh_data: Dict[str, torch.Tensor],
        mesh_batch: Optional[Meshes],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return render_inputs(
            renderer=self.renderer,
            initial_images=initial_images,
            camera_poses_batch=camera_poses_batch,
            gt_mesh_data=gt_mesh_data,
            mesh_batch=mesh_batch,
            dtype=self.dtype,
        )

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> PreparedBatch:
        inputs = batch.get("inputs", {})
        targets = batch.get("targets", {})
        mesh_data = batch.get("mesh", {})
        meta = batch.get("meta")

        initial_images = inputs.get("images")
        camera_poses_batch = inputs.get("camera_poses")

        gt_mesh_data = targets.get("gt_mesh_data", {})
        mesh_paths, normalize_methods = parse_mesh_metadata(meta)
        cache_paths = None
        if self.render_cache is not None:
            cache_paths = self.render_cache.build_paths(
                mesh_paths=mesh_paths,
                normalize_methods=normalize_methods,
                camera_poses_batch=camera_poses_batch,
            )

        mesh_batch = mesh_data.get("normalized")
        if isinstance(mesh_batch, list):
            mesh_batch = None

        if mesh_batch is None:
            mesh_batch = load_meshes_as_batch(
                mesh_paths=mesh_paths,
                normalize_methods=normalize_methods,
                device=camera_poses_batch.device,
                num_workers=self.mesh_load_workers,
            )

        batch_size = camera_poses_batch.shape[0]
        required_keys = ("gt_point_maps", "gt_valid_masks", "depth_z", "depth_z_viz")

        def _as_list(value: Any) -> List[Any]:
            if value is None:
                return [None] * batch_size
            if isinstance(value, list):
                return list(value)
            if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                return list(value.unbind(0))
            return [value] * batch_size

        cache_ready = [True] * batch_size
        if initial_images is None:
            cache_ready = [False] * batch_size
        elif isinstance(initial_images, list):
            for idx, item in enumerate(initial_images):
                if item is None:
                    cache_ready[idx] = False

        for key in required_keys:
            value = gt_mesh_data.get(key)
            if value is None:
                cache_ready = [False] * batch_size
                break
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    if item is None:
                        cache_ready[idx] = False
            elif isinstance(value, torch.Tensor):
                if value.shape[0] != batch_size:
                    cache_ready = [False] * batch_size
                    break

        missing_indices = [idx for idx, ready in enumerate(cache_ready) if not ready]
        rendered = False
        if missing_indices:
            idx_tensor = torch.as_tensor(
                missing_indices, device=camera_poses_batch.device, dtype=torch.long
            )
            subset_mesh_batch = mesh_batch[missing_indices]
            subset_camera_poses = camera_poses_batch.index_select(0, idx_tensor)

            subset_gt_mesh_data: Dict[str, Any] = {}
            for key, value in gt_mesh_data.items():
                if key in required_keys:
                    continue
                if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                    subset_gt_mesh_data[key] = value.index_select(0, idx_tensor)
                elif isinstance(value, list):
                    subset_gt_mesh_data[key] = [value[i] for i in missing_indices]
                else:
                    subset_gt_mesh_data[key] = value

            subset_initial_images, subset_gt_mesh_data = self._render_inputs(
                initial_images=None,
                camera_poses_batch=subset_camera_poses,
                gt_mesh_data=subset_gt_mesh_data,
                mesh_batch=subset_mesh_batch,
            )
            rendered = True

            initial_images_list = _as_list(initial_images)
            for offset, idx in enumerate(missing_indices):
                initial_images_list[idx] = subset_initial_images[offset]
            initial_images = torch.stack(initial_images_list, dim=0)

            for key in required_keys:
                existing_list = _as_list(gt_mesh_data.get(key))
                subset_value = subset_gt_mesh_data.get(key)
                for offset, idx in enumerate(missing_indices):
                    existing_list[idx] = subset_value[offset]
                gt_mesh_data[key] = torch.stack(existing_list, dim=0)
        else:
            if isinstance(initial_images, list):
                initial_images = torch.stack(initial_images, dim=0)
            for key in required_keys:
                value = gt_mesh_data.get(key)
                if isinstance(value, list):
                    gt_mesh_data[key] = torch.stack(value, dim=0)

        if rendered and cache_paths and self.render_cache is not None:
            self.render_cache.save_batch(
                cache_paths=cache_paths,
                mesh_batch=mesh_batch,
                initial_images=initial_images,
                gt_mesh_data=gt_mesh_data,
                is_global_zero=getattr(self.trainer, "is_global_zero", True),
            )
        depth_z_batch = gt_mesh_data.get("depth_z")

        initial_images, camera_poses_batch, depth_z_batch, selection, active_view_count = self._select_initial_views(
            initial_images,
            camera_poses_batch,
            depth_z=depth_z_batch,
            randomize=self.trainer.training,
        )

        trimmed_gt_mesh_data = trim_gt_mesh_data(gt_mesh_data, selection)

        return PreparedBatch(
            initial_images=initial_images,
            camera_poses=camera_poses_batch,
            depth_z=None,
            gt_mesh_data=gt_mesh_data,
            trimmed_gt_mesh_data=trimmed_gt_mesh_data,
            mesh_batch=mesh_batch,
            mesh_paths=mesh_paths,
            selection=selection,
            active_view_count=active_view_count,
        )

    def _infer_next_pose(
        self,
        *,
        initial_images: torch.Tensor,
        camera_poses_batch: torch.Tensor,
        depth_z_batch: Optional[torch.Tensor],
    ) -> PolicyInferenceOutput:
        scene_features, views = self._extract_scene_features(
            initial_images,
            camera_poses_batch,
            depth_z_batch,
        )

        camera_poses_batch_across_views = self._compute_pose_for_across_views_in_ref_view(views)
        policy_output = self.policy_network(scene_features, camera_poses_batch_across_views)

        next_camera_pose, predicted_relative_position, _ = self._compute_policy_pose(
            policy_output,
            camera_poses_batch,
        )

        return PolicyInferenceOutput(
            next_camera_pose=next_camera_pose,
            predicted_relative_position=predicted_relative_position,
        )

    def _maybe_track_policy_gradients(
        self,
        predicted_relative_position: torch.Tensor,
        next_camera_pose: torch.Tensor,
    ) -> None:
        if not self.trainer.training:
            return
        try:
            if predicted_relative_position.requires_grad:
                def _capture_pred_rel_grad(grad: torch.Tensor) -> torch.Tensor:
                    if grad is not None:
                        self._last_predicted_relative_position_grad_norm = (
                            grad.norm(dim=-1).mean().detach()
                        )
                    return grad
                predicted_relative_position.register_hook(_capture_pred_rel_grad)

            if next_camera_pose.requires_grad:
                def _capture_next_pose_grad(grad: torch.Tensor) -> torch.Tensor:
                    if grad is not None and grad.numel() > 0:
                        grad = grad.detach()
                        self._last_next_pose_position_grad_norm = (
                            grad[:, :3].norm(dim=-1).mean()
                        )
                        self._last_next_pose_quaternion_grad_norm = (
                            grad[:, 3:].norm(dim=-1).mean()
                        )
                    return grad
                next_camera_pose.register_hook(_capture_next_pose_grad)
        except RuntimeError:
            self._last_predicted_relative_position_grad_norm = None
            self._last_next_pose_position_grad_norm = None
            self._last_next_pose_quaternion_grad_norm = None

    def _set_last_new_point_maps_render(self, value: Optional[torch.Tensor]) -> None:
        self._last_new_point_maps_grad_norm = None
        if value is None or not value.requires_grad:
            return
        try:
            def _capture_new_point_maps_grad(grad: torch.Tensor) -> torch.Tensor:
                if grad is not None and grad.numel() > 0:
                    self._last_new_point_maps_grad_norm = grad.norm(dim=-1).mean().detach()
                return grad
            value.register_hook(_capture_new_point_maps_grad)
        except RuntimeError:
            self._last_new_point_maps_grad_norm = None

    def _build_loss_dict(
        self,
        loss_components: Dict[str, float],
        random_baseline: Optional[RandomBaselineOutput],
        active_view_count: int,
    ) -> Dict[str, float]:
        logged_loss_keys = (
            "total_loss",
            # "chamfer_loss",
            "weighted_chamfer_loss",
            # "chamfer_pred_points_mean",
            # "chamfer_pred_points_min",
            # "chamfer_pred_points_zero_frac",
            # "chamfer_pred_points_last_view_mean",
            # "chamfer_pred_points_last_view_min",
            # "chamfer_pred_points_last_view_zero_frac",
            # "confidence_loss",
            # "weighted_confidence_loss",
            # "viewpoint_loss",
            # "weighted_viewpoint_loss",
            # "pose_penalty_loss",
            "weighted_pose_penalty_loss",
        )
        loss_dict = {
            key: loss_components[key] for key in logged_loss_keys if key in loss_components
        }
        if random_baseline is not None:
            loss_dict["random_chamfer_loss"] = random_baseline.chamfer_loss
        # loss_dict["num_initial_views"] = float(active_view_count)
        return loss_dict

    def _maybe_compute_random_baseline(
        self,
        *,
        initial_images: torch.Tensor,
        camera_poses_batch: torch.Tensor,
        gt_mesh_data: Dict[str, torch.Tensor],
        mesh_batch,
        mesh_paths: Optional[Sequence[Optional[str]]] = None,
    ) -> Optional[RandomBaselineOutput]:
        if not self.trainer.training or not self.enable_random_baseline:
            return None
        random_chamfer, random_images, random_position_norm_mean = compute_random_baseline(
            renderer=self.renderer,
            loss_fn=self.loss_fn,
            initial_images=initial_images,
            camera_poses_batch=camera_poses_batch,
            gt_mesh_data=gt_mesh_data,
            mesh_batch=mesh_batch,
            mesh_paths=mesh_paths,
        )
        return RandomBaselineOutput(
            chamfer_loss=float(random_chamfer),
            images=random_images,
            position_norm_mean=float(random_position_norm_mean),
        )

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
        return evaluate_candidate_pose(
            renderer=self.renderer,
            loss_fn=self.loss_fn,
            pose=pose,
            initial_images=initial_images,
            camera_poses_batch=camera_poses_batch,
            gt_mesh_data=gt_mesh_data,
            mesh_batch=mesh_batch,
            point_cloud_dir=point_cloud_dir,
            on_new_point_maps=self._set_last_new_point_maps_render,
        )

    def _select_initial_views(
        self,
        initial_images: torch.Tensor,
        camera_poses: torch.Tensor,
        *,
        depth_z: Optional[torch.Tensor] = None,
        randomize: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, int]:
        initial_images, camera_poses, depth_z, selection, num_views = select_initial_views(
            initial_images,
            camera_poses,
            depth_z=depth_z,
            randomize=randomize,
            min_initial_views=self.min_initial_views,
            max_initial_views=self.max_initial_views,
            randomize_initial_views=self.randomize_initial_views,
        )
        self._last_initial_view_count = num_views
        self._last_initial_view_indices = selection.detach().cpu()
        return initial_images, camera_poses, depth_z, selection, num_views

    def _get_log_batch_size(self) -> Optional[int]:
        if self._last_batch_size is None:
            return None
        return int(self._last_batch_size)

    def _process_batch(
        self,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict[str, float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        单个训练步骤

        Args:
            batch: 训练批次数据

        Returns:
            total_loss: 总损失
            loss_dict: 损失字典
            new_images: 渲染的新视图
            initial_images: 初始视图
        """
        prepared = self._prepare_batch(batch)
        self._last_batch_size = int(prepared.camera_poses.shape[0])
        policy_inference = self._infer_next_pose(
            initial_images=prepared.initial_images,
            camera_poses_batch=prepared.camera_poses,
            depth_z_batch=prepared.depth_z,
        )

        self._maybe_track_policy_gradients(
            policy_inference.predicted_relative_position,
            policy_inference.next_camera_pose,
        )

        step_output_dir = resolve_step_output_dir(self) 

        policy_eval = self._evaluate_candidate_pose(
            pose=policy_inference.next_camera_pose,
            initial_images=prepared.initial_images,
            camera_poses_batch=prepared.camera_poses,
            gt_mesh_data=prepared.trimmed_gt_mesh_data,
            mesh_batch=prepared.mesh_batch,
            point_cloud_dir=step_output_dir,
        )

        total_loss = policy_eval.total_loss
        new_images = policy_eval.new_images
        
        loss_dict = None
        
        random_baseline = self._maybe_compute_random_baseline(
            initial_images=prepared.initial_images,
            camera_poses_batch=prepared.camera_poses,
            gt_mesh_data=prepared.trimmed_gt_mesh_data,
            mesh_batch=prepared.mesh_batch,
            mesh_paths=prepared.mesh_paths,
        )
        
        loss_dict = self._build_loss_dict(
            policy_eval.loss_components,
            random_baseline,
            prepared.active_view_count,
        )

        log_step_outputs(
            self,
            prepared=prepared,
            policy_inference=policy_inference,
            policy_eval=policy_eval,
            random_baseline=random_baseline,
            loss_dict=loss_dict,
            step_output_dir=step_output_dir,
        )

        return total_loss, loss_dict, new_images, prepared.initial_images

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

    def training_step(self, batch: Dict, batch_idx: int):
        loss, _, _, _ = self._process_batch(batch)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        loss, _, _, _ = self._process_batch(batch)
        return loss

    def on_validation_epoch_start(self) -> None:
        # Allow one validation image save per validation epoch.
        self._val_images_saved = False

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
            batch_size=self._get_log_batch_size(),
        )
        self.log(
            "gradients/grad_rms",
            grad_rms,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=self._get_log_batch_size(),
        )
        if clip_val is not None and float(clip_val) > 0:
            self.log(
                "gradients/global_norm_post_clip",
                post_clip_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=self._get_log_batch_size(),
            )

    def on_after_backward(self) -> None:
        """Log gradients w.r.t. pose tensors to pinpoint spike sources."""
        if not self.trainer.training:
            self._last_predicted_relative_position_grad_norm = None
            self._last_next_pose_position_grad_norm = None
            self._last_next_pose_quaternion_grad_norm = None
            self._last_new_point_maps_grad_norm = None
            return

        # rel_grad_norm = self._last_predicted_relative_position_grad_norm
        # if rel_grad_norm is not None:
        #     self.log(
        #         "gradients/relative_position_grad_norm",
        #         rel_grad_norm,
        #         on_step=True,
        #         on_epoch=False,
        #         prog_bar=False,
        #         logger=True,
        #     )

        if self._last_new_point_maps_grad_norm is not None:
            self.log(
                "gradients/new_point_maps_render_grad_norm",
                self._last_new_point_maps_grad_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=self._get_log_batch_size(),
            )

        if self._last_next_pose_position_grad_norm is not None:
            self.log(
                "gradients/next_pose_position_grad_norm",
                self._last_next_pose_position_grad_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=self._get_log_batch_size(),
            )
        if self._last_next_pose_quaternion_grad_norm is not None:
            self.log(
                "gradients/next_pose_quaternion_grad_norm",
                self._last_next_pose_quaternion_grad_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=self._get_log_batch_size(),
            )

        self._last_predicted_relative_position_grad_norm = None
        self._last_next_pose_position_grad_norm = None
        self._last_next_pose_quaternion_grad_norm = None
        self._last_new_point_maps_grad_norm = None

    def transfer_batch_to_device(self, batch: Dict[str, Any], device: torch.device, dataloader_idx: int):
        
        def move_item(x):
            # 处理 Tensor：只移动设备，不改变精度(保持 float32 以保证几何计算稳定)
            if isinstance(x, torch.Tensor):
                # non_blocking=True 是一个小优化，允许数据传输和GPU计算重叠
                return x.to(device, non_blocking=True)
                
            # 处理 PyTorch3D Meshes：调用其内部的 .to() 方法
            if isinstance(x, Meshes):
                return x.to(device)
                
            return x

        moved: Dict[str, Any] = {}
        # 只处理这三个 key，meta 信息保持原样留在 CPU
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
