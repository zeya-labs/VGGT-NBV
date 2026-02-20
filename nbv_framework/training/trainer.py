"""
NBV策略训练器
实现端到端的目标驱动策略学习训练流程
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING

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
from .trainer_batch import NBVTrainerBatchMixin
from .trainer_eval import NBVTrainerEvalMixin
from .trainer_policy import NBVTrainerPolicyMixin
from .trainer_test import NBVTrainerTestMixin
from ..cache.render_cache import RenderCache


class NBVTrainer(
    NBVTrainerTestMixin,
    NBVTrainerEvalMixin,
    NBVTrainerPolicyMixin,
    NBVTrainerBatchMixin,
    LightningModule,
):
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
                 test_chamfer_metrics: Optional[Sequence[str]] = None,
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
        self.test_chamfer_metrics = list(test_chamfer_metrics) if test_chamfer_metrics else ["geomloss"]
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

        loss_dict = self._build_loss_dict(
            policy_eval.loss_components,
            prepared.active_view_count,
        )

        log_step_outputs(
            self,
            prepared=prepared,
            policy_inference=policy_inference,
            policy_eval=policy_eval,
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
