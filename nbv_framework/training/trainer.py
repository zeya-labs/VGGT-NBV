"""
NBV策略训练器

实现端到端的目标驱动策略学习训练流程：
1. 状态编码：VGGT提取场景特征
2. 动作提议：策略网络输出相机位姿
3. 环境交互：可微分渲染生成新视图
4. 质量评估：VGGT重建并计算质量损失
5. 策略更新：反向传播更新策略网络
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
import os
import logging
import math
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.utils.camera_conversions import opencv_from_cameras_projection

from ..models import VGGTWrapper, BaseNBVPolicy
from ..rendering import DifferentiableRenderer
from .loss import ReconstructionLoss, ChamferDistance
from ..utils.camera_utils import position_to_pose_tensor


class NBVTrainer:
    """
    NBV策略训练器
    
    实现完整的目标驱动策略学习训练流程。
    """
    
    def __init__(self,
                 vggt_wrapper: VGGTWrapper,
                 policy_network: BaseNBVPolicy,
                 renderer: DifferentiableRenderer,
                 loss_fn: ReconstructionLoss,
                 num_epochs: int = 1000,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 log_dir: str = "runs/nbv_experiment",
                 device: str = "cuda",
                 enable_validation: bool = False):
        """
        初始化训练器
        
        Args:
            vggt_wrapper: 冻结的VGGT基础模型
            policy_network: 可训练的NBV策略网络
            renderer: 可微分渲染器
            loss_fn: 重建质量损失函数
            learning_rate: 学习率
            weight_decay: 权重衰减
            log_dir: TensorBoard日志目录
            device: 计算设备
            enable_validation: 是否在训练过程中执行验证流程
        """
        self.vggt_wrapper = vggt_wrapper
        self.policy_network = policy_network
        self.renderer = renderer
        self.loss_fn = loss_fn
        self.device = device
        self.num_epochs = num_epochs
        self.log_dir = log_dir
        self.enable_validation = enable_validation

        # 启用VGGT梯度捕获，便于调试NBV梯度链路
        self._vggt_grad_keys = ("world_points", "world_points_conf")
        self.vggt_wrapper.configure_gradient_capture(
            enable=True,
            keys=self._vggt_grad_keys,
            capture_input=True
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
            self.optimizer, T_max=self.num_epochs, eta_min=1e-6
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

    def _log_vggt_gradient_stats(self, new_images: torch.Tensor) -> None:
        """记录VGGT相关的梯度统计信息到TensorBoard。"""
        grad_stats = self.vggt_wrapper.collect_gradient_stats()

        for key in getattr(self, "_vggt_grad_keys", ("depth", "world_points_from_depth")):
            norm_key = f"{key}/grad_norm"
            mean_key = f"{key}/grad_mean_abs"

            norm_val = grad_stats.get(norm_key, 0.0)
            mean_val = grad_stats.get(mean_key, 0.0)
            has_grad = 1.0 if norm_key in grad_stats else 0.0

            self.writer.add_scalar(f'train/gradients/vggt/{key}_grad_norm', norm_val, self.global_step)
            self.writer.add_scalar(f'train/gradients/vggt/{key}_grad_mean_abs', mean_val, self.global_step)
            self.writer.add_scalar(f'train/gradients/vggt/{key}_has_grad', has_grad, self.global_step)

        input_norm = grad_stats.get('input/grad_norm', 0.0)
        input_mean = grad_stats.get('input/grad_mean_abs', 0.0)
        input_has_grad = 1.0 if 'input/grad_norm' in grad_stats else 0.0

        self.writer.add_scalar('train/gradients/vggt/input_grad_norm', input_norm, self.global_step)
        self.writer.add_scalar('train/gradients/vggt/input_grad_mean_abs', input_mean, self.global_step)
        self.writer.add_scalar('train/gradients/vggt/input_has_grad', input_has_grad, self.global_step)

        has_new_grad = 1.0 if new_images.grad is not None else 0.0
        self.writer.add_scalar('train/gradients/new_view_has_grad', has_new_grad, self.global_step)

        if new_images.grad is not None:
            new_view_norm = new_images.grad.norm().detach().item()
            new_view_mean = new_images.grad.abs().mean().detach().item()
        else:
            new_view_norm = 0.0
            new_view_mean = 0.0
            self.logger.warning("new_images gradient is None at global step %d", self.global_step)

        self.writer.add_scalar('train/gradients/new_view_grad_norm', new_view_norm, self.global_step)
        self.writer.add_scalar('train/gradients/new_view_grad_mean_abs', new_view_mean, self.global_step)
    
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
        if backprop:
            self.policy_network.train()
            self.optimizer.zero_grad()
        else:
            self.policy_network.eval()
        
        initial_images = batch["initial_images"]  # [B, N, 3, H, W]
        gt_mesh_data = batch["gt_mesh_data"]
        camera_poses_batch = batch.get("camera_poses")

        if camera_poses_batch is None:
            raise KeyError("Batch is missing 'camera_poses', which are required for correspondence-guided losses.")

        # 步骤1: 状态编码 - VGGT提取场景特征
        scene_features = self.vggt_wrapper.extract_scene_features(initial_images)
        
        # 步骤2: 动作提议 - 策略网络输出下一个相机位姿
        next_camera_pose = self.policy_network(scene_features)
        
        # 检查输出维度，如果是[B,3]则转换为[B,7]
        if next_camera_pose.shape[-1] == 3:
            next_camera_pose = position_to_pose_tensor(next_camera_pose)

        if backprop and next_camera_pose.requires_grad:
            next_camera_pose.retain_grad()

        # 记录位姿数据到TensorBoard
        if self.global_step % 1 == 0:  # 每10步记录一次，避免过于频繁
            # 记录位置信息（前3维）
            positions = next_camera_pose[:, :3]  # [B, 3]
            
            # 记录四元数信息（后4维）
            quaternions = next_camera_pose[:, 3:]  # [B, 4]

            # 记录位置的统计信息
            position_norms = torch.norm(positions, dim=1)  # 计算位置向量的模长
            self.writer.add_scalar('camera_pose/position_norm_mean', position_norms.mean(), self.global_step)
            if position_norms.numel() > 1:  # 只有当样本数大于1时才计算标准差
                self.writer.add_scalar('camera_pose/position_norm_std', position_norms.std(), self.global_step)
            
            # 记录四元数的模长（应该接近1）
            quaternion_norms = torch.norm(quaternions, dim=1)
            # self.writer.add_scalar('camera_pose/quaternion_norm_mean', quaternion_norms.mean(), self.global_step)
            if quaternion_norms.numel() > 1:  # 只有当样本数大于1时才计算标准差
                self.writer.add_scalar('camera_pose/quaternion_norm_std', quaternion_norms.std(), self.global_step)

        # 构建包含初始视图和新视图的相机位姿列表
        camera_poses_batch = camera_poses_batch.to(next_camera_pose.dtype)

        if camera_poses_batch.dim() == 2:
            camera_poses_batch = camera_poses_batch.unsqueeze(1)
        combined_camera_poses = torch.cat([
            camera_poses_batch,
            next_camera_pose.unsqueeze(1)
        ], dim=1)  # [B, N+1, 7]

        # 步骤3: 环境交互 - 可微分渲染生成新视图
        batched_mesh = gt_mesh_data['normalized_mesh'] # 这现在是单个批次化的 Meshes 对象

        # 确保整个批次化的 mesh 对象位于渲染器设备上
        batched_mesh = batched_mesh.to(self.renderer.device)
        new_images = self.renderer(
            gt_mesh=batched_mesh,
            camera_poses=next_camera_pose,
        )

        # 确保与 initial_images 在同一设备
        if new_images.device != initial_images.device:
            new_images = new_images.to(initial_images.device)

        if backprop and new_images.requires_grad:
            new_images.retain_grad()
        
        # 步骤4: 质量评估 - VGGT重建并计算质量
        # 将 new_images 从 [B, 3, H, W] 扩展为 [B, 1, 3, H, W]
        new_images_expanded = new_images.unsqueeze(1)  # [B, 1, 3, H, W]
        
        # 直接在第二个维度上拼接，得到 [B, N+1, 3, H, W]
        combined_images_batch = torch.cat([initial_images, new_images_expanded], dim=1)
        
        # 保存视图数据（图像 + 相机标定）到日志目录
        self._save_combined_images(
            combined_images_batch=combined_images_batch,
            combined_camera_poses=combined_camera_poses
        )
        
        # VGGT一次性对整个batch进行重建与评估
        recon_data = self.vggt_wrapper.reconstruct_and_evaluate(
            combined_images_batch  # [B, N+1, 3, H, W]
        )
        # print("===================loss开始计算===================")
        # 计算重建质量损失
        # 在训练时传递writer和step参数以启用点云可视化
        if backprop:
            total_loss, loss_components = self.loss_fn(
                recon_data, gt_mesh_data, combined_images_batch,
                combined_camera_poses,
                return_components=True, writer=self.writer, step=self.global_step,
                train_flag=True
            )
        else:
            total_loss, loss_components = self.loss_fn(
                recon_data, gt_mesh_data, combined_images_batch,
                combined_camera_poses,
                return_components=True, writer=self.writer, step=self.val_image_step,
                train_flag=False
            )
        # print("===================loss计算完成===================")
        # 步骤5: 策略更新 - 反向传播（仅训练时）
        if backprop:
            total_loss.backward()

            pose_grad = next_camera_pose.grad
            pose_grad_norm = pose_grad.norm().detach().item() if pose_grad is not None else 0.0
            pose_grad_mean = pose_grad.abs().mean().detach().item() if pose_grad is not None else 0.0

            if pose_grad is None:
                self.logger.warning("next_camera_pose grad is None")

            self.writer.add_scalar('train/gradients/next_camera_grad_norm', pose_grad_norm, self.global_step)
            self.writer.add_scalar('train/gradients/next_camera_grad_mean_abs', pose_grad_mean, self.global_step)
            self.writer.add_scalar('train/gradients/next_camera_has_grad', 1.0 if pose_grad is not None else 0.0, self.global_step)

            # grad_stats, missing = [], []
            # for name, param in self.policy_network.named_parameters():
            #     if not param.requires_grad:
            #         continue
            #     grad = param.grad
            #     if grad is None:
            #         missing.append(name)
            #         continue
            #     grad_stats.append((name, grad.norm().item(), grad.abs().mean().item()))

            # for name, norm_val, mean_abs in grad_stats:  # 只打印前几个防止刷屏
            #     self.logger.info(
            #         "grad %s |norm| %.4e |mean|grad| %.4e",
            #         name, norm_val, mean_abs
            #     )
            # if len(grad_stats) > 8:
            #     overall = torch.sqrt(sum(
            #         param.grad.pow(2).sum()
            #         for _, param in self.policy_network.named_parameters()
            #         if param.grad is not None
            #     )).item()
            #     self.logger.info("total grad norm %.4e (remaining layers truncated)", overall)
            # if missing:
            #     self.logger.warning("layers without grad: %s", ", ".join(missing[:5]))

            # 记录VGGT与新视图的梯度统计信息
            self._log_vggt_gradient_stats(new_images)

            # 记录策略网络梯度的整体范数（clip_grad_norm_返回裁剪前范数）
            total_policy_norm = torch.nn.utils.clip_grad_norm_(
                self.policy_network.parameters(), max_norm=1.0
            )
            self.writer.add_scalar('train/gradients/policy_total_norm', total_policy_norm.item(), self.global_step)
            # self.writer.add_scalar('train/gradients/policy_missing_params', float(len(missing)), self.global_step)

            self.optimizer.step()
        
        # 记录
        loss_dict = {
            "total_loss": loss_components['total_loss'],
            "chamfer_loss": loss_components['chamfer_loss'],
            "weighted_chamfer_loss": loss_components['weighted_chamfer_loss'],
            "confidence_loss": loss_components['confidence_loss'],
            "weighted_confidence_loss": loss_components['weighted_confidence_loss'],
            "viewpoint_loss": loss_components['viewpoint_loss'],
            "weighted_viewpoint_loss": loss_components['weighted_viewpoint_loss'],
            "pose_penalty_loss": loss_components['pose_penalty_loss'],
            "weighted_pose_penalty_loss": loss_components['weighted_pose_penalty_loss'],
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
        
        # TensorBoard logging
        if backprop:
            # 记录总损失和学习率
            self.writer.add_scalar('train/total_loss', loss_dict['total_loss'], self.global_step)
            self.writer.add_scalar('train/learning_rate', loss_dict['learning_rate'], self.global_step)
            
            # 记录各个损失组件（原始值）
            self.writer.add_scalar('train/losses/chamfer_loss', loss_dict['chamfer_loss'], self.global_step)
            # self.writer.add_scalar('train/losses/confidence_loss', loss_dict['confidence_loss'], self.global_step)
            # self.writer.add_scalar('train/losses/viewpoint_loss', loss_dict['viewpoint_loss'], self.global_step)
            
            # # 记录加权后的损失组件
            # self.writer.add_scalar('train/weighted_losses/chamfer_loss', loss_dict['weighted_chamfer_loss'], self.global_step)
            # self.writer.add_scalar('train/weighted_losses/confidence_loss', loss_dict['weighted_confidence_loss'], self.global_step)
            # self.writer.add_scalar('train/weighted_losses/viewpoint_loss', loss_dict['weighted_viewpoint_loss'], self.global_step)
            self.writer.add_scalar('train/losses/pose_penalty_loss', loss_dict['pose_penalty_loss'], self.global_step)
            # self.writer.add_scalar('train/weighted_losses/pose_penalty_loss', loss_dict['weighted_pose_penalty_loss'], self.global_step)
            
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
                "lr": f"{loss_dict['learning_rate']:.2e}"
            })
        
        # 计算epoch平均损失
        avg_loss_dict = self._average_loss_dicts(epoch_losses)
        
        return avg_loss_dict
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        epoch_losses = []
        
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
                "val_loss": f"{loss_dict['total_loss']:.4f}"
            })
        
        avg_loss_dict = self._average_loss_dicts(epoch_losses)
        
        # 记录验证损失（以 epoch 作为 step）
        self.writer.add_scalar('val/total_loss', avg_loss_dict['total_loss'], self.current_epoch)
        
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
        """将batch数据递归移到设备（支持嵌套的dict/list/tuple结构）"""
        def _to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device)
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
    
    def _compute_intrinsics(self, height: int, width: int, fov_degrees: float) -> torch.Tensor:
        """构建简单的针孔相机内参矩阵。"""
        fov_radians = math.radians(float(fov_degrees))
        fy = 0.5 * height / math.tan(fov_radians / 2.0)
        fx = 0.5 * width / math.tan(fov_radians / 2.0)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        intrinsics = torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        return intrinsics

    # def pose_to_opencv_cam2world(self, pose: torch.Tensor) -> torch.Tensor:
    #     """
    #     将 PyTorch3D 格式的位姿 (+X Left, +Y Up, +Z Forward)
    #     转换为 OpenCV 约定 (+X Right, +Y Down, +Z Forward) 的 4x4 cam2world 矩阵。
    #     """
    #     if pose.numel() != 7:
    #         raise ValueError(f"Expected pose tensor with 7 elements, got shape {pose.shape}.")

    #     pose = pose.to(dtype=torch.float32)
    #     device = pose.device

    #     # 1. 提取位置 C
    #     position = pose[:3]

    #     # 2. 计算 PyTorch3D 相机的 cam2world 旋转
    #     quaternion_xyzw = pose[3:]
    #     quaternion_wxyz = torch.tensor([
    #         quaternion_xyzw[3], quaternion_xyzw[0],
    #         quaternion_xyzw[1], quaternion_xyzw[2],
    #     ], device=device).unsqueeze(0)

    #     # quaternion_to_matrix 得到 R_w2c_pytorch3d
    #     R_w2c_pyt = quaternion_to_matrix(quaternion_wxyz)[0]
    #     # 转置得到 R_c2w_pytorch3d (+X Left, +Y Up, +Z Forward)
    #     R_c2w_pyt = R_w2c_pyt.T
        
    #     # 3. 定义从 PyTorch3D 相机到 OpenCV 相机的变换
    #     # PyT (+X_L, +Y_U, +Z_F) -> OpenCV (+X_R, +Y_D, +Z_F)
    #     # X翻转(-1), Y翻转(-1), Z不变(1)
    #     axis_flip = torch.tensor([
    #         [-1.0,  0.0,  0.0],
    #         [ 0.0, -1.0,  0.0],
    #         [ 0.0,  0.0,  1.0],
    #     ], dtype=pose.dtype, device=device)

    #     # 4. 计算 OpenCV cam2world 的最终旋转矩阵
    #     # R_c2w_opencv = R_c2w_pyt @ axis_flip
    #     R_c2w_opencv = torch.matmul(R_c2w_pyt, axis_flip)

    #     # 5. 组装最终的 4x4 cam2world 矩阵
    #     cam2world = torch.eye(4, dtype=pose.dtype, device=device)
    #     cam2world[:3, :3] = R_c2w_opencv
    #     cam2world[:3, 3] = position

    #     return cam2world

    def pose_to_opencv_cam2world_with_official_func(
        self,
        pose: torch.Tensor,
        image_size: Tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """
        使用官方的 _opencv_from_cameras_projection 函数，
        将 PyTorch3D 格式的位姿转换为 OpenCV 约定 的 4x4 cam2world 矩阵。
        """
        if pose.numel() != 7:
            raise ValueError(f"Expected pose tensor with 7 elements, got shape {pose.shape}.")

        pose = pose.to(dtype=torch.float32)
        device = pose.device

        # === 第1步: 将您的 cam2world `pose` 转换为 Pytorch3D `world2cam` (R, T) ===
        # 这是我们之前验证过的，从您的位姿到Pytorch3D相机参数的正确转换
        position_c2w = pose[:3]  # 相机在世界中的位置 C
        quaternion_xyzw = pose[3:]
        quaternion_wxyz = torch.tensor([
            quaternion_xyzw[3], quaternion_xyzw[0],
            quaternion_xyzw[1], quaternion_xyzw[2],
        ], device=device).unsqueeze(0)

        # 从四元数得到 Pytorch3D 的 world-to-camera 旋转
        R_w2c_pyt = quaternion_to_matrix(quaternion_wxyz) # Shape: [1, 3, 3]

        # 从世界位置 C 和 R_w2c 计算 Pytorch3D 的 world-to-camera 平移 T
        # T = -C @ R  (行向量约定)
        T_w2c_pyt = -(position_c2w.unsqueeze(0).unsqueeze(0) @ R_w2c_pyt).squeeze(1) # Shape: [1, 3]

        # 创建一个临时的 PyTorch3D 相机对象
        # 注意：相机内参(focal_length, principal_point)对于位姿转换是无关的，
        # 但创建对象时需要它们，所以我们使用默认值。
        cameras = PerspectiveCameras(
            R=R_w2c_pyt,
            T=T_w2c_pyt,
            device=device
        )
        
        image_size_tensor = torch.tensor([image_size], device=device, dtype=torch.float32)

        # === 第2步: 调用官方函数，得到 OpenCV 风格的 `world2cam` 参数 ===
        # R_w2c_opencv: 从世界旋转到OpenCV相机的旋转矩阵
        # t_w2c_opencv: 从世界平移到OpenCV相机的平移向量
        R_w2c_opencv, t_w2c_opencv, _ = opencv_from_cameras_projection(
            cameras,
            image_size_tensor
        )

        # === 第3步: 将 OpenCV `world2cam` (R, t) 转换为 `cam2world` 4x4 矩阵 ===
        # 这是标准的逆变换操作
        
        # cam2world 的旋转是 world2cam 旋转的逆(转置)
        R_c2w_opencv = R_w2c_opencv.transpose(1, 2)  # Shape: [1, 3, 3]

        # cam2world 的平移(即相机世界位置) C = -R_c2w @ t_w2c
        position_c2w_opencv = -torch.bmm(R_c2w_opencv, t_w2c_opencv.unsqueeze(-1)).squeeze(-1) # Shape: [1, 3]

        # 组装最终的 4x4 cam2world 矩阵
        cam2world = torch.eye(4, dtype=pose.dtype, device=device)
        cam2world[:3, :3] = R_c2w_opencv[0]
        cam2world[:3, 3] = position_c2w_opencv[0]

        return cam2world

    def _save_combined_images(
        self,
        combined_images_batch: torch.Tensor,
        combined_camera_poses: torch.Tensor,
        fov_degrees: float = 60.0,
    ) -> None:
        """
        保存视图的图像与标定数据到log_dir下的images文件夹。

        Args:
            combined_images_batch: [B, N+1, 3, H, W] 的图片张量
            combined_camera_poses: [B, N+1, 7] 的相机位姿张量
            fov_degrees: 渲染视场角（默认60°），用于构造内参矩阵
        """
        if combined_images_batch is None or combined_camera_poses is None:
            return

        # 确保形状匹配
        if combined_images_batch.shape[:2] != combined_camera_poses.shape[:2]:
            self.logger.warning(
                "Skip saving combined views: image batch shape %s incompatible with pose batch shape %s.",
                combined_images_batch.shape,
                combined_camera_poses.shape,
            )
            return

        images_dir = os.path.join(self.log_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        step_dir = os.path.join(images_dir, f"step_{self.global_step:06d}")
        os.makedirs(step_dir, exist_ok=True)

        images_cpu = combined_images_batch.detach().cpu()
        poses_cpu = combined_camera_poses.detach().cpu()

        batch_size, num_views, _, height, width = images_cpu.shape
        # print("combined_images_batch.shape",combined_images_batch.shape)
        # print("combined_camera_poses.shape",combined_camera_poses.shape)
        intrinsics = self._compute_intrinsics(height, width, fov_degrees)

        for batch_idx in range(batch_size):
            batch_dir = os.path.join(step_dir, f"batch_{batch_idx:03d}")
            os.makedirs(batch_dir, exist_ok=True)

            for view_idx in range(num_views):
                raw_img = images_cpu[batch_idx, view_idx]
                pose_tensor = poses_cpu[batch_idx, view_idx]

                # PNG保存仍使用[3,H,W]且范围[0,1]
                png_img = torch.clamp(raw_img, 0.0, 1.0)
                png_path = os.path.join(batch_dir, f"image_{view_idx:02d}.png")
                torchvision.utils.save_image(png_img, png_path)

                # 视图数据按(H, W, 3)且像素范围[0,255]
                img_uint8 = (png_img.permute(1, 2, 0) * 255.0).round().to(torch.uint8)
                # print("保存",pose_tensor)
                cam2world = self.pose_to_opencv_cam2world_with_official_func(pose_tensor)

                view_payload = {
                    "img": img_uint8,
                    "intrinsics": intrinsics.clone(),
                    "camera_poses": cam2world,
                    "is_metric_scale": torch.tensor([False], dtype=torch.bool),
                }

                payload_path = os.path.join(batch_dir, f"view_{view_idx:02d}.pt")
                torch.save(view_payload, payload_path)

        self.logger.info(
            "Saved view data for %d batches (%d views each) to %s",
            batch_size,
            num_views,
            step_dir,
        )
