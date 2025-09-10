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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision

from ..models import VGGTWrapper, BaseNBVPolicy
from ..rendering import DifferentiableRenderer
from .losses import ReconstructionLoss, ChamferDistance


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
                 learning_rate: float = 1e-4,
                 log_dir: str = "runs/nbv_experiment",
                 device: str = "cuda"):
        """
        初始化训练器
        
        Args:
            vggt_wrapper: 冻结的VGGT基础模型
            policy_network: 可训练的NBV策略网络
            renderer: 可微分渲染器
            loss_fn: 重建质量损失函数
            learning_rate: 学习率
            log_dir: TensorBoard日志目录
            device: 计算设备
        """
        self.vggt_wrapper = vggt_wrapper
        self.policy_network = policy_network
        self.renderer = renderer
        self.loss_fn = loss_fn
        self.device = device
        
        # 初始化TensorBoard Writer
        self.writer = SummaryWriter(log_dir)
        
        # 优化器（只优化策略网络）
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
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
        
        batch_size = initial_images.shape[0]
        
        # 步骤1: 状态编码 - VGGT提取场景特征
        scene_features = self.vggt_wrapper.extract_scene_features(initial_images)
        
        # 步骤2: 动作提议 - 策略网络输出下一个相机位姿
        next_camera_pose = self.policy_network(scene_features)
        
        # 步骤3: 环境交互 - 可微分渲染生成新视图
        batched_mesh = gt_mesh_data['normalized_mesh'] # 这现在是单个批次化的 Meshes 对象

        # 确保整个批次化的 mesh 对象位于渲染器设备上
        batched_mesh = batched_mesh.to(self.renderer.device)
        new_images = self.renderer(
            gt_mesh=batched_mesh,
            camera_poses=next_camera_pose,
            pose_format=self.policy_network.output_mode,
            lighting_type="ambient"
        )

        # 确保与 initial_images 在同一设备
        if new_images.device != initial_images.device:
            new_images = new_images.to(initial_images.device)
        
        # 为每个batch sample添加新图像
        combined_images_list = []
        for i in range(batch_size):
            # 取出当前样本的N个初始图像
            sample_initial = initial_images[i]  # [N, 3, H, W]
            sample_new = new_images[i:i+1]      # [1, 3, H, W]
            
            # 组合为N+1个图像
            combined = torch.cat([sample_initial, sample_new], dim=0)  # [N+1, 3, H, W]
            combined_images_list.append(combined)
        
        # 步骤4: 质量评估 - VGGT重建并计算质量
        # 将每个样本的图像沿 batch 维度堆叠，得到 [B, N+1, 3, H, W]
        combined_images_batch = torch.stack(combined_images_list, dim=0)
        # VGGT一次性对整个batch进行重建与评估
        recon_data = self.vggt_wrapper.reconstruct_and_evaluate(
            combined_images_batch  # [B, N+1, 3, H, W]
        )
        # 计算重建质量损失：直接传入batched的 GT 数据结构（参考 datasets.py）
        total_loss, loss_components = self.loss_fn(recon_data, gt_mesh_data, new_images, return_components=True)
        
        # 步骤5: 策略更新 - 反向传播（仅训练时）
        if backprop:
            total_loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
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
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
        
        # TensorBoard logging
        if backprop:
            # 记录总损失和学习率
            self.writer.add_scalar('train/total_loss', loss_dict['total_loss'], self.global_step)
            self.writer.add_scalar('train/learning_rate', loss_dict['learning_rate'], self.global_step)
            
            # 记录各个损失组件（原始值）
            self.writer.add_scalar('train/losses/chamfer_loss', loss_dict['chamfer_loss'], self.global_step)
            self.writer.add_scalar('train/losses/confidence_loss', loss_dict['confidence_loss'], self.global_step)
            self.writer.add_scalar('train/losses/viewpoint_loss', loss_dict['viewpoint_loss'], self.global_step)
            
            # 记录加权后的损失组件
            self.writer.add_scalar('train/weighted_losses/chamfer_loss', loss_dict['weighted_chamfer_loss'], self.global_step)
            self.writer.add_scalar('train/weighted_losses/confidence_loss', loss_dict['weighted_confidence_loss'], self.global_step)
            self.writer.add_scalar('train/weighted_losses/viewpoint_loss', loss_dict['weighted_viewpoint_loss'], self.global_step)

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
                self.writer.add_image('val/initial_views', initial_grid, self.global_step)

                new_grid = torchvision.utils.make_grid(new_images, nrow=1)
                self.writer.add_image('val/next_best_view', new_grid, self.global_step)

            progress_bar.set_postfix({
                "val_loss": f"{loss_dict['total_loss']:.4f}"
            })
        
        avg_loss_dict = self._average_loss_dicts(epoch_losses)
        
        # 记录验证损失
        self.writer.add_scalar('val/total_loss', avg_loss_dict['total_loss'], self.global_step)
        
        # 记录各个验证损失组件（原始值）
        self.writer.add_scalar('val/losses/chamfer_loss', avg_loss_dict['chamfer_loss'], self.global_step)
        self.writer.add_scalar('val/losses/confidence_loss', avg_loss_dict['confidence_loss'], self.global_step)
        self.writer.add_scalar('val/losses/viewpoint_loss', avg_loss_dict['viewpoint_loss'], self.global_step)
        
        # 记录加权后的验证损失组件
        self.writer.add_scalar('val/weighted_losses/chamfer_loss', avg_loss_dict['weighted_chamfer_loss'], self.global_step)
        self.writer.add_scalar('val/weighted_losses/confidence_loss', avg_loss_dict['weighted_confidence_loss'], self.global_step)
        self.writer.add_scalar('val/weighted_losses/viewpoint_loss', avg_loss_dict['weighted_viewpoint_loss'], self.global_step)

        return avg_loss_dict
    
    def train(self, 
             train_loader: DataLoader,
             val_loader: Optional[DataLoader] = None,
             num_epochs: int = 100,
             save_dir: str = "checkpoints"):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            save_dir: 模型保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Policy network parameters: {sum(p.numel() for p in self.policy_network.parameters())}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss_dict = self.train_epoch(train_loader)
            
            # 验证
            val_loss_dict = None
            if val_loader is not None:
                val_loss_dict = self.validate_epoch(val_loader)
            
            # 学习率调度
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