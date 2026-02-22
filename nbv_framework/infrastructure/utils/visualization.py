"""
可视化工具

用于可视化重建结果、训练过程、NBV策略等。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os

from loguru import logger


def visualize_reconstruction(recon_data: Dict[str, torch.Tensor],
                           save_path: Optional[str] = None,
                           show_confidence: bool = True):
    """
    可视化重建结果
    
    Args:
        recon_data: VGGT重建数据
        save_path: 保存路径
        show_confidence: 是否显示置信度
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 提取数据
    world_points = recon_data.get("world_points")  # [B, S, H, W, 3]
    world_points_conf = recon_data.get("world_points_conf")  # [B, S, H, W]
    depth = recon_data.get("depth")  # [B, S, H, W, 1]
    images = recon_data.get("images")  # [B, S, 3, H, W]
    
    if world_points is None:
        logger.warning("No world points available for visualization")
        return
    
    # 处理批次维度
    if len(world_points.shape) == 5:
        world_points = world_points[0]  # 取第一个batch
        if world_points_conf is not None:
            world_points_conf = world_points_conf[0]
        if depth is not None:
            depth = depth[0]
        if images is not None:
            images = images[0]
    
    num_views = world_points.shape[0]
    
    # 子图布局
    rows = 2
    cols = max(3, num_views)
    
    # 显示输入图像
    if images is not None:
        for i in range(min(num_views, cols)):
            ax = fig.add_subplot(rows, cols, i + 1)
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(f"View {i+1}")
            ax.axis('off')
    
    # 3D点云可视化
    ax_3d = fig.add_subplot(rows, cols, cols + 1, projection='3d')
    
    # 收集所有高置信度点
    all_points = []
    all_colors = []
    
    for view_idx in range(num_views):
        points = world_points[view_idx]  # [H, W, 3]
        
        if world_points_conf is not None:
            conf = world_points_conf[view_idx]  # [H, W]
            high_conf_mask = conf > 0.5
        else:
            high_conf_mask = torch.ones(points.shape[:2], dtype=torch.bool)
        
        # 提取高置信度点
        valid_points = points[high_conf_mask].cpu().numpy()
        
        if len(valid_points) > 0:
            # 随机采样以避免点太多
            if len(valid_points) > 1000:
                indices = np.random.choice(len(valid_points), 1000, replace=False)
                valid_points = valid_points[indices]
            
            all_points.append(valid_points)
            
            # 为不同视图分配不同颜色
            colors = plt.cm.tab10(view_idx / num_views)[:3]
            all_colors.extend([colors] * len(valid_points))
    
    if all_points:
        all_points = np.vstack(all_points)
        all_colors = np.array(all_colors)
        
        ax_3d.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2],
                     c=all_colors, s=1, alpha=0.6)
        
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title('Reconstructed Point Cloud')
    
    # 深度图可视化
    if depth is not None:
        ax_depth = fig.add_subplot(rows, cols, cols + 2)
        depth_vis = depth[0, :, :, 0].cpu().numpy()  # 第一个视图的深度
        im = ax_depth.imshow(depth_vis, cmap='viridis')
        ax_depth.set_title('Depth Map (View 1)')
        ax_depth.axis('off')
        plt.colorbar(im, ax=ax_depth, fraction=0.046, pad=0.04)
    
    # 置信度图可视化
    if show_confidence and world_points_conf is not None:
        ax_conf = fig.add_subplot(rows, cols, cols + 3)
        conf_vis = world_points_conf[0].cpu().numpy()  # 第一个视图的置信度
        im = ax_conf.imshow(conf_vis, cmap='hot', vmin=0, vmax=1)
        ax_conf.set_title('Confidence Map (View 1)')
        ax_conf.axis('off')
        plt.colorbar(im, ax=ax_conf, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Visualization saved to {}", save_path)
    
    plt.show()


def plot_training_curves(train_losses: List[float],
                        val_losses: Optional[List[float]] = None,
                        save_path: Optional[str] = None):
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加最佳点标记
    if val_losses:
        best_epoch = np.argmin(val_losses) + 1
        best_loss = min(val_losses)
        plt.plot(best_epoch, best_loss, 'ro', markersize=8, 
                label=f'Best Val Loss: {best_loss:.4f} (Epoch {best_epoch})')
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Training curves saved to {}", save_path)
    
    plt.show()


def visualize_nbv_strategy(camera_poses: torch.Tensor,
                          scene_center: torch.Tensor = None,
                          save_path: Optional[str] = None):
    """
    可视化NBV策略的相机位姿
    
    Args:
        camera_poses: 相机位姿序列 [T, 3] (spherical) 或 [T, 7] (cartesian)
        scene_center: 场景中心 [3]
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if scene_center is None:
        scene_center = torch.zeros(3)
    
    # 处理不同的位姿格式
    if camera_poses.shape[1] == 3:  # spherical
        # 转换为笛卡尔坐标
        theta, phi, radius = camera_poses[:, 0], camera_poses[:, 1], camera_poses[:, 2]
        x = radius * torch.sin(phi) * torch.cos(theta)
        y = radius * torch.sin(phi) * torch.sin(theta)
        z = radius * torch.cos(phi)
        positions = torch.stack([x, y, z], dim=1)
    else:  # cartesian
        positions = camera_poses[:, :3]
    
    positions = positions.cpu().numpy()
    scene_center = scene_center.cpu().numpy()
    
    # 绘制相机轨迹
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
           'b-', linewidth=2, alpha=0.7, label='Camera Trajectory')
    
    # 绘制相机位置
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
              c=range(len(positions)), cmap='viridis', s=50, 
              label='Camera Positions')
    
    # 绘制场景中心
    ax.scatter(scene_center[0], scene_center[1], scene_center[2],
              c='red', s=100, marker='*', label='Scene Center')
    
    # 绘制视线
    for i in range(len(positions)):
        ax.plot([positions[i, 0], scene_center[0]],
               [positions[i, 1], scene_center[1]],
               [positions[i, 2], scene_center[2]],
               'k--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('NBV Strategy Visualization')
    ax.legend()
    
    # 设置相等的轴比例
    max_range = np.array([positions.max() - positions.min()]).max() / 2.0
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("NBV strategy visualization saved to {}", save_path)
    
    plt.show()


def create_training_report(train_losses: List[float],
                          val_losses: List[float],
                          config: Dict,
                          save_dir: str):
    """
    创建训练报告
    
    Args:
        train_losses: 训练损失
        val_losses: 验证损失
        config: 训练配置
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, 
                        os.path.join(save_dir, "training_curves.png"))
    
    # 创建文本报告
    report_path = os.path.join(save_dir, "training_report.txt")
    with open(report_path, 'w') as f:
        f.write("NBV Policy Training Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Configuration:\n")
        f.write("-" * 20 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("Training Results:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Final Training Loss: {train_losses[-1]:.6f}\n")
        f.write(f"Final Validation Loss: {val_losses[-1]:.6f}\n")
        f.write(f"Best Validation Loss: {min(val_losses):.6f}\n")
        f.write(f"Best Epoch: {np.argmin(val_losses) + 1}\n")
        f.write(f"Total Epochs: {len(train_losses)}\n")
        
        # 计算改进率
        initial_loss = train_losses[0]
        final_loss = train_losses[-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100
        f.write(f"Training Loss Improvement: {improvement:.2f}%\n")
    
    logger.info("Training report saved to {}", save_dir)


def plot_loss_components(loss_history: Dict[str, List[float]],
                        save_path: Optional[str] = None):
    """
    绘制损失组件
    
    Args:
        loss_history: 损失历史字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (loss_name, loss_values) in enumerate(loss_history.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        epochs = range(1, len(loss_values) + 1)
        color = colors[i % len(colors)]
        
        ax.plot(epochs, loss_values, color=color, linewidth=2, label=loss_name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{loss_name.replace("_", " ").title()} Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 隐藏未使用的子图
    for i in range(len(loss_history), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Loss components plot saved to {}", save_path)
    
    plt.show()
