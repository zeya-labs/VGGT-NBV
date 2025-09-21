"""
损失函数模块

实现用于评估重建质量的各种损失函数，包括Chamfer距离、几何损失等。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Literal
import numpy as np
from pytorch3d.ops import iterative_closest_point
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes, Pointclouds
import logging


class ChamferDistance(nn.Module):
    """
    一个计算对齐后倒角距离的模块，提供了多种鲁棒的归一化方法来对抗离群点。
    """
    def __init__(self, 
                 icp_iterations: int = 100,
                 normalization_method: str = 'quantile'):
        """
        初始化模块。
        
        Args:
            icp_iterations (int): ICP算法的迭代次数。
            normalization_method (str): 使用的归一化方法。
                - 'max': 按最大距离归一化 (对离群点敏感)。
                - 'std': 按标准差/RMS距离归一化 (鲁棒)。
                - 'quantile': 按百分位数归一化 (最鲁棒)。
        """
        super().__init__()
        if normalization_method not in ['max', 'std', 'quantile']:
            raise ValueError(f"未知的归一化方法: {normalization_method}")
            
        self.icp_iterations = icp_iterations
        self.normalization_method = normalization_method

    def _normalize_point_clouds(self, p_clouds: Pointclouds) -> Pointclouds:
        """
        根据选定的方法，将一批点云归一化。
        """
        points_list = p_clouds.points_list()
        if not points_list:
            return p_clouds

        normalized_points_list = []
        for p_cloud in points_list:
            if p_cloud.shape[0] == 0:
                normalized_points_list.append(p_cloud)
                continue

            p_cloud_float = p_cloud.float()
            
            # 1. 平移到原点
            centroid = torch.mean(p_cloud_float, dim=0, keepdim=True)
            p_cloud_centered = p_cloud_float - centroid
            
            # 2. 根据选定的方法计算缩放因子并缩放
            distances = torch.norm(p_cloud_centered, p=2, dim=1)
            if distances.shape[0] == 0:
                scale = torch.tensor(1.0, device=distances.device)
            elif self.normalization_method == 'max':
                scale = torch.max(distances)
            elif self.normalization_method == 'std':
                scale = torch.sqrt(torch.mean(distances**2))
            elif self.normalization_method == 'quantile':
                scale = torch.quantile(distances, q=0.95)
            
            p_cloud_normalized = p_cloud_centered / (scale + 1e-8)
            normalized_points_list.append(p_cloud_normalized)
        
        return Pointclouds(points=normalized_points_list)

    def forward(self, p_pred: Pointclouds, p_gt: Pointclouds, writer=None, step=None) -> torch.Tensor:
        """
        计算对齐后的倒角距离。
        
        Args:
            p_pred: 预测点云 (Pointclouds object)
            p_gt: 真实点云 (Pointclouds object)
            writer: TensorBoard SummaryWriter for visualization.
            step: Current training step.
        """
        # 步骤 1: 归一化两个点云
        p_pred_norm = self._normalize_point_clouds(p_pred)
        p_gt_norm = self._normalize_point_clouds(p_gt)
        
        # 步骤 2: 使用ICP将预测点云对齐到GT点云
        # ICP在pytorch3d中不支持不同点数的批处理，所以我们仍然需要一个循环。
        aligned_points_list = []
        with torch.no_grad():
            with torch.autocast(device_type=p_pred.device.type, enabled=False):
                X_list = [p.to(dtype=torch.float32) for p in p_pred_norm.points_list()]
                Y_list = [p.to(dtype=torch.float32) for p in p_gt_norm.points_list()]

                for X, Y in zip(X_list, Y_list):
                    if X.shape[0] == 0 or Y.shape[0] == 0:
                        aligned_points_list.append(X) # Append original (empty) points
                        continue
                    
                    icp_result = iterative_closest_point(
                        X=X.unsqueeze(0),
                        Y=Y.unsqueeze(0),
                        max_iterations=self.icp_iterations
                    )
                    aligned_points_list.append(icp_result.Xt.squeeze(0))

        p_pred_aligned = Pointclouds(points=aligned_points_list)

        # 可视化
        if writer is not None and step is not None:
            # We need to be careful about batching here.
            # Let's visualize the first element of the batch.
            if len(p_pred_norm.points_list()) > 0 and len(p_gt_norm.points_list()) > 0 and len(p_pred_aligned.points_list()) > 0:
                self._visualize_point_clouds(writer, step,
                                             Pointclouds(points=[p_pred_norm.points_list()[0]]),
                                             Pointclouds(points=[p_gt_norm.points_list()[0]]),
                                             Pointclouds(points=[p_pred_aligned.points_list()[0]]))
        
        # 步骤 3: 计算最终的倒角距离 (batched)
        # chamfer_distance可以处理空的点云
        loss, _ = chamfer_distance(p_pred_aligned, p_gt_norm)
        
        return loss

    def _visualize_point_clouds(self, writer, step, p_pred_norm, p_gt_norm, p_pred_aligned):
        """
        使用TensorBoard记录点云以进行可视化

        Args:
            writer: TensorBoard SummaryWriter
            step: 当前训练步数
            p_pred_norm: 预测点云
            p_gt_norm: 真实点云
            p_pred_aligned: 对齐后的预测点云
            
        Returns:
            None

        颜色说明：
            - 蓝色：预测点云
            - 绿色：真实点云
            - 红色：对齐后的预测点云
        """
        if writer is None or step is None:
            return
        
        # 确保点云在CPU上并且是numpy数组
        p_pred_norm_np = p_pred_norm.points_list()[0].detach().cpu().numpy()
        p_gt_norm_np = p_gt_norm.points_list()[0].detach().cpu().numpy()
        p_pred_aligned_np = p_pred_aligned.points_list()[0].detach().cpu().numpy()

        # 为每个点云分配颜色
        pred_colors = np.array([[0, 0, 255]] * p_pred_norm_np.shape[0], dtype=np.uint8)
        gt_colors = np.array([[0, 255, 0]] * p_gt_norm_np.shape[0], dtype=np.uint8)
        aligned_colors = np.array([[255, 0, 0]] * p_pred_aligned_np.shape[0], dtype=np.uint8)

        # 合并点云和颜色
        combined_vertices = np.vstack([p_pred_norm_np, p_gt_norm_np, p_pred_aligned_np])
        combined_colors = np.vstack([pred_colors, gt_colors, aligned_colors])

        # 添加到TensorBoard
        writer.add_mesh("Chamfer/Comparison", 
                        vertices=combined_vertices[np.newaxis, ...], 
                        colors=combined_colors[np.newaxis, ...], 
                        global_step=step)

class ViewpointLoss(nn.Module):
    """
    视角损失
    
    惩罚预测出黑屏、内容单调或缺乏细节的相机视角。
    """ 
    
    def __init__(self,
                 black_screen_threshold: float = 0.5,
                 low_variance_threshold: float = 0.05,
                 edge_density_threshold: float = 0.05):
        """
        初始化视角损失
        
        Args:
            black_screen_threshold: 黑色像素占比阈值，超过此值认为是黑屏。
            low_variance_threshold: 低方差阈值，低于此值认为图像内容单调。
            edge_density_threshold: 边缘密度阈值，低于此值认为图像缺乏细节。
        """
        super().__init__()
        self.black_screen_threshold = black_screen_threshold
        self.low_variance_threshold = low_variance_threshold
        self.edge_density_threshold = edge_density_threshold
        
        # 将 Sobel 算子注册为 buffer，而不是每次都重新创建。
        # 这样它们只会被创建一次，并会自动跟随模型移动到正确的设备。
        sobel_x_kernel = torch.tensor(
            [[-1, 0, 1], 
             [-2, 0, 2], 
             [-1, 0, 1]], 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        
        sobel_y_kernel = torch.tensor(
            [[-1, -2, -1], 
             [0, 0, 0], 
             [1, 2, 1]], 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x_kernel)
        self.register_buffer('sobel_y', sobel_y_kernel)
    
    def compute_black_screen_penalty(self, images: torch.Tensor) -> torch.Tensor:
        """
        计算黑屏惩罚
        
        Args:
            images: 渲染的图像 [B, 3, H, W]
            
        Returns:
            penalty: 黑屏惩罚值 (标量)
        """
        # 转换为灰度图像 [B, H, W]
        gray_images = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        
        # 计算黑色像素占比（像素值小于0.1认为是黑色）
        black_pixels = (gray_images < 0.1).float()
        black_ratio = black_pixels.mean(dim=[1, 2])  # Shape: [B]
        
        # 当黑色占比超过阈值时给予线性惩罚
        penalty = F.relu(black_ratio - self.black_screen_threshold)
        
        return penalty.mean()
    
    def compute_low_variance_penalty(self, images: torch.Tensor) -> torch.Tensor:
        """
        计算低方差惩罚（图像内容单调）
        
        Args:
            images: 渲染的图像 [B, 3, H, W]
            
        Returns:
            penalty: 低方差惩罚值 (标量)
        """
        # 转换为灰度图像 [B, H, W]
        gray_images = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        
        # 计算每个图像的方差 [B]
        variance = torch.var(gray_images.view(gray_images.shape[0], -1), dim=1)
        print("variance:",variance)
        # 当方差低于阈值时给予反比例惩罚
        penalty = F.relu(self.low_variance_threshold - variance) * 10.0
        
        return penalty.mean()
    
    def compute_edge_density_penalty(self, images: torch.Tensor) -> torch.Tensor:
        """
        计算边缘密度惩罚（缺乏细节）
        
        Args:
            images: 渲染的图像 [B, 3, H, W]
            
        Returns:
            penalty: 边缘密度惩罚值 (标量)
        """
        # 转换为灰度图像 [B, 1, H, W] for conv2d
        gray_images = (0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]).unsqueeze(1)
        
        # 使用预先注册的 Sobel 算子计算梯度
        # F.conv2d 需要 [B, C_in, H, W] 格式的输入
        sobel_x = self.sobel_x.to(device=gray_images.device, dtype=gray_images.dtype)
        sobel_y = self.sobel_y.to(device=gray_images.device, dtype=gray_images.dtype)
        grad_x = F.conv2d(gray_images, sobel_x, padding='same')
        grad_y = F.conv2d(gray_images, sobel_y, padding='same')
        
        # 计算梯度幅值
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # 计算边缘密度（强边缘像素占比）
        strong_edges = (edge_magnitude > 0.1).float()
        edge_density = strong_edges.mean(dim=[1, 2, 3])  # Shape: [B]
        
        # 当边缘密度低于阈值时给予惩罚
        penalty = torch.where(
            edge_density < self.edge_density_threshold,
            (self.edge_density_threshold - edge_density) * 3.0,  # 反比例惩罚
            torch.zeros_like(edge_density)
        )
        
        return penalty.mean()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        计算总的视角损失
        
        Args:
            images: 渲染的图像 [B, 3, H, W]
            
        Returns:
            total_penalty: 总惩罚值 (标量)
        """
        black_penalty = self.compute_black_screen_penalty(images)
        variance_penalty = self.compute_low_variance_penalty(images)
        edge_penalty = self.compute_edge_density_penalty(images)
        #输出调试
        print(f"black_penalty: {black_penalty}")
        print(f"variance_penalty: {variance_penalty}")
        print(f"edge_penalty: {edge_penalty}")
        total_penalty = black_penalty + variance_penalty + edge_penalty
        
        return total_penalty

class ReconstructionLoss(nn.Module):
    """
    综合重建损失
    
    结合多种几何损失来评估重建质量。
    """
    
    def __init__(self,
                 chamfer_weight: float = 1.0,
                 confidence_weight: float = 0.0,
                 viewpoint_weight: float = 0.1):
        """
        初始化重建损失
        
        Args:
            chamfer_weight: Chamfer距离权重
            confidence_weight: 置信度损失权重
            viewpoint_weight: 视角损失权重
        """
        super().__init__()
        
        self.chamfer_weight = chamfer_weight
        self.confidence_weight = confidence_weight
        self.viewpoint_weight = viewpoint_weight
        
        self.chamfer_loss = ChamferDistance()
        self.viewpoint_loss = ViewpointLoss()
    
    def extract_point_cloud_from_reconstruction(
        self,
        recon_data: Dict[str, torch.Tensor],
        combined_images_batch: torch.Tensor,
        confidence_threshold: float = 50.0,
        source: Literal['vggt', 'depth'] = 'depth'
    ) -> Pointclouds:
        """
        从重建数据中为批处理中的每个项目高效地提取高置信度点云。

        Args:
            recon_data: 包含重建结果的字典。
            combined_images_batch: 输入图像 [B, N+1, 3, H, W]。
            confidence_threshold: 置信度百分位数阈值 (0-100)，过滤掉置信度最低的百分比点。
            source: 指定点云和置信度的数据源。
                    - 'vggt': 使用 'world_points' 和 'world_points_conf'。
                    - 'depth': 使用 'world_points_from_depth' 和 'depth_conf'。

        Returns:
            point_clouds: 一个Pointclouds对象，包含批处理中每个项目的点云。
        """
        # 1. 根据 'source' 参数选择数据源
        if source == 'vggt':
            points_data = recon_data.get("world_points")       # Shape: [B, S, H, W, 3]
            conf_data = recon_data.get("world_points_conf")     # Shape: [B, S, H, W]
            if points_data is None or conf_data is None:
                raise KeyError("Source 'vggt' selected, but 'world_points' or 'world_points_conf' not found in recon_data.")
        elif source == 'depth':
            points_data = recon_data.get("world_points_from_depth")
            conf_data = recon_data.get("depth_conf")
            if points_data is None or conf_data is None:
                raise KeyError("Source 'depth' selected, but 'world_points_from_depth' or 'depth_conf' not found in recon_data.")
        else:
            raise ValueError(f"未知的 source: {source}。应为 'vggt' 或 'depth'。")

        # 检查输入是否有效，如果数据为空则直接返回
        if points_data is None or conf_data is None:
            return Pointclouds(points=[])

        B, S, H, W, _ = points_data.shape
        
        with torch.no_grad():  # 掩码计算不需要梯度
            # 2. 计算置信度百分位数阈值并生成掩码
            if confidence_threshold == 0.0:
                conf_threshold_value = 0.0
            else:
                # 将置信度数据展平并计算百分位数
                conf_flat = conf_data.reshape(-1)
                conf_threshold_value = torch.quantile(conf_flat, confidence_threshold / 100.0)
            
            # 生成高置信度掩码
            high_conf_mask = (conf_data >= conf_threshold_value) & (conf_data > 1e-5)  # Shape: [B, S, H, W]
            
            # 3. 矢量化计算非黑色像素掩码
            if combined_images_batch is not None:
                # 计算所有像素的平均强度
                pixel_intensity = combined_images_batch.mean(dim=2)  # Shape: [B, S, H, W]
                
                # 定义黑色像素阈值
                black_threshold = 0.05
                non_black_mask = pixel_intensity > black_threshold  # Shape: [B, S, H, W]
                
                # 合并两个掩码
                combined_mask = high_conf_mask & non_black_mask
            else:
                # 如果没有提供图像，只使用置信度掩码
                combined_mask = high_conf_mask
        
        # 4. 应用掩码并生成结果列表
        point_clouds_list = []
        for i in range(B):
            mask_i = combined_mask[i]  # Shape: [S, H, W]
            if mask_i.any():  # 只有当存在有效点时才进行提取
                points_i = points_data[i][mask_i]
                point_clouds_list.append(points_i)
            else:
                # 如果没有有效点，添加空张量
                point_clouds_list.append(torch.empty((0, 3), device=points_data.device, dtype=points_data.dtype))
                
        return Pointclouds(points=point_clouds_list)
    
    def forward(self, 
               recon_data: Dict[str, torch.Tensor],
               gt_data: Dict[str, torch.Tensor],
               combined_images_batch: Optional[torch.Tensor],
               return_components: bool = False,
               writer=None,
               step=None) -> torch.Tensor:
        """
        计算综合重建损失
        
        Args:
            recon_data: VGGT重建数据
            gt_data: 真实数据（可以是mesh、点云等）
            combined_images_batch: 输入图像 [B, N+1, 3, H, W]
            return_components: 是否返回损失组件详情
            writer: TensorBoard SummaryWriter，可选，用于点云可视化
            step: 当前训练步数，可选，用于TensorBoard记录
            
        Returns:
            total_loss: 总损失 或 (总损失, 损失组件字典)
        """
        device = next(iter(recon_data.values())).device if recon_data else "cpu"
        total_loss = torch.tensor(0.0, device=device)
        loss_components = {}

        # Chamfer距离损失
        chamfer_loss_value = torch.tensor(0.0, device=device)
        if self.chamfer_weight > 0 and "gt_points" in gt_data:
            pred_pointclouds = self.extract_point_cloud_from_reconstruction(recon_data, combined_images_batch, source='depth') # 224 效果不好
            # pred_pointclouds = self.extract_point_cloud_from_reconstruction(recon_data, combined_images_batch, source='vggt')

            gt_points_batch = gt_data["gt_points"]
            gt_pointclouds = Pointclouds(points=[p for p in gt_points_batch])

            if len(pred_pointclouds) != len(gt_pointclouds):
                logging.warning("预测点云列表的批次大小与GT点云不匹配。跳过Chamfer损失计算。")
            else:
                chamfer_loss_value = self.chamfer_loss(pred_pointclouds, gt_pointclouds, writer, step)
                total_loss += self.chamfer_weight * chamfer_loss_value
        
        loss_components['chamfer_loss'] = chamfer_loss_value.item()
        loss_components['weighted_chamfer_loss'] = (self.chamfer_weight * chamfer_loss_value).item()
        
        # 置信度正则化
        conf_loss_value = torch.tensor(0.0, device=device)
        if self.confidence_weight > 0:
            world_points_conf = recon_data.get("world_points_conf")
            depth_conf = recon_data.get("depth_conf")
            
            conf_loss = torch.tensor(0.0, device=device)
            
            if world_points_conf is not None:
                conf_loss += -torch.log(world_points_conf.mean() + 1e-8)
            
            if depth_conf is not None:
                conf_loss += -torch.log(depth_conf.mean() + 1e-8)
            
            conf_loss_value = conf_loss
            total_loss += self.confidence_weight * conf_loss
        
        loss_components['confidence_loss'] = conf_loss_value.item()
        loss_components['weighted_confidence_loss'] = (self.confidence_weight * conf_loss_value).item()
        
        # 视角损失（惩罚黑屏和低质量视角）
        viewpoint_loss_value = torch.tensor(0.0, device=device)
        if self.viewpoint_weight > 0 and combined_images_batch is not None:
            new_images = combined_images_batch[:, -1, :, :, :]
            viewpoint_loss_value = self.viewpoint_loss(new_images)
            print("viewpoint_loss_value:",viewpoint_loss_value)
            total_loss += self.viewpoint_weight * viewpoint_loss_value
        
        loss_components['viewpoint_loss'] = viewpoint_loss_value.item()
        loss_components['weighted_viewpoint_loss'] = (self.viewpoint_weight * viewpoint_loss_value).item()
        loss_components['total_loss'] = total_loss.item()
        
        if return_components:
            return total_loss, loss_components
        return total_loss