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
from pytorch3d.structures import Meshes
import cv2

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

    def _normalize_point_cloud(self, p_cloud: torch.Tensor) -> torch.Tensor:
        """
        根据选定的方法，将一批点云归一化。
        【已修改】确保在计算前将点云转换为浮点类型。
        """
        p_cloud_float = p_cloud.float()
        
        # 1. 平移到原点 (在浮点张量上操作)
        centroid = torch.mean(p_cloud_float, dim=1, keepdim=True)
        p_cloud_centered = p_cloud_float - centroid
        
        # 2. 根据选定的方法计算缩放因子并缩放 (在浮点张量上操作)
        if self.normalization_method == 'max':
            distances = torch.norm(p_cloud_centered, p=2, dim=2, keepdim=True)
            scale = torch.max(distances, dim=1, keepdim=True)[0]
        elif self.normalization_method == 'std':
            distances = torch.norm(p_cloud_centered, p=2, dim=2)
            scale = torch.sqrt(torch.mean(distances**2, dim=1, keepdim=True)).unsqueeze(-1)
        elif self.normalization_method == 'quantile':
            # 现在 distances 是浮点类型，quantile 可以正常工作
            distances = torch.norm(p_cloud_centered, p=2, dim=2)
            scale = torch.quantile(distances, q=0.95, dim=1, keepdim=True).unsqueeze(-1)
            
        p_cloud_normalized = p_cloud_centered / (scale + 1e-8)
        
        return p_cloud_normalized

    def forward(self, p_pred: torch.Tensor, p_gt: torch.Tensor, 
                writer=None, step=None) -> torch.Tensor:
        """
        计算对齐后的倒角距离。
        
        Args:
            p_pred: 预测点云 [B, N, 3]
            p_gt: 真实点云 [B, M, 3]
            writer: TensorBoard SummaryWriter，可选
            step: 当前训练步数，可选
        """
        # 步骤 1: 归一化两个点云
        p_pred_norm = self._normalize_point_cloud(p_pred)
        p_gt_norm = self._normalize_point_cloud(p_gt)
        
        # 步骤 2: 使用ICP将预测点云对齐到GT点云
        with torch.no_grad():
            icp_result = iterative_closest_point(
                X=p_pred_norm, 
                Y=p_gt_norm, 
                max_iterations=self.icp_iterations
            )
        p_pred_aligned = icp_result.Xt
        
        # TensorBoard可视化点云
        if writer is not None and step is not None:
            self._visualize_point_clouds(writer, step, p_pred_norm, p_gt_norm, p_pred_aligned)
        
        # 步骤 3: 计算最终的倒角距离
        loss, _ = chamfer_distance(p_pred_aligned, p_gt_norm)
        
        return loss
    
    def _visualize_point_clouds(self, writer, step, p_pred_norm, p_gt_norm, p_pred_aligned):
        """
        在TensorBoard中将多个点云合并到一个视图中进行可视化。
        pred: 红, gt: 绿, aligned: 蓝
        """
        # 只可视化第一个batch
        if p_pred_norm.shape[0] == 0:
            return
        
        # 取第一个样本并分离计算图
        pred_points = p_pred_norm[0].detach()
        gt_points = p_gt_norm[0].detach()
        aligned_points = p_pred_aligned[0].detach()
        
        #输出shape
        print("pred_points shape:", pred_points.shape)
        print("gt_points shape:", gt_points.shape)
        print("aligned_points shape:", aligned_points.shape)

        # 可选：降采样
        def _maybe_subsample(points: torch.Tensor, max_points: int = 20000) -> torch.Tensor:
            if points.shape[0] <= max_points:
                return points
            idx = torch.randperm(points.shape[0])[:max_points]
            return points[idx]
        
        pred_points = _maybe_subsample(pred_points)
        gt_points = _maybe_subsample(gt_points)
        aligned_points = _maybe_subsample(aligned_points)
        
        # 为不同点云创建颜色
        pred_colors = torch.tensor([255, 0, 0], dtype=torch.uint8).expand_as(pred_points)
        gt_colors = torch.tensor([0, 255, 0], dtype=torch.uint8).expand_as(gt_points)
        aligned_colors = torch.tensor([0, 0, 255], dtype=torch.uint8).expand_as(aligned_points)
        
        # 合并顶点和颜色
        all_vertices = torch.cat([pred_points, gt_points, aligned_points], dim=0)
        all_colors = torch.cat([pred_colors, gt_colors, aligned_colors], dim=0)
        
        # 增加 batch 维度
        all_vertices = all_vertices.unsqueeze(0)  # Shape: [1, N_total, 3]
        all_colors = all_colors.unsqueeze(0)      # Shape: [1, N_total, 3]
        
        # 只调用一次 add_mesh
        writer.add_mesh('point_clouds/comparison', vertices=all_vertices, colors=all_colors, global_step=step)

class ViewpointLoss(nn.Module):
    """
    视角损失
    
    惩罚预测出黑屏或偏离物体的相机位置。
    """ 
    
    def __init__(self,
                 black_screen_threshold: float = 0.5,
                 low_variance_threshold: float = 0.01,
                 edge_density_threshold: float = 0.05):
        """
        初始化视角损失
        
        Args:
            black_screen_threshold: 黑色像素占比阈值，超过此值认为是黑屏
            low_variance_threshold: 低方差阈值，低于此值认为图像内容单调
            edge_density_threshold: 边缘密度阈值，低于此值认为图像缺乏细节
        """
        super().__init__()
        self.black_screen_threshold = black_screen_threshold
        self.low_variance_threshold = low_variance_threshold
        self.edge_density_threshold = edge_density_threshold
    
    def compute_black_screen_penalty(self, images: torch.Tensor) -> torch.Tensor:
        """
        计算黑屏惩罚
        
        Args:
            images: 渲染的图像 [B, 3, H, W]
            
        Returns:
            penalty: 黑屏惩罚值
        """
        # 转换为灰度图像
        gray_images = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]  # [B, H, W]
        
        # 计算黑色像素占比（像素值小于0.1认为是黑色）
        black_pixels = (gray_images < 0.1).float()
        black_ratio = black_pixels.mean(dim=[1, 2])  # [B]
        
        # 当黑色占比超过阈值时给予惩罚
        penalty = torch.where(
            black_ratio > self.black_screen_threshold,
            (black_ratio - self.black_screen_threshold) * 10.0,  # 线性惩罚
            torch.zeros_like(black_ratio)
        )
        
        return penalty.mean()
    
    def compute_low_variance_penalty(self, images: torch.Tensor) -> torch.Tensor:
        """
        计算低方差惩罚（图像内容单调）
        
        Args:
            images: 渲染的图像 [B, 3, H, W]
            
        Returns:
            penalty: 低方差惩罚值
        """
        # 转换为灰度图像
        gray_images = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]  # [B, H, W]
        
        # 计算每个图像的方差
        variance = torch.var(gray_images.view(gray_images.shape[0], -1), dim=1)  # [B]
        
        # 当方差低于阈值时给予惩罚
        penalty = torch.where(
            variance < self.low_variance_threshold,
            (self.low_variance_threshold - variance) * 5.0,  # 反比例惩罚
            torch.zeros_like(variance)
        )
        
        return penalty.mean()
    
    def compute_edge_density_penalty(self, images: torch.Tensor) -> torch.Tensor:
        """
        计算边缘密度惩罚（缺乏细节）
        
        Args:
            images: 渲染的图像 [B, 3, H, W]
            
        Returns:
            penalty: 边缘密度惩罚值
        """
        # 转换为灰度图像
        gray_images = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]  # [B, H, W]
        
        # 使用Sobel算子计算边缘
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=images.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=images.device).view(1, 1, 3, 3)
        
        # 添加padding并计算梯度
        gray_padded = F.pad(gray_images.unsqueeze(1), (1, 1, 1, 1), mode='reflect')
        grad_x = F.conv2d(gray_padded, sobel_x)
        grad_y = F.conv2d(gray_padded, sobel_y)
        
        # 计算梯度幅值
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # 计算边缘密度（强边缘像素占比）
        strong_edges = (edge_magnitude > 0.1).float()
        edge_density = strong_edges.mean(dim=[1, 2, 3])  # [B]
        
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
            total_penalty: 总惩罚值
        """
        black_penalty = self.compute_black_screen_penalty(images)
        variance_penalty = self.compute_low_variance_penalty(images)
        edge_penalty = self.compute_edge_density_penalty(images)
        
        total_penalty = black_penalty + variance_penalty + edge_penalty
        
        return total_penalty


class ReconstructionLoss(nn.Module):
    """
    综合重建损失
    
    结合多种几何损失来评估重建质量。
    """
    
    def __init__(self,
                 chamfer_weight: float = 1.0,
                 confidence_weight: float = 0.01,
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
        confidence_threshold: float = 0.5,
        source: Literal['vggt', 'depth'] = 'depth'
    ) -> List[torch.Tensor]:
        
        # print("Shape of combined_images_batch:", combined_images_batch.shape)
        """
        从重建数据中为批处理中的每个项目高效地提取高置信度点云。

        Args:
            recon_data: 包含重建结果的字典。
            combined_images_batch: 输入图像 [B, N+1, 3, H, W]。
            confidence_threshold: 用于筛选点的置信度阈值。
            source: 指定点云和置信度的数据源。
                    - 'vggt': 使用 'world_points' 和 'world_points_conf'。
                    - 'depth': 使用 'world_points_from_depth' 和 'depth_conf'。

        Returns:
            point_clouds_list: 一个列表，包含批处理中每个项目的点云。
                            列表长度为B，每个元素是形状为 [Ni, 3] 的张量，
                            其中 Ni 是第i个项目中通过阈值的点的数量。
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
            return []

        B, S, H, W, _ = points_data.shape
        # 2. 矢量化计算高置信度掩码 (对整个批次一次性计算)
        high_conf_mask = conf_data > confidence_threshold  # Shape: [B, S, H, W]
        
        # 3. 矢量化计算非黑色像素掩码
        if combined_images_batch is not None:
            # 计算所有像素的平均强度
            pixel_intensity = combined_images_batch.mean(dim=2)  # Shape: [B, S, H, W]
            
            # 定义黑色像素阈值
            black_threshold = 0.05
            non_black_mask = pixel_intensity > black_threshold  # Shape: [B, S, H, W]
            
            # 合并两个掩码
            # print("Shape of high_conf_mask:", high_conf_mask.shape)
            # print("Shape of non_black_mask:", non_black_mask.shape)
            combined_mask = high_conf_mask & non_black_mask
        else:
            # 如果没有提供图像，只使用置信度掩码
            combined_mask = high_conf_mask
        
        # 4. 应用掩码并生成结果列表
        # 尽管我们仍然需要一个循环来构建列表（因为每个元素的点数量不同），
        # 但所有昂贵的计算（掩码生成）都已在循环外完成。
        # 列表推导式是完成这个任务的简洁方式。
        point_clouds_list = [
            points_data[i][combined_mask[i]] for i in range(B)
        ]
                
        return point_clouds_list
    
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
        total_loss = torch.tensor(0.0, device=self._get_device(recon_data))
        loss_components = {}
        loss_count = 0

        # Chamfer距离损失
        chamfer_loss_value = torch.tensor(0.0, device=total_loss.device)
        if self.chamfer_weight > 0 and "gt_points" in gt_data:
            # 1. 提取预测点云列表，每个元素是 [Ni, 3]
            pred_points_list = self.extract_point_cloud_from_reconstruction(recon_data, combined_images_batch, source='depth')
            
            # 2. 获取真实的GT点云批处理张量，假设形状为 [B, M, 3]
            gt_points_batch = gt_data["gt_points"]
            
            # 3. 检查批处理大小是否匹配
            if len(pred_points_list) != gt_points_batch.shape[0]:
                print("警告: 预测点云列表的批次大小与GT点云不匹配。跳过Chamfer损失计算。")
            else:
                batch_chamfer_loss = 0.0
                valid_items_count = 0
                
                # 4. 遍历批处理中的每个项目来计算损失
                for i in range(len(pred_points_list)):
                    pred_pc_item = pred_points_list[i]  # 当前预测点云, shape: [Ni, 3]
                    gt_pc_item = gt_points_batch[i]      # 当前GT点云, shape: [M, 3]
                    
                    # 5. 安全检查：如果过滤后没有剩下任何点，则跳过此项
                    if pred_pc_item.shape[0] == 0:
                        continue
                    
                    # 6. 为损失函数准备输入：为当前项增加一个批次维度
                    #    pred_pc_item -> [1, Ni, 3]
                    #    gt_pc_item   -> [1, M, 3]
                    pred_pc_item_batched = pred_pc_item.unsqueeze(0)
                    gt_pc_item_batched = gt_pc_item.unsqueeze(0)

                    # 7. 计算当前项的Chamfer损失
                    # 只在第一个有效项目上进行可视化，避免过多的TensorBoard记录
                    if valid_items_count == 0 and writer is not None and step is not None:
                        item_loss = self.chamfer_loss(pred_pc_item_batched, gt_pc_item_batched, writer, step)
                    else:
                        item_loss = self.chamfer_loss(pred_pc_item_batched, gt_pc_item_batched)
                    
                    # 8. 累加损失并计数有效项目
                    batch_chamfer_loss += item_loss
                    valid_items_count += 1
                    
                # 9. 如果批处理中至少有一个有效项目，则计算平均损失并加到总损失中
                if valid_items_count > 0:
                    mean_chamfer_loss = batch_chamfer_loss / valid_items_count
                    chamfer_loss_value = mean_chamfer_loss
                    total_loss += self.chamfer_weight * mean_chamfer_loss
                    loss_count += 1
        
        loss_components['chamfer_loss'] = chamfer_loss_value.item()
        loss_components['weighted_chamfer_loss'] = (self.chamfer_weight * chamfer_loss_value).item()
        
        # 置信度正则化
        conf_loss_value = torch.tensor(0.0, device=total_loss.device)
        if self.confidence_weight > 0:
            world_points_conf = recon_data.get("world_points_conf")
            depth_conf = recon_data.get("depth_conf")
            
            conf_loss = torch.tensor(0.0, device=total_loss.device)
            
            if world_points_conf is not None:
                # 鼓励高置信度预测
                conf_loss += -torch.log(world_points_conf.mean() + 1e-8)
            
            if depth_conf is not None:
                conf_loss += -torch.log(depth_conf.mean() + 1e-8)
            
            conf_loss_value = conf_loss
            total_loss += self.confidence_weight * conf_loss
            loss_count += 1
        
        loss_components['confidence_loss'] = conf_loss_value.item()
        loss_components['weighted_confidence_loss'] = (self.confidence_weight * conf_loss_value).item()
        
        # 视角损失（惩罚黑屏和低质量视角）
        new_images = combined_images_batch[:, -1, :, :, :]
        viewpoint_loss_value = torch.tensor(0.0, device=total_loss.device)
        if self.viewpoint_weight > 0 and new_images is not None:
            viewpoint_penalty = self.viewpoint_loss(new_images)
            viewpoint_loss_value = viewpoint_penalty
            total_loss += self.viewpoint_weight * viewpoint_penalty
            loss_count += 1
        
        loss_components['viewpoint_loss'] = viewpoint_loss_value.item()
        loss_components['weighted_viewpoint_loss'] = (self.viewpoint_weight * viewpoint_loss_value).item()
        loss_components['total_loss'] = total_loss.item()
        
        if return_components:
            return total_loss, loss_components
        return total_loss
    
    def _get_device(self, data_dict: Dict[str, torch.Tensor]) -> str:
        """获取数据的设备"""
        for v in data_dict.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return "cpu"