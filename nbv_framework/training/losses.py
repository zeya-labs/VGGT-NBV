"""
损失函数模块

实现用于评估重建质量的各种损失函数，包括Chamfer距离、几何损失等。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Literal, TYPE_CHECKING
import numpy as np
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes, Pointclouds
import logging

if TYPE_CHECKING:
    from ..rendering import DifferentiableRenderer


class ChamferDistance(nn.Module):
    """
    一个计算对齐后倒角距离的模块，提供了多种鲁棒的归一化方法来对抗离群点。
    """
    def __init__(self, 
                 normalization_method: str = 'quantile'):
        """
        初始化模块。
        
        Args:
            normalization_method (str): 使用的归一化方法。
                - 'max': 按最大距离归一化 (对离群点敏感)。
                - 'std': 按标准差/RMS距离归一化 (鲁棒)。
                - 'quantile': 按百分位数归一化 (最鲁棒)。
        """
        super().__init__()
        if normalization_method not in ['max', 'std', 'quantile']:
            raise ValueError(f"未知的归一化方法: {normalization_method}")
        self.normalization_method = normalization_method

    def _umeyama_alignment(self, source: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """估计从 source 到 target 的相似变换。"""
        device = source.device
        dtype = source.dtype

        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source and target must be rank-2 tensors shaped [N, 3].")
        if source.shape[0] != target.shape[0]:
            raise ValueError(
                f"source and target must contain the same number of points; got {source.shape[0]} and {target.shape[0]}."
            )

        n_points = source.shape[0]

        if n_points < 3 or target.shape[0] < 3:
            rotation = torch.eye(3, device=device, dtype=dtype)
            translation = torch.zeros(3, device=device, dtype=dtype)
            scale = torch.tensor(1.0, device=device, dtype=dtype)
            return scale, rotation, translation

        # 使用双精度进行SVD以提升稳定性，但保持结果在输入dtype
        source64 = source.to(dtype=torch.float64)
        target64 = target.to(dtype=torch.float64)

        mu_x = source64.mean(dim=0)
        mu_y = target64.mean(dim=0)
        X = source64 - mu_x
        Y = target64 - mu_y

        # Umeyama 协方差矩阵定义为 Y^T @ X / N，其中 Y 为 target
        cov = (Y.T @ X) / n_points

        U, S, Vh = torch.linalg.svd(cov)

        d = torch.ones(3, device=device, dtype=torch.float64)
        if torch.det(U @ Vh) < 0:
            d[-1] = -1

        D = torch.diag(d)
        rotation = U @ D @ Vh

        var_x = torch.clamp((X ** 2).sum() / n_points, min=1e-8)
        scale = torch.sum(S * d) / var_x

        translation = mu_y - scale * (rotation @ mu_x)

        return (scale.to(dtype=dtype),
                rotation.to(dtype=dtype),
                translation.to(dtype=dtype))

    @staticmethod
    def _apply_similarity_transform(points: torch.Tensor,
                                     scale: torch.Tensor,
                                     rotation: torch.Tensor,
                                     translation: torch.Tensor) -> torch.Tensor:
        if points.numel() == 0:
            return points
        return scale * (points @ rotation.transpose(0, 1)) + translation

    def forward(self,
                p_pred: Pointclouds,
                p_gt: Pointclouds,
                correspondence_points: Optional[List[torch.Tensor]] = None,
                writer=None,
                step=None) -> torch.Tensor:
        """
        计算对齐后的倒角距离。

        Args:
            p_pred: 预测点云。
            p_gt: 真实点云。
            correspondence_points: 与预测点一一对应的GT点列表。
        """
        if correspondence_points is None:
            raise ValueError("correspondence_points must be provided for Umeyama alignment.")

        pred_list = [p.to(dtype=torch.float32) for p in p_pred.points_list()]
        gt_list = [p.to(dtype=torch.float32) for p in p_gt.points_list()]
        corr_list = [cp.to(dtype=torch.float32) for cp in correspondence_points]

        aligned_points_list: List[torch.Tensor] = []

        for pred_points, corr_points in zip(pred_list, corr_list):
            with torch.autocast(device_type=pred_points.device.type, enabled=False):
                pred_points_f32 = pred_points.float()
                corr_points_f32 = corr_points.float()

                if corr_points_f32.numel() >= 3 and pred_points_f32.numel() >= 3:
                    scale, rotation, translation = self._umeyama_alignment(pred_points_f32, corr_points_f32)
                    aligned = self._apply_similarity_transform(pred_points_f32, scale, rotation, translation)
                else:
                    aligned = pred_points_f32
            aligned_points_list.append(aligned)

        p_pred_aligned = Pointclouds(points=aligned_points_list)

        # GT 点云只需转为32位即可
        p_gt_float = Pointclouds(points=gt_list)

        if writer is not None and step is not None:
            if len(aligned_points_list) > 0 and len(gt_list) > 0:
                correspondence_cloud = Pointclouds(points=[corr_list[0]]) if len(corr_list) > 0 else None
                self._visualize_point_clouds(
                    writer,
                    step,
                    Pointclouds(points=[pred_list[0]]),
                    Pointclouds(points=[gt_list[0]]),
                    Pointclouds(points=[aligned_points_list[0]]),
                    correspondence_cloud
                )

        loss, _ = chamfer_distance(p_pred_aligned, p_gt_float)

        return loss

    def _visualize_point_clouds(self,
                                writer,
                                step,
                                p_pred_norm,
                                p_gt_norm,
                                p_pred_aligned,
                                p_corr_subset: Optional[Pointclouds] = None):
        """
        使用TensorBoard记录点云以进行可视化

        Args:
            writer: TensorBoard SummaryWriter
            step: 当前训练步数
            p_pred_norm: 预测点云
            p_gt_norm: 真实点云
            p_pred_aligned: 对齐后的预测点云
            p_corr_subset: 用于对齐的对应点子集
            
        Returns:
            None

        颜色说明：
            - 蓝色：预测点云
            - 绿色：真实点云
            - 红色：对齐后的预测点云
            - 黄色：Umeyama 对齐使用的对应点
        """
        if writer is None or step is None:
            return
        
        # 确保点云在CPU上并且是numpy数组
        p_pred_norm_np = p_pred_norm.points_list()[0].detach().cpu().numpy()
        p_gt_norm_np = p_gt_norm.points_list()[0].detach().cpu().numpy()
        p_pred_aligned_np = p_pred_aligned.points_list()[0].detach().cpu().numpy()

        clouds_with_colors = [
            (p_pred_norm_np, np.array([0, 0, 255], dtype=np.uint8)),
            (p_gt_norm_np, np.array([0, 255, 0], dtype=np.uint8)),
            (p_pred_aligned_np, np.array([255, 0, 0], dtype=np.uint8)),
        ]

        if p_corr_subset is not None and len(p_corr_subset.points_list()) > 0:
            corr_np = p_corr_subset.points_list()[0].detach().cpu().numpy()
            if corr_np.size > 0:
                clouds_with_colors.append((corr_np, np.array([255, 255, 0], dtype=np.uint8)))

        vertices_list = []
        colors_list = []
        for points_np, color in clouds_with_colors:
            if points_np.ndim != 2 or points_np.shape[1] != 3:
                continue
            if points_np.shape[0] == 0:
                continue
            vertices_list.append(points_np)
            colors_list.append(np.repeat(color[np.newaxis, :], points_np.shape[0], axis=0))

        if not vertices_list:
            return

        # 合并点云和颜色
        combined_vertices = np.vstack(vertices_list)
        combined_colors = np.vstack(colors_list)

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
        # print("variance:",variance)
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
        # print(f"black_penalty: {black_penalty}")
        # print(f"variance_penalty: {variance_penalty}")
        # print(f"edge_penalty: {edge_penalty}")
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
                 viewpoint_weight: float = 0.0,
                 renderer: Optional["DifferentiableRenderer"] = None,
                 gt_lighting_type: str = "ambient"):
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
        self.renderer = renderer
        self.gt_lighting_type = gt_lighting_type
        self.train_flag = None
        
        self.chamfer_loss = ChamferDistance()
        self.viewpoint_loss = ViewpointLoss()

    def _render_gt_point_maps(
        self,
        mesh_batch: Meshes,
        camera_poses: torch.Tensor,
        writer=None,
        step: Optional[int] = None,
        log_prefix: str = "GTPointMaps"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用可微分渲染器生成GT点映射及有效掩码，并可选记录可视化。"""
        if self.renderer is None:
            raise RuntimeError("Renderer is required to derive ground-truth point correspondences.")

        if camera_poses is None:
            raise ValueError("camera_poses must be provided when computing Chamfer loss.")

        batch_size = len(mesh_batch)
        if camera_poses.dim() == 2:
            camera_poses = camera_poses.unsqueeze(1)

        point_maps_list: List[torch.Tensor] = []
        valid_masks_list: List[torch.Tensor] = []

        mesh_device = self.renderer.device
        with torch.no_grad():
            for i in range(batch_size):
                poses_i = camera_poses[i]
                if poses_i.numel() == 0:
                    raise ValueError("camera_poses contains empty view set, cannot compute correspondences.")

                poses_i = poses_i.to(mesh_device).float()
                mesh_i = mesh_batch[i].to(mesh_device)
                mesh_i = mesh_i.extend(poses_i.shape[0])
                render_out = self.renderer(
                    gt_mesh=mesh_i,
                    camera_poses=poses_i,
                    lighting_type=self.gt_lighting_type,
                    return_point_maps=True
                )

                if not isinstance(render_out, tuple) or len(render_out) != 3:
                    raise RuntimeError("Renderer did not return point maps as expected.")

                _, point_maps, valid_masks = render_out

                if writer is not None and step is not None and self.train_flag:
                    self._log_gt_point_map_tensors(
                        writer=writer,
                        step=step,
                        batch_index=i,
                        point_maps=point_maps,
                        valid_masks=valid_masks,
                        prefix=log_prefix
                    )

                point_maps = point_maps.permute(0, 2, 3, 1).contiguous()  # [S, H, W, 3]
                valid_masks = valid_masks.squeeze(1).contiguous()         # [S, H, W]

                point_maps_list.append(point_maps)
                valid_masks_list.append(valid_masks)

        point_maps_batch = torch.stack(point_maps_list, dim=0)  # [B, S, H, W, 3]
        valid_masks_batch = torch.stack(valid_masks_list, dim=0)  # [B, S, H, W]

        return point_maps_batch, valid_masks_batch

    @staticmethod
    def _log_gt_point_map_tensors(
        writer,
        step: int,
        batch_index: int,
        point_maps: torch.Tensor,
        valid_masks: torch.Tensor,
        prefix: str
    ) -> None:
        """将GT点映射与有效掩码逐视角写入TensorBoard便于排查。"""
        if point_maps.ndim != 4 or valid_masks.ndim != 4:
            return

        point_maps_cpu = point_maps.detach().float().cpu()  # [S, 3, H, W]
        valid_masks_cpu = valid_masks.detach().float().cpu()  # [S, 1, H, W]

        num_views = point_maps_cpu.shape[0]

        for view_idx in range(num_views):
            pm = point_maps_cpu[view_idx]# [3, H, W]
            mask = valid_masks_cpu[view_idx]

            pm_flat = pm.view(3, -1) 
            coord_min = pm_flat.min(dim=1).values.view(3, 1, 1)
            coord_max = pm_flat.max(dim=1).values.view(3, 1, 1)
            denom = (coord_max - coord_min).clamp_min(1e-6)
            pm_norm = (pm - coord_min) / denom

            writer.add_image(
                f"{prefix}/point_map_batch{batch_index}_view{view_idx}",
                pm_norm,
                global_step=step
            )

            writer.add_image(
                f"{prefix}/valid_mask_batch{batch_index}_view{view_idx}",
                mask,
                global_step=step
            )

    def extract_point_cloud_from_reconstruction(
        self,
        recon_data: Dict[str, torch.Tensor],
        combined_images_batch: torch.Tensor,
        confidence_threshold: float = 50.0,
        source: Literal['vggt', 'depth'] = 'depth',
        gt_valid_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[Pointclouds, torch.Tensor]:
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

            if gt_valid_masks is not None:
                # 确保形状匹配 [B, S, H, W]
                if gt_valid_masks.shape != combined_mask.shape:
                    raise ValueError(
                        f"gt_valid_masks shape {gt_valid_masks.shape} does not match combined mask shape {combined_mask.shape}"
                    )
                combined_mask = combined_mask & gt_valid_masks

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
                
        return Pointclouds(points=point_clouds_list), combined_mask
    
    def forward(self, 
               recon_data: Dict[str, torch.Tensor],
               gt_data: Dict[str, torch.Tensor],
               combined_images_batch: Optional[torch.Tensor],
               combined_camera_poses: Optional[torch.Tensor],
               return_components: bool = False,
               writer=None,
               step=None,
               train_flag: bool = False) -> torch.Tensor:
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
        self.train_flag = train_flag

        device = next(iter(recon_data.values())).device if recon_data else "cpu"
        total_loss = torch.tensor(0.0, device=device)
        loss_components = {}

        # Chamfer距离损失
        chamfer_loss_value = torch.tensor(0.0, device=device)
        if self.chamfer_weight > 0 and "gt_points" in gt_data:
            if combined_camera_poses is None:
                raise ValueError("combined_camera_poses must be provided when Chamfer loss is enabled.")

            normalized_mesh = gt_data.get('normalized_mesh')
            if normalized_mesh is None:
                raise KeyError("gt_mesh_data must contain 'normalized_mesh' for Chamfer loss computation.")

            gt_point_maps, gt_valid_masks = self._render_gt_point_maps(
                normalized_mesh,
                combined_camera_poses,
                writer=writer,
                step=step
            )

            sample_tensor = None
            for value in recon_data.values():
                if isinstance(value, torch.Tensor):
                    sample_tensor = value
                    break
            if sample_tensor is None:
                raise ValueError("recon_data must contain tensor values for device inference.")

            target_device = sample_tensor.device
            gt_point_maps = gt_point_maps.to(device=target_device, dtype=torch.float32)
            gt_valid_masks = gt_valid_masks.to(device=target_device)

            pred_pointclouds, correspondence_mask = self.extract_point_cloud_from_reconstruction(
                recon_data,
                combined_images_batch,
                source='vggt',
                gt_valid_masks=gt_valid_masks
            )
            # pred_pointclouds = self.extract_point_cloud_from_reconstruction(recon_data, combined_images_batch, source='vggt')

            gt_points_batch = gt_data["gt_points"]
            gt_pointclouds = Pointclouds(points=[p for p in gt_points_batch])

            correspondence_points: List[torch.Tensor] = []
            for i in range(correspondence_mask.shape[0]):
                mask_i = correspondence_mask[i]
                if mask_i.any():
                    gt_points_i = gt_point_maps[i][mask_i]
                else:
                    gt_points_i = torch.empty((0, 3), device=gt_point_maps.device, dtype=gt_point_maps.dtype)
                correspondence_points.append(gt_points_i)

            if len(pred_pointclouds) != len(gt_pointclouds):
                logging.warning("预测点云列表的批次大小与GT点云不匹配。跳过Chamfer损失计算。")
            else:
                chamfer_loss_value = self.chamfer_loss(
                    pred_pointclouds,
                    gt_pointclouds,
                    correspondence_points=correspondence_points,
                    writer=writer,
                    step=step
                )
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
            # print("viewpoint_loss_value:",viewpoint_loss_value)
            total_loss += self.viewpoint_weight * viewpoint_loss_value
        
        loss_components['viewpoint_loss'] = viewpoint_loss_value.item()
        loss_components['weighted_viewpoint_loss'] = (self.viewpoint_weight * viewpoint_loss_value).item()
        loss_components['total_loss'] = total_loss.item()
        
        if return_components:
            return total_loss, loss_components
        return total_loss
