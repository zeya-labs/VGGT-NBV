"""
相机工具模块
处理相机位姿生成、保存和转换
"""

import json
import math
import numpy as np
import torch
from typing import Dict, List, Optional
from pytorch3d.renderer import look_at_view_transform

try:
    import matplotlib.cm as mpl_cm
except ImportError:  # pragma: no cover
    mpl_cm = None

from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

from .coordinate_utils import get_up_vector, generate_fibonacci_sphere_points, generate_fibonacci_upper_hemisphere_points



class CameraPoseGenerator:
    """相机位姿生成器"""
    
    def __init__(self, up_axis: str = "Y"):
        """
        初始化相机位姿生成器
        
        Args:
            up_axis: 上朝向轴
        """
        self.up_axis = up_axis
        self.up_vector = get_up_vector(up_axis)
    
    def _generate_poses_from_positions(
        self,
        sphere_positions: np.ndarray,
        seed: int = 0,
        base_radius: float = 2.5,
        radius_variation: float = 0,
        radii: Optional[np.ndarray] = None
    ) -> List[Dict[str, List[float]]]:
        """
        从球面位置生成相机位姿的通用方法
        
        Args:
            sphere_positions: 球面位置数组
            seed: 随机种子
            base_radius: 基础相机距离
            radius_variation: 距离变化范围
            radii: 可选的半径数组
        
        Returns:
            camera_poses: 相机位姿列表，每个元素包含position和quaternion
        """
        # 使用局部随机数生成器，避免影响全局随机状态
        rng = np.random.RandomState(seed)
        if radii is not None:
            if len(radii) != len(sphere_positions):
                raise ValueError("radii must have the same length as sphere_positions.")
            radii = np.asarray(radii, dtype=np.float32)
        
        poses = []
        
        for i, direction in enumerate(sphere_positions):
            if radii is not None:
                radius = float(radii[i])
            else:
                radius = base_radius + rng.uniform(-radius_variation, radius_variation)
            
            # 计算相机位置
            position = direction * radius
            
            # 使用PyTorch3D的look_at_view_transform
            # 将numpy数组转换为张量以避免性能警告
            eye = torch.tensor(position.reshape(1, 3), dtype=torch.float32)
            at = torch.tensor([[0, 0, 0]], dtype=torch.float32)  # 看向原点
            
            up = torch.tensor(self.up_vector.reshape(1, -1), dtype=torch.float32)  # 使用指定的up向量
            
            R, T = look_at_view_transform(eye=eye, at=at, up=up)
            
            # 将旋转矩阵转换为四元数
            quaternions_wxyz = matrix_to_quaternion(R)
            q = quaternions_wxyz[0]  # 获取批次中的第一个四元数
            quaternion_xyzw = [q[1].item(), q[2].item(), q[3].item(), q[0].item()]

            poses.append({
                "position": [float(position[0]), float(position[1]), float(position[2])],
                "quaternion": quaternion_xyzw
            })
        
        return poses
    
    def generate_camera_poses(
        self, 
        num_views: int, 
        seed: int = 0,
        base_radius: float = 2.2, # 0.9, 0.8是2.5，0.7是2.86
        radius_variation: float = 0,
        hemisphere: str = 'full',
        radius_mode: str = 'random',
        radius_layers: int = 1
    ) -> List[Dict[str, List[float]]]:
        """
        生成相机位姿
        
        Args:
            num_views: 视图数量
            seed: 随机种子
            base_radius: 基础相机距离
            radius_variation: 距离变化范围
            hemisphere: 球面类型，'full'表示全球面，'upper'表示上半球面
            
        Returns:
            camera_poses: 相机位姿列表，每个元素包含position和quaternion
        """
        radius_mode = (radius_mode or "random").lower()
        radii: Optional[np.ndarray] = None
        if radius_mode == 'layered' and radius_layers > 1 and radius_variation > 0:
            layers = max(1, int(radius_layers))
            radius_min = max(1e-6, base_radius - radius_variation)
            radius_max = max(radius_min, base_radius + radius_variation)
            layer_radii = np.linspace(radius_min, radius_max, layers, dtype=np.float32)
            counts = np.full(layers, num_views // layers, dtype=int)
            counts[: num_views % layers] += 1
            positions_list: List[np.ndarray] = []
            radii_list: List[np.ndarray] = []
            for layer_idx, count in enumerate(counts):
                if count <= 0:
                    continue
                if hemisphere == 'upper':
                    layer_positions, _ = generate_fibonacci_upper_hemisphere_points(
                        count, radius=1.0, up_axis=self.up_axis
                    )
                else:
                    layer_positions, _ = generate_fibonacci_sphere_points(count, radius=1.0)
                positions_list.append(layer_positions)
                radii_list.append(np.full(count, layer_radii[layer_idx], dtype=np.float32))
            if positions_list:
                sphere_positions = np.concatenate(positions_list, axis=0)
                radii = np.concatenate(radii_list, axis=0)
            else:
                if hemisphere == 'upper':
                    sphere_positions, _ = generate_fibonacci_upper_hemisphere_points(
                        num_views, radius=1.0, up_axis=self.up_axis
                    )
                else:
                    sphere_positions, _ = generate_fibonacci_sphere_points(num_views, radius=1.0)
                radii = np.full(num_views, base_radius, dtype=np.float32)
        else:
            if hemisphere == 'upper':
                sphere_positions, _ = generate_fibonacci_upper_hemisphere_points(
                    num_views, radius=1.0, up_axis=self.up_axis
                )
            else:
                sphere_positions, _ = generate_fibonacci_sphere_points(num_views, radius=1.0)

            if radius_mode == 'constant' or radius_variation <= 0:
                radii = np.full(num_views, base_radius, dtype=np.float32)
            elif radius_mode == 'random':
                radii = None  # defer to random sampling
            else:
                # layered with insufficient variation defaults to constant radius
                radii = np.full(num_views, base_radius, dtype=np.float32)

        if radius_mode not in {'constant', 'random', 'layered'}:
            raise ValueError(
                f"Unsupported radius_mode '{radius_mode}'. Expected 'constant', 'random', or 'layered'."
            )

        return self._generate_poses_from_positions(
            sphere_positions,
            seed,
            base_radius,
            radius_variation,
            radii=radii,
        )
    
    def save_camera_poses(self, camera_poses: List[Dict], filepath: str):
        """
        保存相机位姿到JSON文件
        
        Args:
            camera_poses: 相机位姿列表
            filepath: 保存路径
        """
        with open(filepath, 'w') as f:
            json.dump(camera_poses, f, indent=2)
    
    def load_camera_poses(self, filepath: str) -> List[Dict]:
        """
        从JSON文件加载相机位姿
        
        Args:
            filepath: 文件路径
            
        Returns:
            camera_poses: 相机位姿列表
        """
        with open(filepath, 'r') as f:
            return json.load(f)


def pose_dict_to_tensor(pose_dict: Dict[str, List[float]], device: str = "cuda") -> torch.Tensor:
    """
    将相机位姿字典转换为张量格式
    
    Args:
        pose_dict: 包含position和quaternion的字典
        device: 计算设备
        
    Returns:
        pose_tensor: 相机位姿张量 [1, 7] (x, y, z, qx, qy, qz, qw)
    """
    position = pose_dict["position"]
    quaternion = pose_dict["quaternion"]
    
    pose_tensor = torch.tensor([
        position + quaternion
    ], dtype=torch.float32, device=device)
    
    return pose_tensor


def tensor_to_pose_dict(pose_tensor: torch.Tensor) -> Dict[str, List[float]]:
    """
    将相机位姿张量转换为字典格式
    
    Args:
        pose_tensor: 相机位姿张量 [1, 7] (x, y, z, qx, qy, qz, qw)
        
    Returns:
        pose_dict: 包含position和quaternion的字典
    """
    pose_array = pose_tensor.cpu().numpy()[0]
    
    return {
        "position": pose_array[:3].tolist(),
        "quaternion": pose_array[3:].tolist()
    }


def position_to_pose_tensor(positions: torch.Tensor, up_axis: str = "Y") -> torch.Tensor:
    """
    将位置张量转换为完整的相机位姿张量（包含位置和四元数）
    
    Args:
        positions: 相机位置张量 [B, 3] (x, y, z)
        up_axis: 上朝向轴，默认为"Y"
        
    Returns:
        pose_tensor: 相机位姿张量 [B, 7] (x, y, z, qx, qy, qz, qw)
    """
    batch_size = positions.shape[0]
    device = positions.device
    
    # 获取up向量
    up_vector = get_up_vector(up_axis)
    up = torch.tensor(up_vector, dtype=torch.float32).to(device).unsqueeze(0).expand(batch_size, -1)
    
    # 目标点（看向原点）
    at = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)
    
    positions_float32 = positions.to(torch.float32)
    # 使用PyTorch3D的look_at_view_transform生成旋转矩阵
    R, T = look_at_view_transform(eye=positions_float32, at=at, up=up)
    
    # 确保旋转矩阵在正确的设备上
    R = R.to(device)
    
    # 将旋转矩阵转换为四元数
    quaternions_wxyz = matrix_to_quaternion(R)
    # 转换为xyzw格式
    quaternions_xyzw = torch.stack([
        quaternions_wxyz[:, 1],  # x
        quaternions_wxyz[:, 2],  # y
        quaternions_wxyz[:, 3],  # z
        quaternions_wxyz[:, 0]   # w
    ], dim=1)
    
    # 拼接位置和四元数
    pose_tensor = torch.cat([positions, quaternions_xyzw], dim=1)
    
    return pose_tensor


def world_points_to_camera_depth(
    point_maps: torch.Tensor,
    camera_poses: torch.Tensor,
    *,
    valid_masks: Optional[torch.Tensor] = None,
    writer=None,
    step: Optional[int] = None,
    log_prefix: str = "DepthZ",
    train_flag: bool = False,
    depth_cmap: str = "viridis",
) -> torch.Tensor:
    """
    将世界坐标系点云转换为相机坐标系 Z 深度。
    此版本完全遵循 PyTorch3D 的行向量约定，避免了不必要的转置。

    Args:
        point_maps: [..., 3] 世界坐标点，支持 [S, H, W, 3] 或 [B, S, H, W, 3]
        camera_poses: [..., 7] 相机位姿 (position xyz + quaternion qx,qy,qz,qw，一致为 W2C 旋转)
        valid_masks: 与 point_maps 前几维一致的有效像素掩码，缺省表示全部有效

    Returns:
        depth_z: 与 point_maps 同维度的深度张量，最后一维为 1
    """
    if point_maps.shape[-1] != 3:
        raise ValueError(f"point_maps last dim must be 3, got {point_maps.shape}")
    if camera_poses.shape[-1] != 7:
        raise ValueError(f"camera_poses last dim must be 7, got {camera_poses.shape}")

    is_batched = point_maps.ndim == 5 and camera_poses.ndim == 3
    if point_maps.ndim == 4 and camera_poses.ndim == 2:
        batch_views = 1
        views = point_maps.shape[0]
        height, width = point_maps.shape[1:3]
        points_flat = point_maps
        poses_flat = camera_poses
        masks_flat = valid_masks if valid_masks is not None else None
    elif is_batched:
        batch_views = point_maps.shape[0]
        views = point_maps.shape[1]
        height, width = point_maps.shape[2:4]
        points_flat = point_maps.reshape(batch_views * views, height, width, 3)
        poses_flat = camera_poses.reshape(batch_views * views, 7)
        if valid_masks is not None:
            masks_flat = valid_masks.reshape(batch_views * views, height, width)
        else:
            masks_flat = None
    else:
        raise ValueError(
            f"Unsupported shapes for point_maps {point_maps.shape} and camera_poses {camera_poses.shape}."
        )

    device = points_flat.device
    dtype = points_flat.dtype

    positions = poses_flat[:, :3].to(device=device, dtype=dtype)
    quaternions = poses_flat[:, 3:].to(device=device, dtype=dtype)
    quaternion_wxyz = torch.stack(
        (quaternions[:, 3], quaternions[:, 0], quaternions[:, 1], quaternions[:, 2]),
        dim=-1,
    )
    # quaternion_to_matrix 返回的是为行向量设计的矩阵，我们直接使用它
    rotation_w2c_row = quaternion_to_matrix(quaternion_wxyz)  # [N, 3, 3]

    points_vec = points_flat.view(points_flat.shape[0], -1, 3) # [N, H*W, 3]
    relative = points_vec - positions.unsqueeze(1) # [N, H*W, 3], 这就是一批行向量

    camera_points = torch.bmm(relative, rotation_w2c_row)

    depth = camera_points[..., 2:3].view(points_flat.shape[0], height, width, 1)

    # --- 掩码和恢复形状部分，完全正确，无需修改 ---
    if masks_flat is not None:
        mask = masks_flat.to(device=device).unsqueeze(-1)
        depth = depth.masked_fill(~mask, 0.0)

    depth = depth.to(dtype=torch.float32)

    if point_maps.ndim == 4:
        depth_out = depth
        masks_out = masks_flat
        depth_for_log = depth_out
        masks_for_log = masks_out
    else:
        depth_out = depth.view(batch_views, views, height, width, 1)
        masks_out = valid_masks
        depth_for_log = depth_out.view(-1, height, width, 1)
        masks_for_log = None if masks_out is None else masks_out.view(-1, height, width)

    if writer is not None and step is not None and train_flag:
        _log_depth_maps(
            writer=writer,
            step=step,
            log_prefix=log_prefix,
            depth=depth_for_log,
            valid_masks=masks_for_log,
            cmap_name=depth_cmap,
        )

    return depth_out

def normalize_depth_for_visualization(
    depth_z: torch.Tensor,
    valid_masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    将深度值归一化到 [0, 1]，便于保存或可视化。

    Args:
        depth_z: [..., 1] 深度张量，支持 [S, H, W, 1] 或 [B, S, H, W, 1]
        valid_masks: 与 depth_z 前几维匹配的布尔掩码，缺省表示全部有效

    Returns:
        depth_viz: 去掉最后一维的归一化深度张量
    """
    if depth_z.numel() == 0:
        return depth_z.new_zeros(depth_z.shape[:-1])

    depth_shape = depth_z.shape
    is_batched = depth_z.ndim == 5

    if is_batched:
        batch_size, num_views, height, width, _ = depth_shape
        depth_flat = depth_z.view(batch_size * num_views, height, width)
        if valid_masks is not None:
            mask_flat = valid_masks.view(batch_size * num_views, height, width).to(device=depth_flat.device)
        else:
            mask_flat = None
    else:
        num_views, height, width, _ = depth_shape
        depth_flat = depth_z.view(num_views, height, width)
        mask_flat = valid_masks.to(device=depth_flat.device) if valid_masks is not None else None

    depth_viz_flat = torch.zeros_like(depth_flat)

    for view_idx in range(depth_flat.shape[0]):
        depth_view = depth_flat[view_idx]
        if mask_flat is not None:
            mask_view = mask_flat[view_idx]
            valid_values = depth_view[mask_view]
        else:
            mask_view = None
            valid_values = depth_view.view(-1)

        if valid_values.numel() == 0:
            continue

        depth_min = valid_values.min()
        depth_max = valid_values.max()
        denom = depth_max - depth_min
        if denom < 1e-6:
            normalized = torch.zeros_like(depth_view)
        else:
            normalized = (depth_view - depth_min) / (denom + 1e-6)

        if mask_view is not None:
            normalized = normalized.masked_fill(~mask_view, 0.0)

        depth_viz_flat[view_idx] = normalized

    if is_batched:
        return depth_viz_flat.view(batch_size, num_views, height, width)
    return depth_viz_flat


def _log_depth_maps(
    *,
    writer,
    step: int,
    log_prefix: str,
    depth: torch.Tensor,
    valid_masks: Optional[torch.Tensor],
    cmap_name: str = "viridis",
) -> None:
    if depth.numel() == 0:
        return

    if depth.dim() == 3:
        depth = depth.unsqueeze(-1)

    depth_viz = normalize_depth_for_visualization(depth, valid_masks)
    depth_viz_cpu = depth_viz.detach().float().cpu().contiguous()
    color_images = _depth_to_colormap(depth_viz_cpu, cmap_name=cmap_name)
    max_log = min(color_images.shape[0], 4)

    masks_cpu = None
    if valid_masks is not None:
        masks_cpu = valid_masks.detach().float().cpu().contiguous()
        if masks_cpu.dim() == 4:
            masks_cpu = masks_cpu.view(-1, masks_cpu.shape[-2], masks_cpu.shape[-1])
    for idx in range(max_log):
        writer.add_image(
            f"{log_prefix}/depth_view{idx}",
            color_images[idx],
            global_step=step,
        )
        if masks_cpu is not None and idx < masks_cpu.shape[0]:
            writer.add_image(
                f"{log_prefix}/mask_view{idx}",
                masks_cpu[idx].unsqueeze(0),
                global_step=step,
            )


def _depth_to_colormap(depth_viz: torch.Tensor, cmap_name: str = "viridis") -> torch.Tensor:
    """将归一化深度转换为伪彩色图像 (N, 3, H, W)。"""
    if mpl_cm is None:
        # 如果未安装 matplotlib，退化为重复灰度图
        return depth_viz.unsqueeze(1).repeat(1, 3, 1, 1).clamp(0.0, 1.0)

    cmap = mpl_cm.get_cmap(cmap_name)
    depth_np = depth_viz.numpy()  # depth_viz 应已在 CPU 上
    colored_np = cmap(depth_np)[..., :3]
    colored = torch.from_numpy(colored_np).permute(0, 3, 1, 2).contiguous()
    return colored.clamp(0.0, 1.0).to(dtype=torch.float32)
