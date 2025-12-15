"""
相机工具模块
处理相机位姿生成、保存和转换
"""

import json
import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
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
        base_radius: float = 2.6, # 0.9, 0.8是2.5，0.7是2.86
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


def position_to_pose_tensor(
    positions: torch.Tensor,
    up_axis: str = "Y",
    look_at: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    将位置张量转换为完整的相机位姿张量（包含位置和四元数）

    Args:
        positions: 相机位置张量 [B, 3] (x, y, z)
        up_axis: 上朝向轴，默认为"Y"
        look_at: 可选的目标点张量，形状可为 [3] 或 [B, 3]，默认看向原点

    Returns:
        pose_tensor: 相机位姿张量 [B, 7] (x, y, z, qx, qy, qz, qw)
    """
    if not torch.is_tensor(positions):
        positions = torch.tensor(positions, dtype=torch.float32)
    else:
        positions = positions.to(dtype=torch.float32)

    if positions.ndim == 1:
        if positions.numel() != 3:
            raise ValueError(
                f"positions expects 3 values for a single camera, but got shape {tuple(positions.shape)}"
            )
        positions = positions.unsqueeze(0)
    elif positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions must have shape [B, 3], but received {tuple(positions.shape)}"
        )

    batch_size = positions.shape[0]
    device = positions.device

    # 获取up向量
    up_vector = get_up_vector(up_axis)
    up = (
        torch.tensor(up_vector, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )

    if look_at is None:
        at = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)
    else:
        look_at_tensor = torch.as_tensor(look_at, dtype=torch.float32, device=device)
        if look_at_tensor.ndim == 1:
            if look_at_tensor.numel() != 3:
                raise ValueError(
                    f"look_at expects 3 values for a single target, but got shape {tuple(look_at_tensor.shape)}"
                )
            look_at_tensor = look_at_tensor.unsqueeze(0)
        elif look_at_tensor.ndim != 2 or look_at_tensor.shape[1] != 3:
            raise ValueError(
                f"look_at must have shape [B, 3] or [3], but received {tuple(look_at_tensor.shape)}"
            )

        if look_at_tensor.shape[0] == 1 and batch_size > 1:
            look_at_tensor = look_at_tensor.expand(batch_size, -1)
        elif look_at_tensor.shape[0] != batch_size:
            raise ValueError(
                f"look_at batch size ({look_at_tensor.shape[0]}) does not match positions batch size ({batch_size})."
            )
        at = look_at_tensor

    # 使用PyTorch3D的look_at_view_transform生成旋转矩阵
    R, T = look_at_view_transform(eye=positions, at=at, up=up)

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

    Args:
        point_maps: [..., 3] 世界坐标点，支持 [B, H, W, 3] 或 [B, S, H, W, 3]
        camera_poses: [..., 7] 相机位姿 (position xyz + quaternion qx,qy,qz,qw)
        valid_masks: 有效掩码
    """
    if point_maps.shape[-1] != 3:
        raise ValueError(f"point_maps last dim must be 3, got {point_maps.shape}")
    if camera_poses.shape[-1] != 7:
        raise ValueError(f"camera_poses last dim must be 7, got {camera_poses.shape}")

    # --- 1. 形状归一化处理 (Shape Handling) ---
    # 逻辑是正确的：将 [B, S, ...] 或 [B, ...] 统一展平为 [N, ...] 处理
    is_batched_sequence = (point_maps.ndim == 5) and (camera_poses.ndim == 3)
    
    if point_maps.ndim == 4 and camera_poses.ndim == 2:
        # Case: [B, H, W, 3] -> 单视图 Batch
        batch_size = point_maps.shape[0]
        s_dim = 1
        height, width = point_maps.shape[1:3]
        
        points_flat = point_maps.reshape(-1, height, width, 3) # [N, H, W, 3]
        poses_flat = camera_poses.reshape(-1, 7)               # [N, 7]
        masks_flat = valid_masks.reshape(-1, height, width) if valid_masks is not None else None
        
    elif is_batched_sequence:
        # Case: [B, S, H, W, 3] -> 多视图序列 Batch
        batch_size = point_maps.shape[0]
        s_dim = point_maps.shape[1]
        height, width = point_maps.shape[2:4]
        
        points_flat = point_maps.reshape(-1, height, width, 3) # [B*S, H, W, 3]
        poses_flat = camera_poses.reshape(-1, 7)               # [B*S, 7]
        masks_flat = valid_masks.reshape(-1, height, width) if valid_masks is not None else None
    else:
        raise ValueError(
            f"Unsupported shapes: point_maps {point_maps.shape}, camera_poses {camera_poses.shape}"
        )

    device = points_flat.device
    dtype = points_flat.dtype

    # --- 2. 提取位姿参数 ---
    # 假设输入是 [x, y, z, qx, qy, qz, qw]
    positions = poses_flat[:, :3].to(device=device, dtype=dtype)
    quaternions = poses_flat[:, 3:].to(device=device, dtype=dtype)
    
    # PyTorch3D / 内部函数通常需要 (w, x, y, z) 顺序
    quaternion_wxyz = torch.stack(
        (quaternions[:, 3], quaternions[:, 0], quaternions[:, 1], quaternions[:, 2]),
        dim=-1,
    )
    
    # --- 3. 计算旋转矩阵 ---
    # PyTorch3D 的旋转变换采用行向量约定：p_rot = p @ R
    # 其中 quaternion_to_matrix 返回的就是该行向量形式的 world->camera 旋转矩阵 R_w2c_row。
    R_w2c_row = quaternion_to_matrix(quaternion_wxyz)  # [N, 3, 3]

    # --- 4. 坐标变换 ---
    points_vec = points_flat.view(points_flat.shape[0], -1, 3) # [N, H*W, 3]
    
    # 平移: (P - T)
    # positions.unsqueeze(1) -> [N, 1, 3], 广播相减
    relative = points_vec - positions.unsqueeze(1) 
    
    # 旋转: (P - C) @ R_w2c_row
    camera_points = torch.bmm(relative, R_w2c_row)

    # 提取 Z 深度
    depth = camera_points[..., 2:3].view(points_flat.shape[0], height, width, 1)

    # --- 5. 掩码处理 ---
    if masks_flat is not None:
        mask = masks_flat.to(device=device).unsqueeze(-1)
        depth = depth.masked_fill(~mask, 0.0)

    depth = depth.to(dtype=torch.float32)

    # --- 6. 恢复原始形状 ---
    if is_batched_sequence:
        depth_out = depth.view(batch_size, s_dim, height, width, 1)
        masks_out = valid_masks
    else:
        # [B, H, W, 1]
        depth_out = depth
        masks_out = valid_masks

    # --- 7. Logging ---
    if writer is not None and step is not None and train_flag:
        # Log 时统一展平，方便 add_image
        depth_for_log = depth_out.reshape(-1, height, width, 1)
        masks_for_log = masks_out.reshape(-1, height, width) if masks_out is not None else None
        
        _log_depth_maps(
            writer=writer,
            step=step,
            log_prefix=log_prefix,
            depth=depth_for_log,
            valid_masks=masks_for_log,
            cmap_name=depth_cmap,
        )

    return depth_out


def camera_depth_z_to_world_points(
    depth_z: torch.Tensor,
    camera_poses: torch.Tensor,
    *,
    fov_degrees: float = 60.0,
    valid_masks: Optional[torch.Tensor] = None,
    xy_signs: Tuple[int, int] = (1, 1),
) -> torch.Tensor:
    """
    将相机坐标系下的 Z 深度图反投影为世界坐标点云(point maps)。

    该函数与 :func:`world_points_to_camera_depth` 互为“近似逆”：
    - 前者: p_cam = (p_world - C) @ R_w2c
    - 本函数: p_world = p_cam @ R_w2c^T + C

    本项目遵循 PyTorch3D 的行向量约定：
    - world->cam: p_cam = (p_world - C) @ R_w2c_row
    - cam->world: p_world = p_cam @ R_w2c_row^T + C

    其中 (C, R_w2c) 来自 camera_poses: position xyz + quaternion qx,qy,qz,qw (world->camera)。

    注意：仅依赖深度、位姿和固定 pinhole 内参（由 fov+分辨率确定）。

    Args:
        depth_z: [..., H, W] 或 [..., H, W, 1]，相机坐标系下的 Z 分量。
        camera_poses: [..., 7]，与 depth_z 的前置维度对齐。
        fov_degrees: 与渲染器一致的视场角，用于构建 pinhole 内参。
        valid_masks: 可选 [..., H, W] 的有效像素掩码；无效像素输出为 0。
        xy_signs: (sx, sy) 两个符号，用于处理图像坐标到相机坐标的轴向约定差异。

    Returns:
        world_points: [..., H, W, 3] 世界坐标点云。
    """
    if depth_z.ndim < 2:
        raise ValueError(f"depth_z must have at least 2 dims, got {tuple(depth_z.shape)}")
    if camera_poses.shape[-1] != 7:
        raise ValueError(f"camera_poses last dim must be 7, got {tuple(camera_poses.shape)}")

    if depth_z.shape[-1] == 1:
        depth_z = depth_z.squeeze(-1)

    height, width = int(depth_z.shape[-2]), int(depth_z.shape[-1])
    leading_shape = tuple(depth_z.shape[:-2])

    depth_flat = depth_z.reshape(-1, height, width)
    poses_flat = camera_poses.reshape(-1, 7)
    if poses_flat.shape[0] != depth_flat.shape[0]:
        raise ValueError(
            "Leading dimensions of depth_z and camera_poses must match. "
            f"Got depth_z={tuple(depth_z.shape)} vs camera_poses={tuple(camera_poses.shape)}."
        )

    masks_flat: Optional[torch.Tensor]
    if valid_masks is None:
        masks_flat = None
    else:
        if valid_masks.shape[-2:] != (height, width):
            raise ValueError(
                "valid_masks spatial dims must match depth_z. "
                f"Got valid_masks={tuple(valid_masks.shape)} vs depth_z={tuple(depth_z.shape)}."
            )
        masks_flat = valid_masks.reshape(-1, height, width).to(device=depth_flat.device)

    device = depth_flat.device
    dtype = depth_flat.dtype

    fov_radians = math.radians(float(fov_degrees))
    fy = 0.5 * float(height) / math.tan(fov_radians / 2.0)
    fx = 0.5 * float(width) / math.tan(fov_radians / 2.0)
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0

    u = torch.arange(width, device=device, dtype=dtype)
    v = torch.arange(height, device=device, dtype=dtype)
    try:
        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    except TypeError:  # pragma: no cover - older torch
        v_grid, u_grid = torch.meshgrid(v, u)

    sx, sy = int(xy_signs[0]), int(xy_signs[1])
    x_cam = (u_grid - cx) / fx * depth_flat * float(sx)
    y_cam = (v_grid - cy) / fy * depth_flat * float(sy)
    z_cam = depth_flat
    cam_points = torch.stack((x_cam, y_cam, z_cam), dim=-1)  # [N, H, W, 3]

    positions = poses_flat[:, :3].to(device=device, dtype=dtype)
    quaternions = poses_flat[:, 3:].to(device=device, dtype=dtype)
    quaternion_wxyz = torch.stack(
        (quaternions[:, 3], quaternions[:, 0], quaternions[:, 1], quaternions[:, 2]),
        dim=-1,
    )
    # quaternion_to_matrix 返回行向量旋转矩阵 R_w2c_row (world->cam)
    rotation_w2c_row = quaternion_to_matrix(quaternion_wxyz)  # [N, 3, 3]
    rotation_c2w_row = rotation_w2c_row.transpose(1, 2)

    cam_points_vec = cam_points.view(cam_points.shape[0], -1, 3)
    world_points_vec = torch.bmm(cam_points_vec, rotation_c2w_row) + positions.unsqueeze(1)
    world_points = world_points_vec.view(cam_points.shape[0], height, width, 3)

    if masks_flat is not None:
        world_points = world_points.masked_fill(~masks_flat.unsqueeze(-1), 0.0)

    return world_points.reshape(*leading_shape, height, width, 3)


def infer_depth_backprojection_xy_signs(
    depth_z: torch.Tensor,
    camera_poses: torch.Tensor,
    reference_world_points: torch.Tensor,
    *,
    fov_degrees: float = 60.0,
    valid_masks: Optional[torch.Tensor] = None,
    max_samples: int = 4096,
) -> Tuple[int, int]:
    """
    基于渲染得到的 reference_world_points 自动推断反投影 (sx, sy) 约定。

    在不同渲染/相机坐标约定下，像素坐标到相机坐标 (x,y) 的符号可能不同，
    该函数通过最小化反投影点与 reference_world_points 的误差来选择符号组合。

    Returns:
        (sx, sy) ∈ {+1, -1}^2
    """
    with torch.no_grad():
        if depth_z.shape[-1] == 1:
            depth_z = depth_z.squeeze(-1)

        height, width = int(depth_z.shape[-2]), int(depth_z.shape[-1])
        leading_shape = tuple(depth_z.shape[:-2])
        depth_flat = depth_z.reshape(-1, height * width)

        ref = reference_world_points
        if ref.ndim < 3 or ref.shape[-1] != 3:
            raise ValueError(
                "reference_world_points must have shape [..., H, W, 3], "
                f"but got {tuple(reference_world_points.shape)}"
            )
        if tuple(ref.shape[:-3]) != leading_shape or tuple(ref.shape[-3:-1]) != (height, width):
            raise ValueError(
                "reference_world_points must align with depth_z leading/spatial dims. "
                f"Got depth_z={tuple(depth_z.shape)} vs reference_world_points={tuple(reference_world_points.shape)}."
            )
        ref_flat = ref.reshape(-1, height * width, 3)

        poses_flat = camera_poses.reshape(-1, 7)
        if poses_flat.shape[0] != depth_flat.shape[0]:
            raise ValueError(
                "Leading dimensions of depth_z and camera_poses must match. "
                f"Got depth_z={tuple(depth_z.shape)} vs camera_poses={tuple(camera_poses.shape)}."
            )

        if valid_masks is None:
            mask_flat = depth_flat != 0
        else:
            if valid_masks.shape[-2:] != (height, width):
                raise ValueError(
                    "valid_masks spatial dims must match depth_z. "
                    f"Got valid_masks={tuple(valid_masks.shape)} vs depth_z={tuple(depth_z.shape)}."
                )
            mask_flat = valid_masks.reshape(-1, height * width).to(device=depth_flat.device)

        valid_n, valid_idx = mask_flat.nonzero(as_tuple=True)
        if valid_idx.numel() == 0:
            return (1, 1)

        if valid_idx.numel() > max_samples:
            valid_n = valid_n[:max_samples]
            valid_idx = valid_idx[:max_samples]

        device = depth_flat.device
        dtype = depth_flat.dtype

        fov_radians = math.radians(float(fov_degrees))
        fy = 0.5 * float(height) / math.tan(fov_radians / 2.0)
        fx = 0.5 * float(width) / math.tan(fov_radians / 2.0)
        cx = (float(width) - 1.0) / 2.0
        cy = (float(height) - 1.0) / 2.0

        u = (valid_idx % width).to(device=device, dtype=dtype)
        v = (valid_idx // width).to(device=device, dtype=dtype)
        z = depth_flat[valid_n, valid_idx]

        positions = poses_flat[:, :3].to(device=device, dtype=dtype)
        quaternions = poses_flat[:, 3:].to(device=device, dtype=dtype)
        quaternion_wxyz = torch.stack(
            (quaternions[:, 3], quaternions[:, 0], quaternions[:, 1], quaternions[:, 2]),
            dim=-1,
        )
        rotation_w2c_row = quaternion_to_matrix(quaternion_wxyz)  # [N, 3, 3]
        rotation_c2w_row = rotation_w2c_row.transpose(1, 2)

        positions_s = positions.index_select(0, valid_n)
        rotation_s = rotation_c2w_row.index_select(0, valid_n)
        ref_s = ref_flat[valid_n, valid_idx]

        best = (1, 1)
        best_err = None
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            x = (u - cx) / fx * z * float(sx)
            y = (v - cy) / fy * z * float(sy)
            cam = torch.stack((x, y, z), dim=-1)
            world = torch.bmm(cam.unsqueeze(1), rotation_s).squeeze(1) + positions_s
            err = (world - ref_s).pow(2).mean()
            if best_err is None or err < best_err:
                best_err = err
                best = (sx, sy)

        return best

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
