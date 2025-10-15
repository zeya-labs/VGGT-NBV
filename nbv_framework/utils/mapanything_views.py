"""Utilities for preparing MapAnything multi-modal view inputs."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from mapanything.utils.image import IMAGE_NORMALIZATION_DICT, find_closest_aspect_ratio
from mapanything.utils.inference import (
    preprocess_input_views_for_inference,
    validate_input_views_for_inference,
)
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.utils.camera_conversions import opencv_from_cameras_projection

_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def compute_pinhole_intrinsics(height: int, width: int, fov_degrees: float) -> torch.Tensor:
    """Construct a simple pinhole camera intrinsic matrix."""
    fov_radians = math.radians(float(fov_degrees))
    fy = 0.5 * height / math.tan(fov_radians / 2.0)
    fx = 0.5 * width / math.tan(fov_radians / 2.0)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    intrinsics = torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return intrinsics


def _scale_and_crop_intrinsics(
    intrinsics: torch.Tensor,
    scale: float,
    crop_left: float,
    crop_top: float,
) -> torch.Tensor:
    """Match camera_matrix_of_crop from MapAnything utils using torch ops."""
    device = intrinsics.device
    dtype = intrinsics.dtype

    scale_tensor = torch.as_tensor(scale, dtype=dtype, device=device)
    offset = torch.as_tensor([crop_left, crop_top], dtype=dtype, device=device)

    intrinsics_adj = intrinsics.clone()
    intrinsics_adj[0, 2] += 0.5
    intrinsics_adj[1, 2] += 0.5

    intrinsics_adj[:2, :] *= scale_tensor
    intrinsics_adj[0, 2] -= offset[0]
    intrinsics_adj[1, 2] -= offset[1]

    intrinsics_adj[0, 2] -= 0.5
    intrinsics_adj[1, 2] -= 0.5

    return intrinsics_adj


def pose7d_to_opencv_cam2world_with_official_func(
    pose: torch.Tensor,
    image_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Convert one or more 7D cam2world poses into OpenCV cam2world matrices."""
    if pose.dim() == 1:
        pose = pose.unsqueeze(0)
        squeeze_batch_dim = True
    elif pose.dim() == 2 and pose.shape[-1] == 7:
        squeeze_batch_dim = False
    else:
        raise ValueError(f"Expected pose tensor with shape (7,) or (N, 7), got {pose.shape}.")

    pose_float = pose.to(dtype=torch.float32)
    device = pose_float.device

    position_c2w = pose_float[..., :3]  # (N, 3)
    quaternion_xyzw = pose_float[..., 3:]  # (N, 4)
    quaternion_wxyz = quaternion_xyzw[..., [3, 0, 1, 2]]

    rotation_w2c = quaternion_to_matrix(quaternion_wxyz)  # (N, 3, 3)
    translation_w2c = -torch.bmm(position_c2w.unsqueeze(1), rotation_w2c).squeeze(1)  # (N, 3)

    cameras = PerspectiveCameras(
        R=rotation_w2c,
        T=translation_w2c,
        device=device,
    )

    image_size_tensor = torch.as_tensor(image_size, device=device, dtype=pose_float.dtype).view(1, 2)
    image_size_tensor = image_size_tensor.expand(pose.shape[0], -1)

    rotation_w2c_opencv, translation_w2c_opencv, _ = opencv_from_cameras_projection(
        cameras,
        image_size_tensor,
    )

    rotation_c2w_opencv = rotation_w2c_opencv.transpose(1, 2)
    position_c2w_opencv = -torch.bmm(
        rotation_c2w_opencv,
        translation_w2c_opencv.unsqueeze(-1),
    ).squeeze(-1)

    cam2world = torch.eye(4, dtype=pose_float.dtype, device=device).unsqueeze(0).repeat(pose.shape[0], 1, 1)
    cam2world[:, :3, :3] = rotation_c2w_opencv
    cam2world[:, :3, 3] = position_c2w_opencv

    if squeeze_batch_dim:
        cam2world = cam2world.squeeze(0)

    return cam2world

def _build_base_view(
    normalized_img: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_pose_matrix: torch.Tensor,
    data_norm_type: str,
    is_metric: torch.Tensor,
) -> Dict[str, Any]:
    """Assemble a single-view dictionary before inference preprocessing."""
    return {
        "img": normalized_img,
        "data_norm_type": [data_norm_type],
        "intrinsics": intrinsics,
        "camera_poses": camera_pose_matrix,
        "is_metric_scale": is_metric,
    }


def prepare_mapanything_views(
    images: torch.Tensor,
    camera_poses: torch.Tensor,
    *,
    data_norm_type: str,
    resolution_set: int,
    device: torch.device,
    patch_size: int,
    fov_degrees: float = 60.0,
    is_metric_scale: bool = False,
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    """Prepare batched MapAnything views with intrinsics and camera poses."""
    if images.dim() == 4:
        images = images.unsqueeze(0)
    if images.dim() != 5:
        raise ValueError(
            f"Expected images with shape [B, S, 3, H, W] or [S, 3, H, W], got {images.shape}"
        )

    images = images.clamp(0.0, 1.0).to(device)
    batch_size, num_views, _, _, _ = images.shape

    if camera_poses.dim() == 2:
        camera_poses = camera_poses.unsqueeze(1)
    camera_poses = camera_poses.to(device=device, dtype=torch.float32)
    if camera_poses.shape[:2] != (batch_size, num_views):
        raise ValueError(
            f"Camera poses shape {camera_poses.shape} incompatible with images {images.shape}"
        )

    aspect_ratios: List[float] = []
    for view_idx in range(num_views):
        view_tensor = images[:, view_idx]
        _, _, height, width = view_tensor.shape
        if height <= 0 or width <= 0:
            raise ValueError("Image spatial dimensions must be positive")
        aspect_ratios.extend([width / height] * batch_size)

    if not aspect_ratios:
        raise ValueError("Unable to compute aspect ratios from input images")

    average_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
    target_width, target_height = find_closest_aspect_ratio(
        average_aspect_ratio, resolution_set
    )
    if patch_size <= 0:
        patch_size = 14
    target_width = max(patch_size, (target_width // patch_size) * patch_size)
    target_height = max(patch_size, (target_height // patch_size) * patch_size)

    norm_cfg = IMAGE_NORMALIZATION_DICT.get(data_norm_type)
    if norm_cfg is None:
        mean = _DEFAULT_MEAN
        std = _DEFAULT_STD
    else:
        mean = norm_cfg.mean
        std = norm_cfg.std

    mean_tensor = torch.as_tensor(mean, dtype=images.dtype, device=device).view(1, 3, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=images.dtype, device=device).view(1, 3, 1, 1)

    base_views: List[Dict[str, Any]] = []
    normalized_views: List[torch.Tensor] = []

    is_metric_tensor = torch.full((batch_size,), bool(is_metric_scale), dtype=torch.bool, device=device)

    for view_idx in range(num_views):
        view_tensor = images[:, view_idx]
        _, _, source_h, source_w = view_tensor.shape

        scale = max(target_height / source_h, target_width / source_w)
        scaled_h = max(target_height, int(math.floor(source_h * scale)))
        scaled_w = max(target_width, int(math.floor(source_w * scale)))

        if scaled_h != source_h or scaled_w != source_w:
            resized = F.interpolate(
                view_tensor,
                size=(scaled_h, scaled_w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            resized = view_tensor

        top = max((scaled_h - target_height) // 2, 0)
        left = max((scaled_w - target_width) // 2, 0)
        bottom = top + target_height
        right = left + target_width
        cropped = resized[:, :, top:bottom, left:right]
        normalized = (cropped - mean_tensor) / std_tensor

        normalized_views.append(normalized)

        base_intrinsics = compute_pinhole_intrinsics(source_h, source_w, fov_degrees)
        base_intrinsics = base_intrinsics.to(device=device)
        intrinsics_scaled = _scale_and_crop_intrinsics(
            base_intrinsics,
            scale=scale,
            crop_left=float(left),
            crop_top=float(top),
        )
        intrinsics_batched = intrinsics_scaled.unsqueeze(0).repeat(batch_size, 1, 1).to(device=device)

        pose_vectors = camera_poses[:, view_idx]
        camera_pose_tensor = pose7d_to_opencv_cam2world_with_official_func(
            pose_vectors,
            image_size=(target_height, target_width),
        ).to(device=device)

        base_views.append(
            _build_base_view(
                normalized,
                intrinsics_batched,
                camera_pose_tensor,
                data_norm_type=data_norm_type,
                is_metric=is_metric_tensor.clone(),
            )
        )

    validated_views = validate_input_views_for_inference(base_views)
    processed_views = preprocess_input_views_for_inference(validated_views)
    normalized_stack = torch.stack(normalized_views, dim=1)
    return processed_views, normalized_stack
