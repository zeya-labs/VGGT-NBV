"""Utilities for preparing MapAnything multi-modal view inputs."""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torchvision
from mapanything.utils.image import IMAGE_NORMALIZATION_DICT
from mapanything.utils.inference import (
    preprocess_input_views_for_inference,
    validate_input_views_for_inference,
)
from mapanything.utils.geometry import (
    transform_pose_using_quats_and_trans_2_to_1,
)
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.utils.camera_conversions import opencv_from_cameras_projection

_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


logger = logging.getLogger(__name__)


def compute_pinhole_intrinsics(
    height: int,
    width: int,
    fov_degrees: float,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
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
        dtype=dtype,
        device=device,
    )
    return intrinsics


def pose7d_to_opencv_cam2world_with_official_func(
    pose: torch.Tensor,
    image_size: Tuple[int, int] = (518, 518),
) -> torch.Tensor:
    """Convert one or more 7D cam2world poses into OpenCV cam2world matrices."""
    if pose.dim() == 1:
        pose = pose.unsqueeze(0)
        squeeze_batch_dim = True
    elif pose.dim() == 2 and pose.shape[-1] == 7:
        squeeze_batch_dim = False
    else:
        raise ValueError(f"Expected pose tensor with shape (7,) or (N, 7), got {pose.shape}.")

    if not pose.is_floating_point():
        raise TypeError("pose must be a floating point tensor.")

    output_dtype = pose.dtype
    pose_compute = pose.to(dtype=output_dtype)
    device = pose_compute.device

    position_c2w = pose_compute[..., :3]  # (N, 3)
    quaternion_xyzw = pose_compute[..., 3:]  # (N, 4)
    quaternion_wxyz = quaternion_xyzw[..., [3, 0, 1, 2]]

    rotation_w2c = quaternion_to_matrix(quaternion_wxyz)  # (N, 3, 3)
    translation_w2c = -torch.bmm(position_c2w.unsqueeze(1), rotation_w2c).squeeze(1)  # (N, 3)

    cameras = PerspectiveCameras(
        R=rotation_w2c,
        T=translation_w2c,
        device=device,
    )

    image_size_tensor = torch.as_tensor(image_size, device=device, dtype=pose_compute.dtype).view(1, 2)
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

    cam2world = torch.eye(4, dtype=pose_compute.dtype, device=device).unsqueeze(0).repeat(pose.shape[0], 1, 1)
    cam2world[:, :3, :3] = rotation_c2w_opencv
    cam2world[:, :3, 3] = position_c2w_opencv

    if squeeze_batch_dim:
        cam2world = cam2world.squeeze(0)

    return cam2world.to(dtype=output_dtype)

def _compute_pose_quats_and_trans_for_across_views_in_ref_view(
    views,
    num_views,
    device,
    dtype,
    batch_size_per_view,
    per_sample_cam_input_mask,
):
    """
    Compute the pose quats and trans for all the views in the frame of the reference view 0.
    Returns identity pose for views where the camera input mask is False or the pose is not provided.

    Args:
        views (List[dict]): List of dictionaries containing the input views' images and instance information.
        num_views (int): Number of views.
        device (torch.device): Device to use for the computation.
        dtype (torch.dtype): Data type to use for the computation.
        per_sample_cam_input_mask (torch.Tensor): Tensor containing the per sample camera input mask.

    Returns:
        torch.Tensor: A tensor containing the pose quats for all the views in the frame of the reference view 0. (batch_size_per_view * view, 4)
        torch.Tensor: A tensor containing the pose trans for all the views in the frame of the reference view 0. (batch_size_per_view * view, 3)
        torch.Tensor: A tensor containing the per sample camera input mask.
    """
    # Compute the pose quats and trans for all the non-reference views in the frame of the reference view 0
    pose_quats_non_ref_views = []
    pose_trans_non_ref_views = []
    pose_quats_ref_view_0 = []
    pose_trans_ref_view_0 = []
    for view_idx in range(num_views):
        per_sample_cam_input_mask_for_curr_view = per_sample_cam_input_mask[
            view_idx * batch_size_per_view : (view_idx + 1) * batch_size_per_view
        ]
        if (
            "camera_pose_quats" in views[view_idx]
            and "camera_pose_trans" in views[view_idx]
            and per_sample_cam_input_mask_for_curr_view.any()
        ):
            # Get the camera pose quats and trans for the current view
            cam_pose_quats = views[view_idx]["camera_pose_quats"][
                per_sample_cam_input_mask_for_curr_view
            ]
            cam_pose_trans = views[view_idx]["camera_pose_trans"][
                per_sample_cam_input_mask_for_curr_view
            ]
            cam_pose_quats = cam_pose_quats.to(dtype=dtype)
            cam_pose_trans = cam_pose_trans.to(dtype=dtype)
            # Append to the list
            pose_quats_non_ref_views.append(cam_pose_quats)
            pose_trans_non_ref_views.append(cam_pose_trans)
            # Get the camera pose quats and trans for the reference view 0
            cam_pose_quats = views[0]["camera_pose_quats"][
                per_sample_cam_input_mask_for_curr_view
            ]
            cam_pose_trans = views[0]["camera_pose_trans"][
                per_sample_cam_input_mask_for_curr_view
            ]
            cam_pose_quats = cam_pose_quats.to(dtype=dtype)
            cam_pose_trans = cam_pose_trans.to(dtype=dtype)
            # Append to the list
            pose_quats_ref_view_0.append(cam_pose_quats)
            pose_trans_ref_view_0.append(cam_pose_trans)
        else:
            per_sample_cam_input_mask[
                view_idx * batch_size_per_view : (view_idx + 1)
                * batch_size_per_view
            ] = False

    # Initialize the pose quats and trans for all views as identity
    pose_quats_across_views = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], dtype=dtype, device=device
    ).repeat(batch_size_per_view * num_views, 1)  # (q_x, q_y, q_z, q_w)
    pose_trans_across_views = torch.zeros(
        (batch_size_per_view * num_views, 3), dtype=dtype, device=device
    )

    # Compute the pose quats and trans for all the non-reference views in the frame of the reference view 0
    if len(pose_quats_non_ref_views) > 0:
        # Stack the pose quats and trans for all the non-reference views and reference view 0
        pose_quats_non_ref_views = torch.cat(pose_quats_non_ref_views, dim=0)
        pose_trans_non_ref_views = torch.cat(pose_trans_non_ref_views, dim=0)
        pose_quats_ref_view_0 = torch.cat(pose_quats_ref_view_0, dim=0)
        pose_trans_ref_view_0 = torch.cat(pose_trans_ref_view_0, dim=0)

        # Compute the pose quats and trans for all the non-reference views in the frame of the reference view 0
        (
            pose_quats_non_ref_views_in_ref_view_0,
            pose_trans_non_ref_views_in_ref_view_0,
        ) = transform_pose_using_quats_and_trans_2_to_1(
            pose_quats_ref_view_0,
            pose_trans_ref_view_0,
            pose_quats_non_ref_views,
            pose_trans_non_ref_views,
        )

        # Update the pose quats and trans for all the non-reference views
        pose_quats_across_views[per_sample_cam_input_mask] = (
            pose_quats_non_ref_views_in_ref_view_0.to(dtype=dtype)
        )
        pose_trans_across_views[per_sample_cam_input_mask] = (
            pose_trans_non_ref_views_in_ref_view_0.to(dtype=dtype)
        )

    return (
        pose_quats_across_views,
        pose_trans_across_views,
        per_sample_cam_input_mask,
    )

def prepare_mapanything_views(
    images: torch.Tensor,
    camera_poses: torch.Tensor,
    *,
    data_norm_type: str,
    fov_degrees: float = 60.0,
    is_metric_scale: bool = False,
    depth_z: Optional[torch.Tensor] = None,
    save_dir: Optional[str] = None,
    mesh_paths: Optional[Sequence[Optional[str]]] = None,
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    """验证输入并基于原始分辨率构建 MapAnything 视图描述。"""
    if images.dim() != 5 or images.shape[2] != 3:
        raise ValueError(
            f"images 期望形状 [B, S, 3, H, W]，实际 {tuple(images.shape)}"
        )

    if camera_poses.dim() != 3 or camera_poses.shape[-1] != 7:
        raise ValueError(
            f"camera_poses 期望形状 [B, S, 7]，实际 {tuple(camera_poses.shape)}"
        )

    device = images.device

    if depth_z is not None:
        if depth_z.dim() == 5 and depth_z.shape[-1] == 1:
            depth_z = depth_z.squeeze(-1)
        if depth_z.dim() != 4:
            raise ValueError(
                f"depth_z expected shape [B, S, H, W] or [B, S, H, W, 1], got {tuple(depth_z.shape)}"
            )

    batch_size, num_views, num_channels, _, _ = images.shape
    if num_channels != 3:
        raise ValueError("images 需要 3 个通道 (RGB)")
    if camera_poses.shape[:2] != (batch_size, num_views):
        raise ValueError(
            f"camera_poses 与 images 的 batch/view 数不匹配：{camera_poses.shape[:2]} vs {(batch_size, num_views)}"
        )
    if depth_z is not None and depth_z.shape[:2] != (batch_size, num_views):
        raise ValueError(
            f"depth_z 与 images 的 batch/view 数不匹配：{depth_z.shape[:2]} vs {(batch_size, num_views)}"
        )

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
        _, _, height, width = view_tensor.shape
        if height <= 0 or width <= 0:
            raise ValueError("图像空间尺寸必须为正数")

        normalized = (view_tensor - mean_tensor) / std_tensor
        normalized_views.append(normalized)

        base_intrinsics = compute_pinhole_intrinsics(
            height,
            width,
            fov_degrees,
            device=device,
            dtype=camera_poses.dtype,
        )
        intrinsics_batched = base_intrinsics.unsqueeze(0).repeat(batch_size, 1, 1)

        pose_vectors = camera_poses[:, view_idx]
        camera_pose_tensor = pose7d_to_opencv_cam2world_with_official_func(
            pose_vectors,
            image_size=(height, width),
        )

        view_dict: Dict[str, Any] = {
            "img": normalized,
            "data_norm_type": [data_norm_type],
            "intrinsics": intrinsics_batched,
            "camera_poses": camera_pose_tensor,
            "is_metric_scale": is_metric_tensor.clone(),
        }
        if depth_z is not None:
            view_dict["depth_z"] = depth_z[:, view_idx].contiguous()

        base_views.append(view_dict)

    validated_views = validate_input_views_for_inference(base_views)
    processed_views = preprocess_input_views_for_inference(validated_views)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        images_cpu = images.detach().cpu()
        poses_cpu = camera_poses.detach().cpu()
        depth_cpu = depth_z.detach().cpu() if depth_z is not None else None

        batch_size, num_views, _, height, width = images_cpu.shape
        intrinsics = compute_pinhole_intrinsics(height, width, fov_degrees)
        mesh_path_list: Optional[List[Optional[str]]] = None
        if mesh_paths is not None:
            mesh_path_list = list(mesh_paths)
            if len(mesh_path_list) != batch_size:
                logger.warning(
                    "mesh_paths length (%d) does not match batch size (%d); skipping mesh annotations.",
                    len(mesh_path_list),
                    batch_size,
                )
                mesh_path_list = None

        for batch_idx in range(batch_size):
            batch_dir = os.path.join(save_dir, f"batch_{batch_idx:03d}")
            os.makedirs(batch_dir, exist_ok=True)

            mesh_path_value: Optional[str] = None
            if mesh_path_list is not None:
                mesh_entry = mesh_path_list[batch_idx]
                if mesh_entry is not None:
                    mesh_path_value = str(mesh_entry)

            for view_idx in range(num_views):
                raw_img = images_cpu[batch_idx, view_idx]
                pose_tensor = poses_cpu[batch_idx, view_idx]

                png_img = torch.clamp(raw_img, 0.0, 1.0)
                png_path = os.path.join(batch_dir, f"image_{view_idx:02d}.png")
                torchvision.utils.save_image(png_img, png_path)

                img_uint8 = (png_img.permute(1, 2, 0) * 255.0).round().to(torch.uint8)
                cam2world = pose7d_to_opencv_cam2world_with_official_func(pose_tensor)

                view_payload: Dict[str, Any] = {
                    "img": img_uint8,
                    "intrinsics": intrinsics.clone(),
                    "camera_poses": cam2world,
                    "is_metric_scale": torch.tensor([bool(is_metric_scale)], dtype=torch.bool),
                }
                print("mesh_path_value:", mesh_path_value)
                if mesh_path_value is not None:
                    view_payload["mesh_path"] = mesh_path_value
                if depth_cpu is not None:
                    view_payload["depth_z"] = depth_cpu[batch_idx, view_idx].contiguous()

                payload_path = os.path.join(batch_dir, f"view_{view_idx:02d}.pt")
                torch.save(view_payload, payload_path)

        logger.info(
            "Saved view data for %d batches (%d views each) to %s",
            batch_size,
            num_views,
            save_dir,
        )
    normalized_stack = torch.stack(normalized_views, dim=1)
    return processed_views, normalized_stack
