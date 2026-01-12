"""Rendering helpers for GT point maps and masks."""

from typing import List, Optional, Tuple

import torch
from pytorch3d.structures import Meshes


def render_gt_point_maps(
    renderer,
    mesh_batch: Meshes,
    camera_poses: torch.Tensor,
    *,
    output_device: Optional[torch.device] = None,
    device: Optional[torch.device] = None,
    tensor_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render GT point maps and validity masks for a batch of meshes.

    Args:
        renderer: Differentiable renderer instance with ``device`` attribute.
        mesh_batch: Batched mesh data (length equals batch size).
        camera_poses: Camera poses ``[B, S, 7]`` or ``[B, 7]`` or ``[S, 7]`` tensors.
        output_device: Target device for the returned tensors. Defaults to CPU.

    Returns:
        Tuple of ``(point_maps, valid_masks)`` with shapes::

            point_maps: [B, S, H, W, 3]
            valid_masks: [B, S, H, W]

    Raises:
        RuntimeError: If renderer does not return expected outputs.
    """
    if camera_poses is None:
        raise ValueError("camera_poses must be provided when rendering GT point maps.")

    if device is None:
        raise ValueError("device must be provided when rendering GT point maps.")

    if output_device is None:
        output_device = torch.device("cpu")

    renderer_device = device

    batch_size = len(mesh_batch)
    if camera_poses.dim() == 2:
        if camera_poses.shape[0] == batch_size:
            camera_poses = camera_poses.unsqueeze(1)
        else:
            camera_poses = camera_poses.unsqueeze(0)

    if camera_poses.shape[0] != batch_size:
        if camera_poses.shape[0] == 1 and batch_size > 1:
            camera_poses = camera_poses.expand(batch_size, -1, -1)
        else:
            raise ValueError(
                f"camera_poses batch ({camera_poses.shape[0]}) does not match mesh batch ({batch_size})."
            )

    point_maps_list: List[torch.Tensor] = []
    valid_masks_list: List[torch.Tensor] = []

    with torch.no_grad():
        for mesh_idx in range(batch_size):
            poses_i = camera_poses[mesh_idx]
            if poses_i.numel() == 0:
                raise ValueError("camera_poses contains empty view set; cannot compute correspondences.")

            poses_i = poses_i.to(renderer_device, dtype=tensor_dtype)
            mesh_i = mesh_batch[mesh_idx].to(renderer_device)
            mesh_i = mesh_i.extend(poses_i.shape[0])
            render_out = renderer(
                gt_mesh=mesh_i,
                camera_poses=poses_i,
                return_point_maps=True,
            )

            if not isinstance(render_out, tuple) or len(render_out) != 3:
                raise RuntimeError("Renderer did not return point maps as expected.")

            _, point_maps, valid_masks = render_out

            point_maps = point_maps.permute(0, 2, 3, 1).contiguous()  # [S, H, W, 3]
            valid_masks = valid_masks.squeeze(1).contiguous()  # [S, H, W]

            point_maps_list.append(point_maps.to(dtype=tensor_dtype).cpu())
            valid_masks_list.append(valid_masks.cpu())

    point_maps_batch = torch.stack(point_maps_list, dim=0).to(output_device)
    valid_masks_batch = torch.stack(valid_masks_list, dim=0).to(output_device)
    return point_maps_batch, valid_masks_batch
