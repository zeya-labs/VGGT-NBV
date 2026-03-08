"""Domain-level camera pose tensor utilities."""

from __future__ import annotations

from typing import Optional

import torch
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import matrix_to_quaternion


def get_up_vector(up_axis: str) -> torch.Tensor:
    axis = up_axis.upper()
    vectors = {
        "X": torch.tensor([1.0, 0.0, 0.0]),
        "Y": torch.tensor([0.0, 1.0, 0.0]),
        "Z": torch.tensor([0.0, 0.0, 1.0]),
        "-X": torch.tensor([-1.0, 0.0, 0.0]),
        "-Y": torch.tensor([0.0, -1.0, 0.0]),
        "-Z": torch.tensor([0.0, 0.0, -1.0]),
    }
    if axis not in vectors:
        raise ValueError(f"Invalid up_axis: {up_axis}. Must be one of X, Y, Z, -X, -Y, -Z.")
    return vectors[axis]


def position_to_pose_tensor(
    positions: torch.Tensor,
    up_axis: str = "Y",
    look_at: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert camera positions ``[B,3]`` to pose tensors ``[B,7]`` with xyzw quaternion."""
    if not torch.is_tensor(positions):
        positions = torch.as_tensor(positions)

    if positions.ndim == 1:
        if positions.numel() != 3:
            raise ValueError(
                f"positions expects 3 values for a single camera, but got shape {tuple(positions.shape)}"
            )
        positions = positions.unsqueeze(0)
    elif positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape [B, 3], but received {tuple(positions.shape)}")

    if not positions.is_floating_point():
        raise TypeError("positions must be a floating point tensor.")

    output_dtype = positions.dtype
    compute_dtype = output_dtype if output_dtype in {torch.float32, torch.float64} else torch.float32
    batch_size = positions.shape[0]
    device = positions.device

    up = get_up_vector(up_axis).to(device=device, dtype=compute_dtype).unsqueeze(0).expand(batch_size, -1)

    if look_at is None:
        at = torch.zeros(batch_size, 3, dtype=compute_dtype, device=device)
    else:
        look_at_tensor = torch.as_tensor(look_at, dtype=compute_dtype, device=device)
        if look_at_tensor.ndim == 1:
            if look_at_tensor.numel() != 3:
                raise ValueError(
                    f"look_at expects 3 values for a single target, but got shape {tuple(look_at_tensor.shape)}"
                )
            look_at_tensor = look_at_tensor.unsqueeze(0)
        elif look_at_tensor.ndim != 2 or look_at_tensor.shape[1] != 3:
            raise ValueError(f"look_at must have shape [B, 3] or [3], but received {tuple(look_at_tensor.shape)}")

        if look_at_tensor.shape[0] == 1 and batch_size > 1:
            look_at_tensor = look_at_tensor.expand(batch_size, -1)
        elif look_at_tensor.shape[0] != batch_size:
            raise ValueError(
                f"look_at batch size ({look_at_tensor.shape[0]}) does not match positions batch size ({batch_size})."
            )
        at = look_at_tensor

    positions_compute = positions.to(dtype=compute_dtype)
    rotation, _ = look_at_view_transform(eye=positions_compute, at=at, up=up)
    rotation = rotation.to(device=device)

    quaternion_wxyz = matrix_to_quaternion(rotation)
    quaternion_xyzw = torch.stack(
        [
            quaternion_wxyz[:, 1],
            quaternion_wxyz[:, 2],
            quaternion_wxyz[:, 3],
            quaternion_wxyz[:, 0],
        ],
        dim=1,
    )

    return torch.cat(
        [positions_compute.to(dtype=output_dtype), quaternion_xyzw.to(dtype=output_dtype)],
        dim=1,
    ).to(device=device)


__all__ = ["get_up_vector", "position_to_pose_tensor"]
