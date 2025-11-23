"""Pose penalty regularizer for camera positions."""

from typing import Dict, Optional, Tuple

import torch


class PosePenalty:
    """Quadratic penalties to keep poses within a spherical shell and above a floor."""

    def __init__(
        self,
        weight: float = 0.02,
        up_axis: str = "Y",
        outer_radius: float = 4.0,
        inner_radius: float = 2.0,
        floor_margin: float = 1.0,
    ) -> None:
        self.weight = weight
        self.up_axis = up_axis.upper()
        if self.up_axis not in {"X", "Y", "Z"}:
            raise ValueError("pose_up_axis must be one of {'X', 'Y', 'Z' }.")
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.floor_margin = floor_margin

    def __call__(
        self,
        combined_camera_poses: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        zero = torch.zeros((), device=device, dtype=dtype)
        if self.weight <= 0 or combined_camera_poses is None:
            return zero, zero, {
                "pose_penalty_inner": zero,
                "pose_penalty_outer": zero,
                "pose_penalty_floor": zero,
            }

        if combined_camera_poses.dim() == 2:
            target_positions = combined_camera_poses[:, :3]
        else:
            target_positions = combined_camera_poses[:, -1, :3]

        axis_to_index = {"X": 0, "Y": 1, "Z": 2}
        up_axis_index = axis_to_index.get(self.up_axis, 1)

        outer_radius = max(float(self.outer_radius), 1e-3)
        inner_radius = max(float(self.inner_radius), 1e-3)
        floor_margin = max(float(self.floor_margin), 1e-3)

        distances = torch.linalg.norm(target_positions, ord=2, dim=-1)

        inner_violation = torch.relu(inner_radius - distances)
        inner_penalty = (inner_violation / inner_radius).pow(2).mean()

        outer_violation = torch.relu(distances - outer_radius)
        outer_penalty = (outer_violation / outer_radius).pow(2).mean()

        up_axis_values = target_positions[..., up_axis_index]
        floor_violation = torch.relu(-(up_axis_values + floor_margin))
        floor_penalty = (floor_violation / floor_margin).pow(2).mean()

        penalty_terms = {
            "pose_penalty_inner": inner_penalty,
            "pose_penalty_outer": outer_penalty,
            "pose_penalty_floor": floor_penalty,
        }

        penalty_value = torch.stack(list(penalty_terms.values())).sum()
        weighted_penalty = self.weight * penalty_value

        return weighted_penalty, penalty_value, penalty_terms


__all__ = ["PosePenalty"]
