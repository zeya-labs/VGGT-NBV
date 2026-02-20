"""Policy inference helpers for NBVTrainer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from ..geometry.pose_ops import (
    compute_pose_for_across_views_in_ref_view,
    compute_policy_pose,
    compute_pose_scale_factor,
)
from ..pipeline.types import PolicyInferenceOutput


class NBVTrainerPolicyMixin:
    """Policy inference and gradient tracking utilities."""

    def _extract_scene_features(
        self,
        initial_images: torch.Tensor,
        camera_poses_batch: torch.Tensor,
        depth_z_batch: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """封装 MapAnything scene feature 提取，便于复用和测试。"""
        return self.vggt_wrapper.extract_scene_features(
            initial_images,
            camera_poses_batch,
            is_metric_scale=False,
            depth_z=depth_z_batch,
        )

    def _compute_pose_for_across_views_in_ref_view(
        self,
        views: List[Dict[str, Any]],
    ) -> torch.Tensor:
        return compute_pose_for_across_views_in_ref_view(views)

    def _compute_policy_pose(
        self,
        policy_output: torch.Tensor,
        camera_poses_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return compute_policy_pose(policy_output, camera_poses_batch)

    def _compute_pose_scale_factor(self, camera_poses_batch: torch.Tensor) -> torch.Tensor:
        return compute_pose_scale_factor(camera_poses_batch)

    def _infer_next_pose(
        self,
        *,
        initial_images: torch.Tensor,
        camera_poses_batch: torch.Tensor,
        depth_z_batch: Optional[torch.Tensor],
    ) -> PolicyInferenceOutput:
        scene_features, views = self._extract_scene_features(
            initial_images,
            camera_poses_batch,
            depth_z_batch,
        )

        camera_poses_batch_across_views = self._compute_pose_for_across_views_in_ref_view(views)
        policy_output = self.policy_network(scene_features, camera_poses_batch_across_views)

        next_camera_pose, predicted_relative_position, _ = self._compute_policy_pose(
            policy_output,
            camera_poses_batch,
        )

        return PolicyInferenceOutput(
            next_camera_pose=next_camera_pose,
            predicted_relative_position=predicted_relative_position,
        )

    def _maybe_track_policy_gradients(
        self,
        predicted_relative_position: torch.Tensor,
        next_camera_pose: torch.Tensor,
    ) -> None:
        if not self.trainer.training:
            return
        try:
            if predicted_relative_position.requires_grad:
                def _capture_pred_rel_grad(grad: torch.Tensor) -> torch.Tensor:
                    if grad is not None:
                        self._last_predicted_relative_position_grad_norm = (
                            grad.norm(dim=-1).mean().detach()
                        )
                    return grad
                predicted_relative_position.register_hook(_capture_pred_rel_grad)

            if next_camera_pose.requires_grad:
                def _capture_next_pose_grad(grad: torch.Tensor) -> torch.Tensor:
                    if grad is not None and grad.numel() > 0:
                        grad = grad.detach()
                        self._last_next_pose_position_grad_norm = (
                            grad[:, :3].norm(dim=-1).mean()
                        )
                        self._last_next_pose_quaternion_grad_norm = (
                            grad[:, 3:].norm(dim=-1).mean()
                        )
                    return grad
                next_camera_pose.register_hook(_capture_next_pose_grad)
        except RuntimeError:
            self._last_predicted_relative_position_grad_norm = None
            self._last_next_pose_position_grad_norm = None
            self._last_next_pose_quaternion_grad_norm = None

    def _set_last_new_point_maps_render(self, value: Optional[torch.Tensor]) -> None:
        self._last_new_point_maps_grad_norm = None
        if value is None or not value.requires_grad:
            return
        try:
            def _capture_new_point_maps_grad(grad: torch.Tensor) -> torch.Tensor:
                if grad is not None and grad.numel() > 0:
                    self._last_new_point_maps_grad_norm = grad.norm(dim=-1).mean().detach()
                return grad
            value.register_hook(_capture_new_point_maps_grad)
        except RuntimeError:
            self._last_new_point_maps_grad_norm = None
