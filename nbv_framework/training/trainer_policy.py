"""Policy inference helpers for NBVTrainer."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

    def _attach_gradient_metric_hooks(
        self,
        tensor: Optional[torch.Tensor],
        metrics: Sequence[Tuple[str, Callable[[torch.Tensor], torch.Tensor]]],
    ) -> None:
        trainer = self._trainer
        if trainer is None or not trainer.training:
            return
        if tensor is None or not tensor.requires_grad:
            return

        batch_size = self._get_log_batch_size()

        def _capture(grad: torch.Tensor) -> torch.Tensor:
            if grad is None or grad.numel() == 0:
                return grad
            detached_grad = grad.detach()
            for log_key, selector in metrics:
                selected_grad = selector(detached_grad)
                if selected_grad is None or selected_grad.numel() == 0:
                    continue
                self.log(
                    log_key,
                    selected_grad.float().norm(dim=-1).mean(),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    batch_size=batch_size,
                )
            return grad

        try:
            tensor.register_hook(_capture)
        except RuntimeError:
            return

    def _track_policy_gradients(
        self,
        predicted_relative_position: torch.Tensor,
        next_camera_pose: torch.Tensor,
    ) -> None:
        self._attach_gradient_metric_hooks(
            predicted_relative_position,
            (
                (
                    "gradients/predicted_relative_position_grad_norm",
                    lambda grad: grad,
                ),
            ),
        )
        self._attach_gradient_metric_hooks(
            next_camera_pose,
            (
                (
                    "gradients/next_pose_position_grad_norm",
                    lambda grad: grad[:, :3],
                ),
                (
                    "gradients/next_pose_quaternion_grad_norm",
                    lambda grad: grad[:, 3:],
                ),
            ),
        )

    def _track_new_point_maps_gradients(self, value: Optional[torch.Tensor]) -> None:
        self._attach_gradient_metric_hooks(
            value,
            (
                (
                    "gradients/new_point_maps_render_grad_norm",
                    lambda grad: grad,
                ),
            ),
        )
