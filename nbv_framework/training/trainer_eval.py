"""Evaluation helpers for NBVTrainer."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from ..pipeline.step_ops import evaluate_candidate_pose
from ..pipeline.types import PoseEvaluationResult


class NBVTrainerEvalMixin:
    """Evaluation helpers for training/validation/test."""

    def _build_loss_dict(
        self,
        loss_components: Dict[str, float],
        active_view_count: int,
    ) -> Dict[str, float]:
        logged_loss_keys = (
            "total_loss",
            # "chamfer_loss",
            "weighted_chamfer_loss",
            # "chamfer_pred_points_mean",
            # "chamfer_pred_points_min",
            # "chamfer_pred_points_zero_frac",
            # "chamfer_pred_points_last_view_mean",
            # "chamfer_pred_points_last_view_min",
            # "chamfer_pred_points_last_view_zero_frac",
            # "confidence_loss",
            # "weighted_confidence_loss",
            # "viewpoint_loss",
            # "weighted_viewpoint_loss",
            # "pose_penalty_loss",
            "weighted_pose_penalty_loss",
        )
        loss_dict = {
            key: loss_components[key] for key in logged_loss_keys if key in loss_components
        }
        # loss_dict["num_initial_views"] = float(active_view_count)
        return loss_dict

    def _evaluate_candidate_pose(
        self,
        *,
        pose: torch.Tensor,
        initial_images: torch.Tensor,
        camera_poses_batch: torch.Tensor,
        gt_mesh_data: Dict[str, torch.Tensor],
        mesh_batch,
        point_cloud_dir: Optional[str],
        mesh_paths: Optional[Sequence[Optional[str]]] = None,
    ) -> PoseEvaluationResult:
        return evaluate_candidate_pose(
            renderer=self.renderer,
            loss_fn=self.loss_fn,
            pose=pose,
            initial_images=initial_images,
            camera_poses_batch=camera_poses_batch,
            gt_mesh_data=gt_mesh_data,
            mesh_batch=mesh_batch,
            point_cloud_dir=point_cloud_dir,
            on_new_point_maps=self._set_last_new_point_maps_render,
        )
