"""Main training-step orchestration use case."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from nbv_framework.dto import PolicyInferenceResult, PoseEvaluationResult, PreparedBatch


class TrainingStepUseCase:
    def __init__(
        self,
        *,
        batch_preparation,
        policy_inference,
        candidate_evaluation,
    ) -> None:
        self.batch_preparation = batch_preparation
        self.policy_inference = policy_inference
        self.candidate_evaluation = candidate_evaluation

    def run_step(
        self,
        batch: Dict,
        *,
        training: bool,
        point_cloud_dir: Optional[str],
        on_new_point_maps=None,
    ) -> Tuple[torch.Tensor, Dict[str, float], PreparedBatch, PolicyInferenceResult, PoseEvaluationResult]:
        prepared = self.batch_preparation.prepare(batch, randomize=training)
        policy_inference = self.policy_inference.infer_next_pose(
            initial_images=prepared.initial_images,
            camera_poses_batch=prepared.camera_poses,
            depth_z_batch=prepared.depth_z,
        )
        policy_eval = self.candidate_evaluation.evaluate_candidate_pose(
            pose=policy_inference.next_camera_pose,
            initial_images=prepared.initial_images,
            camera_poses_batch=prepared.camera_poses,
            gt_mesh_data=prepared.trimmed_gt_mesh_data,
            mesh_batch=prepared.mesh_batch,
            point_cloud_dir=point_cloud_dir,
            on_new_point_maps=on_new_point_maps,
        )

        loss_dict = self._build_loss_dict(
            policy_eval.loss_components,
            prepared.active_view_count,
        )
        return policy_eval.total_loss, loss_dict, prepared, policy_inference, policy_eval

    @staticmethod
    def _build_loss_dict(
        loss_components: Dict[str, float],
        active_view_count: int,
    ) -> Dict[str, float]:
        logged_loss_keys = (
            "total_loss",
            "weighted_chamfer_loss",
            "weighted_pose_penalty_loss",
        )
        loss_dict = {
            key: loss_components[key] for key in logged_loss_keys if key in loss_components
        }
        loss_dict["num_initial_views"] = float(active_view_count)
        return loss_dict

