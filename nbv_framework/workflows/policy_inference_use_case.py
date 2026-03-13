"""Policy inference use case."""

from __future__ import annotations

from typing import Optional

import torch

from nbv_framework.dto import PolicyInferenceResult
from nbv_framework.geometry.pose_ops import (
    compute_policy_pose,
    compute_pose_for_across_views_in_ref_view,
)
from nbv_framework.ports import PolicyNetworkPort, SceneEncoderPort


class PolicyInferenceUseCase:
    def __init__(
        self,
        *,
        scene_encoder: SceneEncoderPort,
        policy_network: PolicyNetworkPort,
    ) -> None:
        self.scene_encoder = scene_encoder
        self.policy_network = policy_network

    def infer_next_pose(
        self,
        *,
        initial_images: torch.Tensor,
        camera_poses_batch: torch.Tensor,
        depth_z_batch: Optional[torch.Tensor],
    ) -> PolicyInferenceResult:
        scene_feature_batch = self.scene_encoder.extract_scene_features(
            initial_images,
            camera_poses_batch,
            depth_z=depth_z_batch,
        )

        camera_poses_batch_across_views = compute_pose_for_across_views_in_ref_view(
            scene_feature_batch.views
        )
        policy_output = self.policy_network(
            scene_feature_batch.features,
            camera_poses_batch_across_views,
        )

        next_camera_pose, predicted_relative_position, _ = compute_policy_pose(
            policy_output,
            camera_poses_batch,
        )
        return PolicyInferenceResult(
            next_camera_pose=next_camera_pose,
            predicted_relative_position=predicted_relative_position,
        )
