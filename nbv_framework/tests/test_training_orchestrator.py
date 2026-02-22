from __future__ import annotations

import torch

from nbv_framework.application.dto import PolicyInferenceResult, PoseEvaluationResult, PreparedBatch
from nbv_framework.application.use_cases.training_step_use_case import TrainingStepUseCase


class _BatchPreparation:
    def prepare(self, batch, *, randomize: bool):
        _ = batch
        _ = randomize
        return PreparedBatch(
            initial_images=torch.zeros(1, 2, 3, 4, 4),
            camera_poses=torch.zeros(1, 2, 7),
            depth_z=None,
            gt_mesh_data={"gt_points": torch.zeros(1, 8, 3)},
            trimmed_gt_mesh_data={"gt_points": torch.zeros(1, 8, 3)},
            mesh_batch=torch.zeros(1, 1),
            mesh_paths=None,
            selection=torch.tensor([0, 1]),
            active_view_count=2,
        )


class _PolicyInference:
    def infer_next_pose(self, **kwargs):
        _ = kwargs
        return PolicyInferenceResult(
            next_camera_pose=torch.zeros(1, 7),
            predicted_relative_position=torch.zeros(1, 3),
        )


class _CandidateEvaluation:
    def evaluate_candidate_pose(self, **kwargs):
        _ = kwargs
        return PoseEvaluationResult(
            total_loss=torch.tensor(1.25),
            loss_components={
                "total_loss": 1.25,
                "weighted_chamfer_loss": 0.5,
                "weighted_pose_penalty_loss": 0.2,
            },
            new_images=torch.zeros(1, 3, 4, 4),
            gt_mesh_data={"gt_points": torch.zeros(1, 8, 3)},
            depth_z=None,
        )


def test_training_orchestrator_builds_loss_dict() -> None:
    orchestrator = TrainingStepUseCase(
        batch_preparation=_BatchPreparation(),
        policy_inference=_PolicyInference(),
        candidate_evaluation=_CandidateEvaluation(),
    )

    loss, loss_dict, prepared, policy_inf, policy_eval = orchestrator.run_step(
        {},
        training=True,
        point_cloud_dir=None,
    )

    assert float(loss) == 1.25
    assert "total_loss" in loss_dict
    assert "weighted_chamfer_loss" in loss_dict
    assert loss_dict["num_initial_views"] == 2.0
    assert prepared.active_view_count == 2
    assert policy_inf.next_camera_pose.shape == (1, 7)
    assert policy_eval.new_images.shape == (1, 3, 4, 4)
