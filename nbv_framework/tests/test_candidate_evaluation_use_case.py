from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from nbv_framework.application.use_cases.candidate_evaluation_use_case import (
    CandidateEvaluationUseCase,
)
from nbv_framework.domain.services import ReconstructionData


class _Renderer:
    def render_candidate(self, **kwargs):
        _ = kwargs
        batch_size, height, width = 1, 4, 4
        return {
            "rgb": torch.full((batch_size, 3, height, width), 0.25, dtype=torch.float32),
            "depth": torch.full((batch_size, 1, height, width), 0.6, dtype=torch.float32),
            "points": torch.full((batch_size, 3, height, width), 0.1, dtype=torch.float32),
            "mask": torch.ones((batch_size, 1, height, width), dtype=torch.bool),
        }


class _SceneEncoder:
    def __init__(self, recon_data: ReconstructionData) -> None:
        self.recon_data = recon_data
        self.calls = []

    def reconstruct_and_evaluate(
        self,
        images: torch.Tensor,
        camera_poses: torch.Tensor,
        *,
        depth_z: Optional[torch.Tensor] = None,
    ) -> ReconstructionData:
        self.calls.append((images, camera_poses, depth_z))
        return self.recon_data


class _Loss:
    def __init__(self) -> None:
        self.compute_calls = []
        self.export_calls = []

    def compute_loss(
        self,
        recon_data: ReconstructionData,
        gt_data: Dict[str, torch.Tensor],
        combined_images_batch: Optional[torch.Tensor],
        combined_camera_poses: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self.compute_calls.append(
            (recon_data, gt_data, combined_images_batch, combined_camera_poses)
        )
        return torch.tensor(2.5), {"total_loss": 2.5}

    def export_point_clouds(
        self,
        recon_data: ReconstructionData,
        gt_data: Dict[str, torch.Tensor],
        combined_images_batch: Optional[torch.Tensor],
        *,
        point_cloud_dir: Optional[str],
    ) -> None:
        self.export_calls.append((recon_data, gt_data, combined_images_batch, point_cloud_dir))


def test_candidate_evaluation_uses_scene_encoder_reconstruction() -> None:
    batch_size, num_views, height, width = 1, 2, 4, 4
    initial_images = torch.zeros(batch_size, num_views, 3, height, width)
    camera_poses = torch.zeros(batch_size, num_views, 7)
    candidate_pose = torch.zeros(batch_size, 7)

    gt_mesh_data = {
        "gt_point_maps": torch.zeros(batch_size, num_views, height, width, 3),
        "gt_valid_masks": torch.ones(batch_size, num_views, height, width, dtype=torch.bool),
        "depth_z": torch.ones(batch_size, num_views, height, width, 1),
    }

    expected_recon = ReconstructionData(
        recon_world_points=torch.ones(batch_size, num_views + 1, height, width, 3),
        recon_conf=torch.ones(batch_size, num_views + 1, height, width),
        recon_mask=torch.ones(batch_size, num_views + 1, height, width, dtype=torch.bool),
    )
    scene_encoder = _SceneEncoder(expected_recon)
    loss = _Loss()
    use_case = CandidateEvaluationUseCase(
        renderer=_Renderer(),
        loss=loss,
        scene_encoder=scene_encoder,
    )

    result = use_case.evaluate_candidate_pose(
        pose=candidate_pose,
        initial_images=initial_images,
        camera_poses_batch=camera_poses,
        gt_mesh_data=gt_mesh_data,
        mesh_batch=torch.zeros(batch_size, 1),
        point_cloud_dir="debug-point-clouds",
    )

    assert len(scene_encoder.calls) == 1
    combined_images, combined_camera_poses, combined_depth_z = scene_encoder.calls[0]
    assert combined_images.shape == (batch_size, num_views + 1, 3, height, width)
    assert combined_camera_poses.shape == (batch_size, num_views + 1, 7)
    assert combined_depth_z is not None
    assert combined_depth_z.shape == (batch_size, num_views + 1, height, width, 1)

    assert len(loss.compute_calls) == 1
    used_recon, _, _, _ = loss.compute_calls[0]
    assert used_recon is expected_recon

    assert len(loss.export_calls) == 1
    export_recon, _, _, export_dir = loss.export_calls[0]
    assert export_recon is expected_recon
    assert export_dir == "debug-point-clouds"

    assert float(result.total_loss) == 2.5
    assert result.loss_components["total_loss"] == 2.5

