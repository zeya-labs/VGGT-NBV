from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from nbv_framework.application.use_cases.candidate_evaluation_use_case import (
    CandidateEvaluationUseCase,
)
from nbv_framework.domain.services import (
    ReconstructionData,
    build_recon_from_depth_z,
    build_recon_from_point_maps,
)


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


class _GradRenderer(_Renderer):
    def __init__(self) -> None:
        self.last_depth: Optional[torch.Tensor] = None

    def render_candidate(self, **kwargs):
        outputs = super().render_candidate(**kwargs)
        depth = outputs["depth"].clone().requires_grad_(True)
        depth.retain_grad()
        self.last_depth = depth
        outputs["depth"] = depth
        return outputs


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


class _GradLoss(_Loss):
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
        total_loss = recon_data.recon_world_points.sum()
        return total_loss, {"total_loss": float(total_loss.detach())}


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


def test_candidate_evaluation_can_use_point_map_reconstruction() -> None:
    batch_size, num_views, height, width = 1, 2, 4, 4
    initial_images = torch.zeros(batch_size, num_views, 3, height, width)
    camera_poses = torch.zeros(batch_size, num_views, 7)
    candidate_pose = torch.zeros(batch_size, 7)

    gt_mesh_data = {
        "gt_point_maps": torch.full((batch_size, num_views, height, width, 3), 0.3, dtype=torch.float32),
        "gt_valid_masks": torch.ones(batch_size, num_views, height, width, dtype=torch.bool),
        "depth_z": torch.ones(batch_size, num_views, height, width, 1),
    }

    scene_encoder = _SceneEncoder(
        ReconstructionData(
            recon_world_points=torch.zeros(batch_size, num_views + 1, height, width, 3),
            recon_conf=torch.zeros(batch_size, num_views + 1, height, width),
            recon_mask=torch.zeros(batch_size, num_views + 1, height, width, dtype=torch.bool),
        )
    )
    loss = _Loss()
    use_case = CandidateEvaluationUseCase(
        renderer=_Renderer(),
        loss=loss,
        scene_encoder=scene_encoder,
        reconstruction_mode="point_maps",
    )

    use_case.evaluate_candidate_pose(
        pose=candidate_pose,
        initial_images=initial_images,
        camera_poses_batch=camera_poses,
        gt_mesh_data=gt_mesh_data,
        mesh_batch=torch.zeros(batch_size, 1),
        point_cloud_dir=None,
    )

    assert len(scene_encoder.calls) == 0
    used_recon, _, _, _ = loss.compute_calls[0]

    expected_recon = build_recon_from_point_maps(
        point_maps=torch.cat(
            [
                gt_mesh_data["gt_point_maps"],
                torch.full((batch_size, 1, height, width, 3), 0.1, dtype=torch.float32),
            ],
            dim=1,
        ),
        valid_masks=torch.ones(batch_size, num_views + 1, height, width, dtype=torch.bool),
    )

    assert torch.allclose(used_recon.recon_world_points, expected_recon.recon_world_points)
    assert torch.equal(used_recon.recon_mask, expected_recon.recon_mask)
    assert torch.equal(used_recon.recon_conf, expected_recon.recon_conf)


def test_candidate_evaluation_can_use_depth_z_reconstruction() -> None:
    batch_size, num_views, height, width = 1, 2, 4, 4
    initial_images = torch.zeros(batch_size, num_views, 3, height, width)
    camera_poses = torch.zeros(batch_size, num_views, 7)
    camera_poses[..., -1] = 1.0
    candidate_pose = torch.zeros(batch_size, 7)
    candidate_pose[..., -1] = 1.0

    gt_mesh_data = {
        "gt_point_maps": torch.zeros(batch_size, num_views, height, width, 3),
        "gt_valid_masks": torch.ones(batch_size, num_views, height, width, dtype=torch.bool),
        "depth_z": torch.ones(batch_size, num_views, height, width, 1),
    }

    scene_encoder = _SceneEncoder(
        ReconstructionData(
            recon_world_points=torch.zeros(batch_size, num_views + 1, height, width, 3),
            recon_conf=torch.zeros(batch_size, num_views + 1, height, width),
            recon_mask=torch.zeros(batch_size, num_views + 1, height, width, dtype=torch.bool),
        )
    )
    loss = _Loss()
    use_case = CandidateEvaluationUseCase(
        renderer=_Renderer(),
        loss=loss,
        scene_encoder=scene_encoder,
        reconstruction_mode="depth_z",
    )

    use_case.evaluate_candidate_pose(
        pose=candidate_pose,
        initial_images=initial_images,
        camera_poses_batch=camera_poses,
        gt_mesh_data=gt_mesh_data,
        mesh_batch=torch.zeros(batch_size, 1),
        point_cloud_dir=None,
    )

    assert len(scene_encoder.calls) == 0
    used_recon, _, _, _ = loss.compute_calls[0]

    expected_depth_z = torch.cat(
        [
            gt_mesh_data["depth_z"],
            torch.full((batch_size, 1, height, width, 1), 0.6, dtype=torch.float32),
        ],
        dim=1,
    )
    expected_recon = build_recon_from_depth_z(
        camera_poses=torch.cat([camera_poses, candidate_pose.unsqueeze(1)], dim=1),
        depth_z=expected_depth_z,
        valid_masks=torch.ones(batch_size, num_views + 1, height, width, dtype=torch.bool),
    )

    assert torch.allclose(used_recon.recon_world_points, expected_recon.recon_world_points)
    assert torch.equal(used_recon.recon_mask, expected_recon.recon_mask)
    assert torch.equal(used_recon.recon_conf, expected_recon.recon_conf)


def test_candidate_evaluation_rejects_unknown_reconstruction_mode() -> None:
    scene_encoder = _SceneEncoder(
        ReconstructionData(
            recon_world_points=torch.zeros(1, 1, 1, 1, 3),
            recon_conf=torch.zeros(1, 1, 1, 1),
            recon_mask=torch.zeros(1, 1, 1, 1, dtype=torch.bool),
        )
    )

    try:
        CandidateEvaluationUseCase(
            renderer=_Renderer(),
            loss=_Loss(),
            scene_encoder=scene_encoder,
            reconstruction_mode="unknown",
        )
    except ValueError as exc:
        assert "Unsupported candidate reconstruction mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown reconstruction mode")


def test_candidate_evaluation_depth_z_detach_only_cuts_depth_gradient() -> None:
    batch_size, num_views, height, width = 1, 2, 4, 4
    initial_images = torch.zeros(batch_size, num_views, 3, height, width)
    camera_poses = torch.zeros(batch_size, num_views, 7)
    camera_poses[..., -1] = 1.0
    candidate_pose = torch.zeros(batch_size, 7)
    candidate_pose[..., -1] = 1.0
    candidate_pose.requires_grad_()

    def _run(depth_z_detach: bool):
        gt_depth_z = torch.ones(batch_size, num_views, height, width, 1, requires_grad=True)
        gt_mesh_data = {
            "gt_point_maps": torch.zeros(batch_size, num_views, height, width, 3),
            "gt_valid_masks": torch.ones(batch_size, num_views, height, width, dtype=torch.bool),
            "depth_z": gt_depth_z,
        }
        renderer = _GradRenderer()
        scene_encoder = _SceneEncoder(
            ReconstructionData(
                recon_world_points=torch.zeros(batch_size, num_views + 1, height, width, 3),
                recon_conf=torch.zeros(batch_size, num_views + 1, height, width),
                recon_mask=torch.zeros(batch_size, num_views + 1, height, width, dtype=torch.bool),
            )
        )
        loss = _GradLoss()
        use_case = CandidateEvaluationUseCase(
            renderer=renderer,
            loss=loss,
            scene_encoder=scene_encoder,
            reconstruction_mode="depth_z",
            depth_z_detach=depth_z_detach,
        )
        if candidate_pose.grad is not None:
            candidate_pose.grad.zero_()

        result = use_case.evaluate_candidate_pose(
            pose=candidate_pose,
            initial_images=initial_images,
            camera_poses_batch=camera_poses,
            gt_mesh_data=gt_mesh_data,
            mesh_batch=torch.zeros(batch_size, 1),
            point_cloud_dir=None,
        )
        result.total_loss.backward()

        return {
            "candidate_pose_grad": candidate_pose.grad.detach().clone(),
            "gt_depth_grad": None if gt_depth_z.grad is None else gt_depth_z.grad.detach().clone(),
            "new_depth_grad": None
            if renderer.last_depth is None or renderer.last_depth.grad is None
            else renderer.last_depth.grad.detach().clone(),
        }

    without_detach = _run(depth_z_detach=False)
    with_detach = _run(depth_z_detach=True)

    assert without_detach["candidate_pose_grad"].abs().sum() > 0
    assert with_detach["candidate_pose_grad"].abs().sum() > 0

    assert without_detach["gt_depth_grad"] is not None
    assert without_detach["gt_depth_grad"].abs().sum() > 0
    assert without_detach["new_depth_grad"] is not None
    assert without_detach["new_depth_grad"].abs().sum() > 0

    assert with_detach["gt_depth_grad"] is None
    assert with_detach["new_depth_grad"] is None
