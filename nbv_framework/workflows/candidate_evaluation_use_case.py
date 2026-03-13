"""Candidate-pose evaluation use case."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

from nbv_framework.dto import PoseEvaluationResult
from nbv_framework.ports import LossPort, RendererPort, SceneEncoderPort
from nbv_framework.reconstruction.service import (
    build_recon_from_depth_z,
    build_recon_from_point_maps,
)

_SUPPORTED_RECONSTRUCTION_MODES = {"scene_encoder", "point_maps", "depth_z"}


class CandidateEvaluationUseCase:
    def __init__(
        self,
        *,
        renderer: RendererPort,
        loss: LossPort,
        scene_encoder: SceneEncoderPort,
        reconstruction_mode: str = "scene_encoder",
        depth_z_detach: bool = False,
    ) -> None:
        reconstruction_mode_normalized = str(reconstruction_mode).lower().strip()
        if reconstruction_mode_normalized not in _SUPPORTED_RECONSTRUCTION_MODES:
            raise ValueError(
                "Unsupported candidate reconstruction mode "
                f"{reconstruction_mode!r}. Expected one of: "
                f"{', '.join(sorted(_SUPPORTED_RECONSTRUCTION_MODES))}"
            )
        self.renderer = renderer
        self.loss = loss
        self.scene_encoder = scene_encoder
        self.reconstruction_mode = reconstruction_mode_normalized
        self.depth_z_detach = bool(depth_z_detach)

    def evaluate_candidate_pose(
        self,
        *,
        pose: torch.Tensor,
        initial_images: torch.Tensor,
        camera_poses_batch: torch.Tensor,
        gt_mesh_data: Dict[str, torch.Tensor],
        mesh_batch,
        point_cloud_dir: Optional[str],
        on_new_point_maps: Optional[Callable[[Optional[torch.Tensor]], None]] = None,
    ) -> PoseEvaluationResult:
        gt_point_maps = gt_mesh_data.get("gt_point_maps")
        gt_valid_masks = gt_mesh_data.get("gt_valid_masks")
        if gt_point_maps is None or gt_valid_masks is None:
            raise RuntimeError("gt_point_maps and gt_valid_masks are required for evaluation.")

        new_render = self.renderer.render_candidate(
            mesh_batch=mesh_batch,
            pose=pose,
            out_rgb=True,
            out_points=True,
            out_mask=True,
            out_depth=True,
        )
        new_images = new_render.rgb
        new_depth_z = new_render.depth
        new_point_maps = new_render.points
        new_valid_masks = new_render.mask
        if (
            new_images is None
            or new_depth_z is None
            or new_point_maps is None
            or new_valid_masks is None
        ):
            raise RuntimeError("Renderer returned incomplete candidate outputs for pose evaluation.")

        new_point_maps_render = new_point_maps.unsqueeze(1)
        if new_valid_masks.dim() == gt_valid_masks.dim() - 1:
            new_valid_masks = new_valid_masks.unsqueeze(1)
        if new_valid_masks.shape[0] != gt_valid_masks.shape[0] or new_valid_masks.shape[-2:] != gt_valid_masks.shape[-2:]:
            raise RuntimeError(
                "Renderer returned candidate mask with incompatible shape. "
                f"Expected batch/spatial dims matching {tuple(gt_valid_masks.shape)}, "
                f"got {tuple(new_valid_masks.shape)}."
            )

        if on_new_point_maps is not None:
            try:
                on_new_point_maps(new_point_maps_render)
            except RuntimeError:
                on_new_point_maps(None)

        updated_point_maps = torch.cat([gt_point_maps, new_point_maps_render], dim=1).contiguous()
        updated_valid_masks = torch.cat([gt_valid_masks, new_valid_masks], dim=1).contiguous()

        updated_gt_mesh_data = dict(gt_mesh_data)
        updated_gt_mesh_data["gt_point_maps"] = updated_point_maps
        updated_gt_mesh_data["gt_valid_masks"] = updated_valid_masks

        depth_z_batch = gt_mesh_data.get("depth_z")
        updated_depth_z = None
        if depth_z_batch is not None:
            if depth_z_batch.dim() == 4 and new_depth_z.dim() == 4 and new_depth_z.shape[-1] == 1:
                next_depth = new_depth_z.squeeze(-1).unsqueeze(1)
            else:
                next_depth = new_depth_z.unsqueeze(1)
            updated_depth_z = torch.cat([depth_z_batch, next_depth], dim=1).contiguous()
            updated_gt_mesh_data["depth_z"] = updated_depth_z

        combined_images_batch = torch.cat([initial_images, new_images.unsqueeze(1)], dim=1)
        combined_camera_poses = torch.cat([camera_poses_batch, pose.unsqueeze(1)], dim=1)

        recon_data = self._build_reconstruction_data(
            combined_images_batch=combined_images_batch,
            combined_camera_poses=combined_camera_poses,
            updated_point_maps=updated_point_maps,
            updated_valid_masks=updated_valid_masks,
            updated_depth_z=updated_depth_z,
        )

        total_loss, loss_components = self.loss.compute_loss(
            recon_data,
            updated_gt_mesh_data,
            combined_images_batch,
            combined_camera_poses,
        )

        if point_cloud_dir is not None:
            self.loss.export_point_clouds(
                recon_data,
                updated_gt_mesh_data,
                combined_images_batch,
                point_cloud_dir=point_cloud_dir,
            )

        return PoseEvaluationResult(
            total_loss=total_loss,
            loss_components=loss_components,
            new_images=new_images,
            gt_mesh_data=updated_gt_mesh_data,
            depth_z=updated_depth_z,
        )

    def _build_reconstruction_data(
        self,
        *,
        combined_images_batch: torch.Tensor,
        combined_camera_poses: torch.Tensor,
        updated_point_maps: torch.Tensor,
        updated_valid_masks: torch.Tensor,
        updated_depth_z: Optional[torch.Tensor],
    ):
        if self.reconstruction_mode == "point_maps":
            return build_recon_from_point_maps(
                point_maps=updated_point_maps,
                valid_masks=updated_valid_masks,
            )

        if self.reconstruction_mode == "depth_z":
            if updated_depth_z is None:
                raise RuntimeError(
                    "candidate reconstruction mode `depth_z` requires `gt_mesh_data['depth_z']` "
                    "to be available for all input views."
                )
            depth_z_input = updated_depth_z.detach() if self.depth_z_detach else updated_depth_z
            return build_recon_from_depth_z(
                camera_poses=combined_camera_poses,
                depth_z=depth_z_input,
                valid_masks=updated_valid_masks,
            )

        return self.scene_encoder.reconstruct_and_evaluate(
            combined_images_batch,
            combined_camera_poses,
            depth_z=updated_depth_z,
        )
