"""Candidate-pose evaluation service."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

from nbv_framework.application.contracts import PoseEvaluationResult
from nbv_framework.domain.models.direct_reconstruction import build_recon_from_point_maps
from nbv_framework.application.ports import LossPort, RendererPort


class CandidateEvaluationService:
    def __init__(self, *, renderer: RendererPort, loss: LossPort) -> None:
        self.renderer = renderer
        self.loss = loss

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
        new_images = new_render["rgb"]
        new_depth_z = new_render["depth"]
        new_point_maps_render = new_render["points"].permute(0, 2, 3, 1).unsqueeze(1)
        new_valid_masks = new_render["mask"]

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
            updated_depth_z = torch.cat([depth_z_batch, new_depth_z.unsqueeze(-1)], dim=1).contiguous()
            updated_gt_mesh_data["depth_z"] = updated_depth_z

        combined_images_batch = torch.cat([initial_images, new_images.unsqueeze(1)], dim=1)
        combined_camera_poses = torch.cat([camera_poses_batch, pose.unsqueeze(1)], dim=1)

        recon_data = build_recon_from_point_maps(
            point_maps=updated_point_maps,
            camera_poses=combined_camera_poses,
            valid_masks=updated_valid_masks,
            depth_z=updated_depth_z,
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
