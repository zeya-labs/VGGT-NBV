"""PyTorch3D renderer implementation of RendererPort."""

from __future__ import annotations

from typing import Dict

import torch

from nbv_framework.infrastructure.rendering.differentiable_renderer import DifferentiableRenderer


class PyTorch3DRendererAdapter:
    def __init__(self, renderer: DifferentiableRenderer) -> None:
        self.renderer = renderer

    def render_views(
        self,
        *,
        mesh_batch,
        camera_poses: torch.Tensor,
        out_rgb: bool,
        out_points: bool,
        out_mask: bool,
        out_depth: bool,
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(mesh_batch)

        if camera_poses.dim() == 2:
            if camera_poses.shape[0] == batch_size:
                camera_poses = camera_poses.unsqueeze(1)
            elif batch_size == 1:
                camera_poses = camera_poses.unsqueeze(0)
            else:
                raise ValueError(
                    f"Ambiguous camera_poses shape: {camera_poses.shape} for batch size {batch_size}"
                )

        if camera_poses.dim() != 3 or camera_poses.shape[-1] != 7:
            raise ValueError(f"camera_poses must have shape [B, S, 7], got {camera_poses.shape}")

        B, S, _ = camera_poses.shape
        device = mesh_batch.device

        cameras_flat = camera_poses.reshape(B * S, -1)
        mesh_indices = torch.arange(B, device=device).repeat_interleave(S)
        mesh_batch_flat = mesh_batch[mesh_indices]

        render_out = self.renderer(
            gt_mesh=mesh_batch_flat,
            camera_poses=cameras_flat,
            out_rgb=out_rgb,
            out_points=out_points,
            out_mask=out_mask,
            out_depth=out_depth,
        )

        outputs: Dict[str, torch.Tensor] = {}
        if not render_out:
            return outputs

        sample = next(iter(render_out.values()))
        height, width = sample.shape[-2], sample.shape[-1]

        if out_rgb:
            outputs["rgb"] = render_out["rgb"].view(B, S, 3, height, width)
        if out_points:
            points = render_out["points"].permute(0, 2, 3, 1).view(B, S, height, width, 3)
            outputs["points"] = points
        if out_mask:
            mask = render_out["mask"]
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            outputs["mask"] = mask.view(B, S, height, width)
        if out_depth:
            depth = render_out["depth"]
            if depth.dim() != 4:
                raise ValueError(f"Expected depth output shape [N, 1, H, W], got {tuple(depth.shape)}")
            depth = depth.view(B, S, 1, height, width).permute(0, 1, 3, 4, 2).contiguous()
            outputs["depth"] = depth

        return outputs

    def render_candidate(
        self,
        *,
        mesh_batch,
        pose: torch.Tensor,
        out_rgb: bool,
        out_points: bool,
        out_mask: bool,
        out_depth: bool,
    ) -> Dict[str, torch.Tensor]:
        return self.renderer(
            gt_mesh=mesh_batch,
            camera_poses=pose,
            out_depth=out_depth,
            out_points=out_points,
            out_mask=out_mask,
            out_rgb=out_rgb,
        )
