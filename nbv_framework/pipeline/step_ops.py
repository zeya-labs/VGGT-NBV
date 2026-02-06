from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from pytorch3d.structures import Meshes

from .types import PoseEvaluationResult
from ..models.direct_reconstruction import build_recon_from_point_maps
from ..utils.camera_utils import normalize_depth_for_visualization, position_to_pose_tensor
from ..utils.render_utils import render_mesh_views


def render_inputs(
    *,
    renderer,
    initial_images: Optional[torch.Tensor],
    camera_poses_batch: torch.Tensor,
    gt_mesh_data: Dict[str, torch.Tensor],
    mesh_batch: Optional[Meshes],
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    needs_images = initial_images is None
    gt_point_maps = gt_mesh_data.get("gt_point_maps")
    gt_valid_masks = gt_mesh_data.get("gt_valid_masks")
    needs_points = gt_point_maps is None
    needs_depth = gt_mesh_data.get("depth_z") is None
    needs_mask = gt_valid_masks is None or (needs_depth and gt_valid_masks is None)

    if not (needs_images or needs_points or needs_depth or needs_mask):
        return initial_images, gt_mesh_data

    if (needs_images or needs_points or needs_depth or needs_mask) and mesh_batch is None:
        raise ValueError("mesh_batch is required to render initial views on GPU.")

    render_out = None
    if needs_images or needs_points or needs_depth or needs_mask:
        with torch.no_grad():
            render_out = render_mesh_views(
                renderer=renderer,
                mesh_batch=mesh_batch,
                camera_poses=camera_poses_batch,
                out_rgb=needs_images,
                out_points=needs_points,
                out_mask=needs_mask,
                out_depth=needs_depth,
            )

        if needs_images:
            initial_images = render_out.get("rgb")
            if initial_images is None:
                raise RuntimeError("Renderer did not return rgb output.")
            if initial_images.is_floating_point() and initial_images.dtype != dtype:
                initial_images = initial_images.to(dtype=dtype)

        if needs_points:
            gt_point_maps = render_out.get("points")
            if gt_point_maps is None:
                raise RuntimeError("Renderer did not return point maps.")
            gt_mesh_data["gt_point_maps"] = gt_point_maps

        if needs_mask:
            gt_valid_masks = render_out.get("mask")
            if gt_valid_masks is None:
                raise RuntimeError("Renderer did not return masks.")
            gt_valid_masks = gt_valid_masks.to(dtype=torch.bool)
            gt_mesh_data["gt_valid_masks"] = gt_valid_masks

    if needs_depth:
        depth_z = render_out.get("depth")
        if depth_z is None:
            raise RuntimeError("Renderer did not return depth output.")
        gt_mesh_data["depth_z"] = depth_z
        gt_mesh_data["depth_z_viz"] = normalize_depth_for_visualization(depth_z, gt_valid_masks)

    return initial_images, gt_mesh_data


def select_initial_views(
    initial_images: torch.Tensor,
    camera_poses: torch.Tensor,
    *,
    depth_z: Optional[torch.Tensor] = None,
    randomize: bool,
    min_initial_views: int,
    max_initial_views: int,
    randomize_initial_views: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, int]:
    min_views = max(min_initial_views, 1)
    max_views = min(max_initial_views, initial_images.shape[1])

    total_views = initial_images.shape[1]
    should_randomize = randomize and randomize_initial_views

    if should_randomize:
        sampled = torch.randint(
            low=min_views,
            high=max_views + 1,
            size=(1,),
            device=initial_images.device,
        )
        num_views = int(sampled.item())
    else:
        num_views = max_views

    if should_randomize:
        perm = torch.randperm(total_views, device=initial_images.device, dtype=torch.long)
    else:
        perm = torch.arange(total_views, device=initial_images.device, dtype=torch.long)
    selection = perm[:num_views]
    selection, _ = torch.sort(selection)
    initial_images = initial_images.index_select(1, selection)
    camera_poses = camera_poses.index_select(1, selection)
    if depth_z is not None:
        depth_z = depth_z.index_select(1, selection)

    return initial_images, camera_poses, depth_z, selection, num_views


def evaluate_candidate_pose(
    *,
    renderer,
    loss_fn,
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

    new_render = renderer(
        gt_mesh=mesh_batch,
        camera_poses=pose,
        out_depth=True,
        out_points=True,
        out_mask=True,
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

    total_loss, loss_components = loss_fn(
        recon_data,
        updated_gt_mesh_data,
        combined_images_batch,
        combined_camera_poses,
        return_components=True,
        point_cloud_dir=None,
    )

    return PoseEvaluationResult(
        total_loss=total_loss,
        loss_components=loss_components,
        new_images=new_images,
        gt_mesh_data=updated_gt_mesh_data,
        depth_z=updated_depth_z,
    )


def sample_random_positions(
    *,
    batch_size: int,
    device: torch.device,
    loss_fn,
) -> torch.Tensor:
    inner_radius = float(getattr(loss_fn, "pose_inner_radius", 1.5))
    outer_radius = float(getattr(loss_fn, "pose_outer_radius", inner_radius + 1.0))

    floor_margin = float(getattr(loss_fn, "pose_floor_margin", 1.0))
    up_axis = getattr(loss_fn, "pose_up_axis", "Y").upper()
    axis_index = {"X": 0, "Y": 1, "Z": 2}.get(up_axis, 1)
    min_height = -floor_margin

    dtype = torch.float32
    positions = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    filled = 0
    attempts = 0
    while filled < batch_size and attempts < 20:
        remaining = batch_size - filled
        sample_count = max(remaining * 2, 4)
        directions = torch.randn(sample_count, 3, device=device, dtype=dtype)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        radii = torch.rand(sample_count, 1, device=device, dtype=dtype)
        radii = radii * (outer_radius - inner_radius) + inner_radius
        samples = directions * radii
        valid_mask = samples[:, axis_index] >= min_height
        valid_samples = samples[valid_mask]
        if valid_samples.numel() == 0:
            attempts += 1
            continue
        take = min(valid_samples.size(0), remaining)
        positions[filled:filled + take] = valid_samples[:take]
        filled += take
        attempts += 1

    if filled < batch_size:
        fallback = torch.randn(batch_size - filled, 3, device=device, dtype=dtype)
        fallback = fallback / fallback.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        radius = (inner_radius + outer_radius) * 0.5
        fallback = fallback * radius
        fallback[:, axis_index] = torch.clamp(fallback[:, axis_index], min=min_height + 1e-4)
        positions[filled:] = fallback

    return positions


def compute_random_baseline(
    *,
    renderer,
    loss_fn,
    initial_images: torch.Tensor,
    camera_poses_batch: torch.Tensor,
    gt_mesh_data: Dict[str, torch.Tensor],
    mesh_batch,
    mesh_paths: Optional[Sequence[Optional[str]]] = None,
) -> Tuple[float, torch.Tensor, float]:
    device = initial_images.device
    random_positions = sample_random_positions(
        batch_size=initial_images.shape[0],
        device=device,
        loss_fn=loss_fn,
    )
    random_pose = position_to_pose_tensor(random_positions)
    position_norm_mean = random_positions.norm(dim=1).mean().item()

    with torch.no_grad():
        result = evaluate_candidate_pose(
            renderer=renderer,
            loss_fn=loss_fn,
            pose=random_pose,
            initial_images=initial_images,
            camera_poses_batch=camera_poses_batch,
            gt_mesh_data=gt_mesh_data,
            mesh_batch=mesh_batch,
            point_cloud_dir=None,
        )

    return (
        float(result.loss_components.get("chamfer_loss", 0.0)),
        result.new_images,
        position_norm_mean,
    )
