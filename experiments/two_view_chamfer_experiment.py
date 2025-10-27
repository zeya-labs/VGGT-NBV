#!/usr/bin/env python3
"""
Two-view Chamfer distance experiment.

Given a mesh, this script samples camera poses on a sphere, renders candidate views,
and evaluates MapAnything reconstructions formed by pairing a fixed first view with
each remaining candidate as the second view. The Chamfer distance for every pair is
computed and visualised with seaborn (or a compatible fallback).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:  # prefer user-requested resun if available
    import resun as sns  # type: ignore
except ImportError:  # pragma: no cover - fallback to seaborn
    try:
        import seaborn as sns  # type: ignore
    except ImportError:  # pragma: no cover - seaborn might be absent in minimal envs
        sns = None  # type: ignore

import matplotlib.pyplot as plt

try:
    from torchvision.utils import save_image
except ImportError as exc:  # pragma: no cover - torchvision required for image dumps
    raise ImportError("torchvision is required to save rendered images.") from exc

from nbv_framework.models.mapanything_wrapper import MapAnythingWrapper
from nbv_framework.rendering.differentiable_renderer import DifferentiableRenderer
from nbv_framework.training.loss import ReconstructionLoss
from mapanything.utils.inference import postprocess_model_outputs_for_inference
from nbv_framework.utils.camera_utils import (
    CameraPoseGenerator,
    world_points_to_camera_depth,
)
from nbv_framework.utils.mesh_utils import load_and_normalize_mesh
from nbv_framework.utils.mapanything_views import (
    pose7d_to_opencv_cam2world_with_official_func,
    prepare_mapanything_views,
)
from mapanything.utils.hf_utils.viz import predictions_to_glb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate two-view reconstructions across candidate viewpoints."
    )
    parser.add_argument("--mesh_path", type=Path, required=True, help="Path to the mesh to evaluate.")
    parser.add_argument(
        "--num_views",
        type=int,
        default=16,
        help="Number of candidate views uniformly sampled on the sphere.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/two_view_experiment"),
        help="Directory to store rendered views, metadata, and plots.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=518,
        help="Square render分辨率；会自动调整为14的倍数以匹配MapAnything patch大小。",
    )
    parser.add_argument(
        "--fov_degrees",
        type=float,
        default=60.0,
        help="Horizontal field of view for both rendering and MapAnything.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold applied to MapAnything world points.",
    )
    parser.add_argument(
        "--normalize_method",
        type=str,
        default="quantile",
        help="Mesh normalization method passed to load_and_normalize_mesh.",
    )
    parser.add_argument(
        "--num_gt_samples",
        type=int,
        default=10000,
        help="Number of points sampled on the ground-truth mesh surface.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling camera sampling and tie-breaking.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device (cuda, cpu, or auto).",
    )
    parser.add_argument(
        "--max_points_per_cloud",
        type=int,
        default=4096,
        help="Maximum number of points retained per cloud when logging.",
    )
    parser.add_argument(
        "--hemisphere",
        type=str,
        default="upper",
        choices=("upper", "full"),
        help="Hemisphere selection for camera pose generation.",
    )
    parser.add_argument(
        "--skip_same_view",
        action="store_true",
        help="Skip evaluating the pair where the second view equals the fixed first view.",
    )
    parser.add_argument(
        "--skip_glb",
        action="store_true",
        help="Do not export GLB files for each view pair.",
    )
    parser.add_argument(
        "--glb_as_mesh",
        action="store_true",
        help="Export GLB as triangular mesh instead of point cloud.",
    )
    parser.add_argument(
        "--render_batch_size",
        type=int,
        default=50,
        help="Number of candidate views to render per batch to limit VRAM usage.",
    )
    parser.add_argument(
        "--skip_visualization",
        action="store_true",
        help="Do not export the interactive camera visualisation HTML.",
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pose_dict_to_tensor(pose: Dict[str, Sequence[float]]) -> torch.Tensor:
    position = list(pose["position"])
    quaternion = list(pose["quaternion"])
    if len(position) != 3 or len(quaternion) != 4:
        raise ValueError(f"Invalid pose dictionary: {pose}")
    return torch.tensor(position + quaternion, dtype=torch.float32)


def render_candidate_views(
    mesh,
    poses: torch.Tensor,
    renderer: DifferentiableRenderer,
    fov_degrees: float,
    batch_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Render colour images and point maps for every sampled pose."""
    total_views = poses.shape[0]
    if total_views == 0:
        raise ValueError("No poses provided for rendering.")

    if batch_size is None or batch_size <= 0 or batch_size >= total_views:
        batch_size = total_views

    image_chunks: List[torch.Tensor] = []
    point_map_chunks: List[torch.Tensor] = []
    valid_mask_chunks: List[torch.Tensor] = []
    have_valid_masks = False

    for start in range(0, total_views, batch_size):
        end = min(start + batch_size, total_views)
        pose_chunk = poses[start:end]
        mesh_chunk = mesh.extend(len(pose_chunk)).to(renderer.device)
        render_out = renderer.forward(
            mesh_chunk,
            pose_chunk,
            pose_format="cartesian",
            fov=float(fov_degrees),
            return_point_maps=True,
        )
        if not isinstance(render_out, tuple):
            raise RuntimeError("Renderer did not return point maps; enable return_point_maps=True.")
        images_chunk, point_maps_chunk, valid_masks_chunk = render_out
        image_chunks.append(images_chunk.detach())
        point_map_chunks.append(point_maps_chunk.detach())
        if valid_masks_chunk is not None:
            have_valid_masks = True
            valid_mask_chunks.append(valid_masks_chunk.detach())

    images = torch.cat(image_chunks, dim=0)
    point_maps = torch.cat(point_map_chunks, dim=0)
    valid_masks: Optional[torch.Tensor]
    if have_valid_masks:
        if len(valid_mask_chunks) != len(image_chunks):
            # Some batches may not return masks; pad with None behaviour.
            raise RuntimeError("Renderer returned valid masks inconsistently across batches.")
        valid_masks = torch.cat(valid_mask_chunks, dim=0)
    else:
        valid_masks = None

    return images, point_maps, valid_masks


def compute_depth_maps(
    point_maps: torch.Tensor,
    poses: torch.Tensor,
    valid_masks: Optional[torch.Tensor],
) -> torch.Tensor:
    """Convert rendered world coordinates to per-pixel depth."""
    masks_hw = None
    if valid_masks is not None:
        masks_hw = valid_masks.squeeze(1)
    point_maps_hw = point_maps.permute(0, 2, 3, 1)
    depth_maps = world_points_to_camera_depth(
        point_maps_hw,
        poses,
        valid_masks=masks_hw,
    )
    return depth_maps.float()


def save_candidate_artifacts(
    output_dir: Path,
    images: torch.Tensor,
    depth_maps: torch.Tensor,
    pose_dicts: Sequence[Dict[str, Sequence[float]]],
) -> None:
    """Persist rendered RGB, depth, and pose metadata for inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    depth_dir = output_dir / "depth"
    pose_dir = output_dir / "poses"
    image_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    pose_dir.mkdir(exist_ok=True)

    for idx in range(images.shape[0]):
        save_path = image_dir / f"view_{idx:03d}.png"
        save_image(images[idx].cpu(), str(save_path))

        depth_path = depth_dir / f"view_{idx:03d}.pt"
        torch.save(depth_maps[idx].cpu(), depth_path)

        pose_path = pose_dir / f"view_{idx:03d}.json"
        with open(pose_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "position": list(pose_dicts[idx]["position"]),
                    "quaternion": list(pose_dicts[idx]["quaternion"]),
                },
                handle,
                indent=2,
            )


def save_depth_visualizations(
    output_dir: Path,
    depth_batch: torch.Tensor,
    pair_name: str,
) -> None:
    """Save depth maps as grayscale images for inspection."""
    depth_dir = output_dir / "depth_pairs"
    depth_dir.mkdir(parents=True, exist_ok=True)

    depth_cpu = depth_batch.detach().cpu()
    batch, views = depth_cpu.shape[:2]
    for view_idx in range(views):
        depth_map = depth_cpu[0, view_idx, ..., 0]
        finite_mask = torch.isfinite(depth_map)
        if finite_mask.any():
            valid_values = depth_map[finite_mask]
            min_val = valid_values.min()
            max_val = valid_values.max()
        else:
            min_val = depth_map.min()
        max_val = depth_map.max()
    denom = (max_val - min_val).clamp_min(1e-6)
    normalized = (depth_map - min_val) / denom
    save_path = depth_dir / f"{pair_name}_view_{view_idx}.png"
    save_image(normalized.unsqueeze(0), str(save_path))


def collate_predictions_for_glb(
    predictions: List[Dict[str, torch.Tensor]],
    *,
    confidence_threshold: Optional[float] = None,
    filter_black_bg: bool = True,
    black_bg_threshold: float = 0.2,
) -> Dict[str, np.ndarray]:
    """Convert post-processed predictions into numpy arrays expected by predictions_to_glb."""
    world_points: List[np.ndarray] = []
    images: List[np.ndarray] = []
    final_masks: List[np.ndarray] = []
    extrinsics: List[np.ndarray] = []
    confidence_frames: List[np.ndarray] = []
    has_conf = any("conf" in pred for pred in predictions)

    for pred_idx, pred in enumerate(predictions):
        required_keys = ("pts3d", "img_no_norm", "mask", "camera_poses")
        for key in required_keys:
            if key not in pred:
                raise KeyError(f"Prediction {pred_idx} missing key '{key}' required for GLB export.")

        pts3d_np = pred["pts3d"].detach().cpu().numpy()
        images_np = pred["img_no_norm"].detach().cpu().numpy()
        mask_np = pred["mask"].detach().cpu().numpy()
        poses_np = pred["camera_poses"].detach().cpu().numpy()
        conf_np = pred.get("conf")
        if conf_np is not None:
            conf_np = conf_np.detach().cpu().numpy()

        batch_size = pts3d_np.shape[0]
        for batch_idx in range(batch_size):
            pts_frame = np.asarray(pts3d_np[batch_idx], dtype=np.float32)
            image_frame = images_np[batch_idx]
            mask_frame = mask_np[batch_idx]
            pose_frame = poses_np[batch_idx]

            if image_frame.ndim == 3 and image_frame.shape[0] in {3, 4} and image_frame.shape[-1] != image_frame.shape[0]:
                image_frame = np.transpose(image_frame, (1, 2, 0))
            image_frame = np.asarray(image_frame, dtype=np.float32)

            if mask_frame.ndim == 3 and mask_frame.shape[-1] == 1:
                mask_frame = mask_frame[..., 0]
            frame_mask = np.asarray(mask_frame, dtype=bool)

            if conf_np is not None and confidence_threshold is not None:
                conf_frame = conf_np[batch_idx]
                if conf_frame.ndim == 3 and conf_frame.shape[-1] == 1:
                    conf_frame = conf_frame[..., 0]
                frame_mask &= conf_frame >= confidence_threshold

            if filter_black_bg and image_frame.ndim == 3 and image_frame.shape[-1] >= 3:
                color_for_filter = image_frame[..., :3]
                max_val = float(color_for_filter.max()) if color_for_filter.size else 0.0
                if max_val > 1.5:
                    color_for_filter = color_for_filter / 255.0
                frame_mask &= np.max(color_for_filter, axis=-1) > black_bg_threshold

            world_points.append(pts_frame)
            images.append(image_frame)
            final_masks.append(frame_mask)
            extrinsics.append(np.asarray(pose_frame, dtype=np.float32))

            if conf_np is not None:
                conf_frame = conf_np[batch_idx]
                confidence_frames.append(np.asarray(conf_frame, dtype=np.float32))

    if not world_points:
        raise ValueError("No predictions available for GLB export.")

    result = {
        "world_points": np.stack(world_points, axis=0).astype(np.float32),
        "images": np.stack(images, axis=0).astype(np.float32),
        "final_mask": np.stack(final_masks, axis=0).astype(bool),
        "extrinsic": np.stack(extrinsics, axis=0).astype(np.float32),
    }

    if has_conf and confidence_frames:
        result["conf"] = np.stack(confidence_frames, axis=0).astype(np.float32)

    return result


def export_glb_for_pair(
    output_path: Path,
    predictions: List[Dict[str, torch.Tensor]],
    *,
    as_mesh: bool,
    confidence_threshold: Optional[float] = None,
    filter_black_bg: bool = True,
    black_bg_threshold: float = 0.2,
) -> None:
    """Export MapAnything predictions to a GLB file."""
    predictions_np = collate_predictions_for_glb(
        predictions,
        confidence_threshold=confidence_threshold,
        filter_black_bg=filter_black_bg,
        black_bg_threshold=black_bg_threshold,
    )
    scene = predictions_to_glb(
        predictions_np,
        show_cam=True,
        as_mesh=as_mesh,
        mask_black_bg=True,
        mask_ambiguous=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_path))
    print(f"Saved GLB to {output_path}")


def plot_results(
    results: List[Dict[str, Optional[float]]],
    first_view: int,
    output_path: Path,
) -> None:
    """Visualise Chamfer distance progression across second-view choices."""
    indices: List[int] = []
    chamfers: List[float] = []
    for item in results:
        value = item["chamfer"]
        if value is None or not math.isfinite(value):
            continue
        indices.append(int(item["second_view_index"]))
        chamfers.append(float(value))

    if not chamfers:
        print("No valid Chamfer distances to plot; skipping visualisation.")
        return

    plt.figure(figsize=(max(6, len(chamfers) * 0.4), 4))
    if sns is not None:  # type: ignore
        sns.barplot(x=indices, y=chamfers, color="#1f77b4")  # type: ignore
    else:
        plt.bar(indices, chamfers, color="#1f77b4")

    plt.axvline(first_view, color="#d62728", linestyle="--", linewidth=1.5, label="fixed first view")
    plt.xlabel("Second view index")
    plt.ylabel("Chamfer distance (lower is better)")
    plt.title("Two-view reconstruction quality per candidate second view")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def export_camera_heatmap_visualisation(
    *,
    mesh,
    pose_tensor: torch.Tensor,
    chamfer_results: List[Dict[str, Optional[float]]],
    first_view: int,
    output_path: Path,
    image_hw: Tuple[int, int],
) -> None:
    """Export an interactive Plotly visualisation of cameras, mesh, and Chamfer values."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is not installed; skipping interactive visualisation export.")
        return

    if pose_tensor.ndimension() != 2 or pose_tensor.shape[-1] != 7:
        raise ValueError(f"Expected pose tensor of shape (N, 7); got {pose_tensor.shape}.")

    mesh_cpu = mesh.detach().to("cpu") if hasattr(mesh, "detach") else mesh.to("cpu")
    verts = mesh_cpu.verts_list()[0].cpu().numpy()
    faces = mesh_cpu.faces_list()[0].cpu().numpy()

    cam2world = pose7d_to_opencv_cam2world_with_official_func(
        pose_tensor.detach().to(device="cpu", dtype=torch.float32),
        image_size=image_hw,
    )
    if cam2world.dim() == 3:
        cam2world_mats = cam2world
    elif cam2world.dim() == 2:
        cam2world_mats = cam2world.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected cam2world shape: {cam2world.shape}")

    cam2world_np = cam2world_mats.cpu().numpy()
    positions = cam2world_np[:, :3, 3]
    rotations = cam2world_np[:, :3, :3]

    num_views = positions.shape[0]
    chamfer_values = np.full(num_views, np.nan, dtype=np.float32)
    for item in chamfer_results:
        value = item.get("chamfer")
        idx = item.get("second_view_index")
        if idx is None:
            continue
        if idx < 0 or idx >= num_views:
            continue
        if value is None or not math.isfinite(value):
            continue
        chamfer_values[idx] = float(value)

    valid_mask = np.isfinite(chamfer_values)
    valid_indices = np.where(valid_mask)[0]
    valid_values = chamfer_values[valid_mask]

    if valid_values.size > 0:
        cmin = float(np.nanmin(valid_values))
        cmax = float(np.nanmax(valid_values))
    else:
        cmin, cmax = 0.0, 1.0

    if not np.isfinite(cmax - cmin):
        cmax = cmin + 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="lightgray",
            opacity=0.35,
            name="GT mesh",
            hoverinfo="skip",
        )
    )

    vert_norms = np.linalg.norm(verts, axis=1)
    mesh_extent = float(vert_norms.max()) if vert_norms.size else 1.0
    first_pos = positions[first_view]
    first_forward = rotations[first_view] @ np.array([0.0, 0.0, 1.0])

    fig.add_trace(
        go.Scatter3d(
            x=[first_pos[0]],
            y=[first_pos[1]],
            z=[first_pos[2]],
            mode="markers+text",
            marker=dict(
                size=10,
                color="red",
                symbol="diamond",
            ),
            text=[f"first view {first_view}"],
            textposition="top center",
            name="Fixed first view",
        )
    )
    fig.add_trace(
        go.Cone(
            x=[first_pos[0]],
            y=[first_pos[1]],
            z=[first_pos[2]],
            u=[first_forward[0]],
            v=[first_forward[1]],
            w=[first_forward[2]],
            anchor="tip",
            colorscale=[[0.0, "red"], [1.0, "red"]],
            showscale=False,
            name="First view direction",
            sizemode="absolute",
            sizeref=max(mesh_extent * 0.2, 1e-3),
        )
    )

    if valid_indices.size > 0:
        scatter_positions = positions[valid_indices]
        scatter_values = chamfer_values[valid_indices]
        view_labels = [
            f"view {idx} | chamfer={val:.4f}" for idx, val in zip(valid_indices, scatter_values)
        ]
        if cmax - cmin < 1e-6:
            marker_sizes = np.full_like(scatter_values, 12.0, dtype=np.float32)
        else:
            normalized = (scatter_values - cmin) / (cmax - cmin)
            marker_sizes = 10.0 + normalized * 20.0

        fig.add_trace(
            go.Scatter3d(
                x=scatter_positions[:, 0],
                y=scatter_positions[:, 1],
                z=scatter_positions[:, 2],
                mode="markers+text",
                marker=dict(
                    size=marker_sizes,
                    sizemode="diameter",
                    color=scatter_values,
                    colorscale="Turbo",
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(
                        title="Chamfer distance",
                    ),
                    opacity=0.9,
                ),
                text=view_labels,
                textposition="bottom center",
                name="Second-view candidates",
                hovertemplate="view %{text}<extra></extra>",
            )
        )

    missing_mask = ~valid_mask
    missing_mask[first_view] = False
    missing_indices = np.where(missing_mask)[0]
    if missing_indices.size > 0:
        missing_positions = positions[missing_indices]
        fig.add_trace(
            go.Scatter3d(
                x=missing_positions[:, 0],
                y=missing_positions[:, 1],
                z=missing_positions[:, 2],
                mode="markers+text",
                marker=dict(size=6, color="gray", symbol="x"),
                text=[f"view {idx} (no chamfer)" for idx in missing_indices],
                textposition="top center",
                name="Missing chamfer",
            )
        )

    axis_cfg = dict(
        showbackground=False,
        showgrid=False,
        zeroline=False,
    )
    fig.update_layout(
        title="Two-view Chamfer camera heatmap",
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="center", x=0.5),
        scene=dict(
            xaxis=dict(title="X", **axis_cfg),
            yaxis=dict(title="Y", **axis_cfg),
            zaxis=dict(title="Z", **axis_cfg),
            aspectmode="data",
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Saved interactive camera heatmap to {output_path}")




def main() -> None:
    args = parse_args()
    if args.device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = args.device
    device = torch.device(resolved_device)
    device_str = device.type if device.index is None else f"{device.type}:{device.index}"
    set_random_seed(args.seed)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_data = load_and_normalize_mesh(
        mesh_path=str(args.mesh_path),
        normalize_method=args.normalize_method,
        num_samples=args.num_gt_samples,
    )
    mesh = mesh_data["normalized_mesh"].to(device)
    gt_points_tensor = mesh_data["gt_points"].to(device=device, dtype=torch.float32)

    pose_generator = CameraPoseGenerator()
    pose_dicts = pose_generator.generate_camera_poses(
        args.num_views,
        seed=args.seed,
        hemisphere=args.hemisphere,
    )
    pose_tensor = torch.stack([pose_dict_to_tensor(p) for p in pose_dicts]).to(device)

    renderer = DifferentiableRenderer(image_size=args.image_size, device=device_str)
    images, point_maps, valid_masks = render_candidate_views(
        mesh,
        pose_tensor,
        renderer,
        args.fov_degrees,
        batch_size=args.render_batch_size,
    )
    images = images.to(device)
    point_maps = point_maps.to(device)
    valid_masks = valid_masks.to(device) if valid_masks is not None else None
    depth_maps = compute_depth_maps(point_maps, pose_tensor, valid_masks)
    depth_maps = depth_maps.to(device)

    save_candidate_artifacts(output_dir / "views", images, depth_maps, pose_dicts)

    mapanything = MapAnythingWrapper(device=device_str)
    mapanything.eval()
    loss_fn = ReconstructionLoss(renderer=renderer, log_tensorboard=False)
    loss_fn.eval()

    first_view = random.randrange(args.num_views)
    results: List[Dict[str, Optional[float]]] = []

    # Prepare fixed initial view tensors following training_step convention
    initial_image = images[first_view].unsqueeze(0).unsqueeze(0)  # [1,1,3,H,W]
    initial_pose = pose_tensor[first_view].unsqueeze(0).unsqueeze(0)  # [1,1,7]
    initial_depth = depth_maps[first_view].unsqueeze(0).unsqueeze(0)  # [1,1,H,W,1]
    mesh_batch = mesh
    gt_points_batch = gt_points_tensor.unsqueeze(0)

    point_maps_hw = point_maps.permute(0, 2, 3, 1).contiguous()
    if valid_masks is not None:
        valid_masks_hw = valid_masks.squeeze(1).bool()
    else:
        valid_masks_hw = None

    with torch.no_grad():
        for idx in range(args.num_views):
            if idx == first_view and args.skip_same_view:
                continue

            new_image = images[idx].unsqueeze(0).unsqueeze(0)  # [1,1,3,H,W]
            new_pose = pose_tensor[idx].unsqueeze(0).unsqueeze(0)  # [1,1,7]
            new_depth = depth_maps[idx].unsqueeze(0).unsqueeze(0)  # [1,1,H,W,1]

            combined_images_batch = torch.cat([initial_image, new_image], dim=1).contiguous()
            combined_camera_poses = torch.cat([initial_pose, new_pose], dim=1).contiguous()
            combined_depth = torch.cat([initial_depth, new_depth], dim=1).contiguous()

            processed_views, normalized_views = prepare_mapanything_views(
                combined_images_batch,
                combined_camera_poses,
                data_norm_type=mapanything.data_norm_type,
                device=mapanything.device,
                fov_degrees=args.fov_degrees,
                is_metric_scale=False,
                depth_z=combined_depth,
            )

            mapanything._configure_geometric_inputs(
                use_calibration=True,
                use_pose=True,
                use_depth=combined_depth is not None,
            )
            try:
                raw_predictions = mapanything.base_model.forward(
                    processed_views,
                    memory_efficient_inference=mapanything.memory_efficient_inference,
                )
            finally:
                mapanything._restore_geometric_inputs()

            recon = mapanything._stack_predictions(raw_predictions)
            mapanything._maybe_retain_grad_from_result(recon, normalized_views)

            postprocessed_predictions = postprocess_model_outputs_for_inference(
                raw_outputs=raw_predictions,
                input_views=processed_views,
                apply_mask=True,
                mask_edges=True,
                edge_normal_threshold=5.0,
                edge_depth_threshold=0.03,
                apply_confidence_mask=False,
                confidence_percentile=10,
            )

            pair_name = f"pair_{first_view:03d}_{idx:03d}"
            save_depth_visualizations(output_dir, combined_depth, pair_name)

            initial_point_map = point_maps_hw[first_view].unsqueeze(0).unsqueeze(0).to(device)
            new_point_map = point_maps_hw[idx].unsqueeze(0).unsqueeze(0).to(device)
            gt_point_maps = torch.cat([initial_point_map, new_point_map], dim=1).contiguous()

            if valid_masks_hw is not None:
                initial_mask = valid_masks_hw[first_view].unsqueeze(0).unsqueeze(0).to(device)
                new_mask = valid_masks_hw[idx].unsqueeze(0).unsqueeze(0).to(device)
                gt_valid_masks = torch.cat([initial_mask, new_mask], dim=1).contiguous()
            else:
                gt_valid_masks = torch.ones_like(gt_point_maps[..., 0], dtype=torch.bool).contiguous()

            gt_mesh_data_pair: Dict[str, torch.Tensor] = {
                "normalized_mesh": mesh_batch,
                "gt_points": gt_points_batch,
                "gt_point_maps": gt_point_maps,
                "gt_valid_masks": gt_valid_masks,
                "depth_z": combined_depth,
            }

            total_loss, loss_components = loss_fn(
                recon,
                gt_mesh_data_pair,
                combined_images_batch,
                combined_camera_poses,
                return_components=True,
                writer=None,
                step=None,
                train_flag=False,
                point_cloud_dir=str((output_dir / "loss_pointclouds")),
            )

            chamfer_value = loss_components.get("chamfer_loss")
            chamfer_value = float(chamfer_value) if chamfer_value is not None else None

            results.append(
                {
                    "second_view_index": idx,
                    "chamfer": chamfer_value,
                    "total_loss": float(total_loss.item()),
                }
            )

            if not args.skip_glb:
                glb_name = f"{pair_name}.glb"
                glb_path = (output_dir / "glb") / glb_name
                export_glb_for_pair(
                    glb_path,
                    postprocessed_predictions,
                    as_mesh=args.glb_as_mesh,
                    confidence_threshold=args.confidence_threshold,
                    filter_black_bg=True,
                    black_bg_threshold=0.2,
                )

    summary_path = output_dir / "chamfer_results.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "mesh_path": str(args.mesh_path),
                "num_views": args.num_views,
                "fixed_first_view": first_view,
                "confidence_threshold": args.confidence_threshold,
                "results": results,
            },
            handle,
            indent=2,
        )

    plot_path = output_dir / "chamfer_per_second_view.png"
    plot_results(results, first_view, plot_path)
    if not args.skip_visualization:
        interactive_path = output_dir / "camera_heatmap.html"
        export_camera_heatmap_visualisation(
            mesh=mesh,
            pose_tensor=pose_tensor,
            chamfer_results=results,
            first_view=first_view,
            output_path=interactive_path,
            image_hw=tuple(int(dim) for dim in images.shape[-2:]),
        )
    print(f"Experiment complete. Results saved to {summary_path} and {plot_path}.")


if __name__ == "__main__":
    main()
