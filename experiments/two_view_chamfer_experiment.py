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
import contextlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import queue
import random
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import multiprocessing as mp
import torch.multiprocessing as tmp

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
    from tqdm import tqdm
except ImportError:  # pragma: no cover - progress bar optional
    tqdm = None  # type: ignore

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
from pytorch3d.renderer.mesh import TexturesVertex, TexturesUV
from pytorch3d.structures import Meshes

POSTPROCESS_LOCK = threading.Lock()


@dataclass
class ArtifactTask:
    kind: str
    payload: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate two-view reconstructions across candidate viewpoints."
    )
    parser.add_argument("--mesh_path", type=Path, required=False, help="Path to the mesh to evaluate.")
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
        "--device_ids",
        type=str,
        default=None,
        help="Comma-separated CUDA device ids to use (e.g. '0,1,2,3') or 'auto' to select automatically.",
    )
    parser.add_argument(
        "--max_devices",
        type=int,
        default=4,
        help="Maximum number of devices to use when auto-selecting GPUs.",
    )
    parser.add_argument(
        "--max_points_per_cloud",
        type=int,
        default=4096,
        help="Maximum number of points retained per cloud when logging.",
    )
    parser.add_argument(
        "--camera_base_radius",
        type=float,
        default=2.2,
        help="Base camera radius used for spherical sampling (see camera generator).",
    )
    parser.add_argument(
        "--camera_radius_variation",
        type=float,
        default=0.0,
        help="Uniform variation around base radius for sampling cameras.",
    )
    parser.add_argument(
        "--camera_radius_mode",
        type=str,
        default="random",
        choices=("constant", "random", "layered"),
        help="How to sample camera radius: constant, random (default), or layered shells.",
    )
    parser.add_argument(
        "--camera_radius_layers",
        type=int,
        default=1,
        help="Number of radial layers when radius mode is 'layered'.",
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
        "--skip_depth_png",
        action="store_true",
        help="Skip saving per-pair depth PNG previews.",
    )
    parser.add_argument(
        "--skip_view_artifacts",
        action="store_true",
        help="Skip saving rendered candidate view RGB/depth/pose files under outputs/views.",
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
    parser.add_argument(
        "--max_glb",
        type=int,
        default=None,
        help="Maximum number of GLB files to export (default: export all).",
    )
    parser.add_argument(
        "--regenerate_html_only",
        action="store_true",
        help="Rebuild visualisations from existing outputs and exit.",
    )
    return parser.parse_args()


def _serialize_mesh(mesh: Meshes) -> Dict[str, Any]:
    """Serialize a PyTorch3D mesh for multiprocessing transport."""
    verts_list = [verts.cpu() for verts in mesh.verts_list()]
    faces_list = [faces.cpu() for faces in mesh.faces_list()]

    textures = mesh.textures
    if textures is None:
        texture_type = None
        texture_data = None
    elif isinstance(textures, TexturesVertex):
        texture_type = "vertex"
        texture_data = [feat.cpu() for feat in textures.verts_features_list()]
    elif isinstance(textures, TexturesUV):
        texture_type = "uv"
        texture_data = {
            "maps": [m.cpu() for m in textures.maps_list()],
            "faces_uvs": [f.cpu() for f in textures.faces_uvs_list()],
            "verts_uvs": [v.cpu() for v in textures.verts_uvs_list()],
        }
    else:
        raise NotImplementedError(
            f"Unsupported mesh texture type for multiprocessing rendering: {type(textures).__name__}"
        )

    return {
        "verts": verts_list,
        "faces": faces_list,
        "texture_type": texture_type,
        "texture_data": texture_data,
    }


def _deserialize_mesh(serialized: Dict[str, Any], device: torch.device) -> Meshes:
    """Reconstruct a PyTorch3D mesh from serialized components."""
    verts = [v.to(device) for v in serialized["verts"]]
    faces = [f.to(device) for f in serialized["faces"]]

    texture_type = serialized["texture_type"]
    if texture_type is None:
        textures = None
    elif texture_type == "vertex":
        texture_data = [t.to(device) for t in serialized["texture_data"]]
        textures = TexturesVertex(verts_features=texture_data)
    elif texture_type == "uv":
        data = serialized["texture_data"]
        textures = TexturesUV(
            maps=[m.to(device) for m in data["maps"]],
            faces_uvs=[f.to(device) for f in data["faces_uvs"]],
            verts_uvs=[v.to(device) for v in data["verts_uvs"]],
        )
    else:
        raise NotImplementedError(f"Unsupported texture type '{texture_type}' during deserialization.")

    return Meshes(verts=verts, faces=faces, textures=textures)


def resolve_device_strings(
    device_arg: str,
    device_ids_arg: Optional[str],
    max_devices: int,
) -> List[str]:
    """Resolve CLI device arguments into an ordered list of torch device strings."""

    def auto_devices() -> List[str]:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if max_devices > 0:
                count = min(count, max_devices)
            devices = [f"cuda:{idx}" for idx in range(count)]
            if devices:
                return devices
        return ["cpu"]

    if device_ids_arg is not None:
        token = device_ids_arg.strip()
        if not token:
            raise ValueError("device_ids cannot be an empty string.")
        if token.lower() == "auto":
            return auto_devices()
        raw_devices = [item.strip() for item in token.split(",") if item.strip()]
        if not raw_devices:
            raise ValueError(f"Could not parse any devices from device_ids='{device_ids_arg}'.")
        resolved: List[str] = []
        for dev in raw_devices:
            lower_dev = dev.lower()
            candidate: Optional[str]
            if lower_dev in {"cpu"}:
                candidate = "cpu"
            elif lower_dev.startswith("cuda"):
                candidate = lower_dev
            elif lower_dev.isdigit():
                candidate = f"cuda:{lower_dev}"
            else:
                candidate = dev
            if candidate not in resolved:
                resolved.append(candidate)
        return resolved

    if device_arg == "auto":
        return auto_devices()

    return [device_arg]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class DeviceContext:
    device: torch.device
    device_str: str
    mapanything: MapAnythingWrapper
    loss_fn: ReconstructionLoss
    mesh: Any
    gt_points: torch.Tensor
    initial_image: torch.Tensor
    initial_pose: torch.Tensor
    initial_depth: torch.Tensor
    initial_point_map: torch.Tensor
    initial_mask: Optional[torch.Tensor]

def pose_dict_to_tensor(pose: Dict[str, Sequence[float]]) -> torch.Tensor:
    position = list(pose["position"])
    quaternion = list(pose["quaternion"])
    if len(position) != 3 or len(quaternion) != 4:
        raise ValueError(f"Invalid pose dictionary: {pose}")
    return torch.tensor(position + quaternion, dtype=torch.float32)


def _to_cpu_recursive(data: Any) -> Any:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    if isinstance(data, dict):
        return {key: _to_cpu_recursive(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_to_cpu_recursive(item) for item in data]
    if isinstance(data, tuple):
        return tuple(_to_cpu_recursive(item) for item in data)
    return data


def _render_candidate_views_single_device(
    mesh,
    poses: torch.Tensor,
    renderer: DifferentiableRenderer,
    fov_degrees: float,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Any] = None,
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
        if progress_callback is not None:
            progress_callback(images_chunk.shape[0])

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

    images_cpu = images.cpu()
    point_maps_cpu = point_maps.cpu()
    valid_masks_cpu = valid_masks.cpu() if valid_masks is not None else None

    return images_cpu, point_maps_cpu, valid_masks_cpu


def _render_device_worker(
    device_str: str,
    image_size: int,
    fov_degrees: float,
    batch_size: Optional[int],
    mesh_serialized: Dict[str, Any],
    pose_subset: torch.Tensor,
    start_index: int,
    result_queue,
) -> None:
    """Worker process that renders a contiguous subset of poses on a dedicated device."""
    try:
        device = torch.device(device_str)
        if device.type == "cuda":
            torch.cuda.set_device(device)

        mesh = _deserialize_mesh(mesh_serialized, device)
        renderer = DifferentiableRenderer(image_size=image_size, device=device_str)

        poses_device = pose_subset.to(device)
        images_chunk, point_maps_chunk, valid_masks_chunk = _render_candidate_views_single_device(
            mesh,
            poses_device,
            renderer,
            fov_degrees,
            batch_size,
        )

        result_queue.put(
            (
                "__result__",
                start_index,
                images_chunk,
                point_maps_chunk,
                valid_masks_chunk,
            )
        )
    except Exception as exc:  # pragma: no cover - worker errors bubble up
        result_queue.put(("__error__", device_str, exc))
    finally:
        result_queue.put(("__done__", device_str))
        if "poses_device" in locals():
            del poses_device
        if locals().get("device", torch.device("cpu")).type == "cuda":
            torch.cuda.empty_cache()


def _artifact_worker(task_queue: "queue.Queue[Optional[ArtifactTask]]") -> None:
    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break
        try:
            if task.kind == "depth_png":
                save_depth_visualizations(
                    task.payload["output_dir"],
                    task.payload["depth_batch"],
                    task.payload["pair_name"],
                )
            elif task.kind == "glb":
                export_glb_for_pair(
                    task.payload["output_path"],
                    task.payload["predictions"],
                    as_mesh=task.payload["as_mesh"],
                    confidence_threshold=task.payload["confidence_threshold"],
                    filter_black_bg=task.payload.get("filter_black_bg", True),
                    black_bg_threshold=task.payload.get("black_bg_threshold", 0.2),
                )
        except Exception as exc:  # pragma: no cover - background errors surface via stdout
            print(f"[artifact] Error while processing {task.kind}: {exc}")
        finally:
            task_queue.task_done()


def render_candidate_views(
    mesh_cpu,
    poses_cpu: torch.Tensor,
    *,
    image_size: int,
    fov_degrees: float,
    batch_size: Optional[int],
    device_strings: Sequence[str],
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Render candidate views, distributing work across available devices with progress feedback."""
    total_views = poses_cpu.shape[0]
    if total_views == 0:
        raise ValueError("No poses provided for rendering.")

    if batch_size is None or batch_size <= 0:
        batch_size = total_views

    if not device_strings:
        raise ValueError("render_candidate_views requires at least one device string.")

    render_devices: List[str] = []
    for dev_str in device_strings:
        device = torch.device(dev_str)
        if device.type == "cuda" and torch.cuda.is_available():
            render_devices.append(dev_str)
    if not render_devices:
        render_devices = [device_strings[0]]
    render_devices = list(dict.fromkeys(render_devices))

    progress_bar = None
    if show_progress and tqdm is not None:
        progress_bar = tqdm(total=total_views, desc="Rendering candidate views", unit="view", leave=False)

    if len(render_devices) == 1:
        device_str = render_devices[0]
        device = torch.device(device_str)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        mesh_device = mesh_cpu.to(device)
        poses_device = poses_cpu.to(device)
        renderer = DifferentiableRenderer(image_size=image_size, device=device_str)
        try:
            images, point_maps, valid_masks = _render_candidate_views_single_device(
                mesh_device,
                poses_device,
                renderer,
                fov_degrees,
                batch_size,
                progress_callback=progress_bar.update if progress_bar is not None else None,
            )
        finally:
            del mesh_device, poses_device, renderer
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if progress_bar is not None:
            progress_bar.close()
        return images, point_maps, valid_masks

    # Multi-device rendering via multiprocessing workers (one worker per device).
    mesh_serialized = _serialize_mesh(mesh_cpu)
    ctx = tmp.get_context("spawn")
    result_queue = ctx.SimpleQueue()

    total_devices = len(render_devices)
    base_count = total_views // total_devices
    remainder = total_views % total_devices

    assignments: List[Tuple[str, int, int]] = []
    start_idx = 0
    for idx, device_str in enumerate(render_devices):
        count = base_count + (1 if idx < remainder else 0)
        if count <= 0:
            continue
        end_idx = start_idx + count
        assignments.append((device_str, start_idx, end_idx))
        start_idx = end_idx

    workers = []
    for device_str, start_idx, end_idx in assignments:
        pose_subset = poses_cpu[start_idx:end_idx].clone()
        worker = ctx.Process(
            target=_render_device_worker,
            args=(
                device_str,
                image_size,
                fov_degrees,
                batch_size,
                mesh_serialized,
                pose_subset,
                start_idx,
                result_queue,
            ),
        )
        worker.daemon = False
        worker.start()
        workers.append(worker)

    chunk_outputs: Dict[int, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = {}
    completed_workers = 0

    try:
        while completed_workers < len(workers):
            message = result_queue.get()
            if not message:
                continue
            kind = message[0]
            if kind == "__result__":
                _, start_idx, images_chunk, point_maps_chunk, valid_masks_chunk = message
                chunk_outputs[start_idx] = (images_chunk, point_maps_chunk, valid_masks_chunk)
                if progress_bar is not None:
                    progress_bar.update(images_chunk.shape[0])
            elif kind == "__error__":
                _, device_str, exc = message
                raise RuntimeError(f"Rendering worker on device {device_str} failed") from exc
            elif kind == "__done__":
                completed_workers += 1
    finally:
        if progress_bar is not None:
            progress_bar.close()
        for worker in workers:
            worker.join()

    if len(chunk_outputs) != len(assignments):
        raise RuntimeError("Rendering produced incomplete results.")

    ordered_starts = sorted(chunk_outputs.keys())
    images_list = [chunk_outputs[start][0] for start in ordered_starts]
    point_maps_list = [chunk_outputs[start][1] for start in ordered_starts]

    images = torch.cat(images_list, dim=0)
    point_maps = torch.cat(point_maps_list, dim=0)

    has_masks = any(chunk_outputs[start][2] is not None for start in ordered_starts)
    if has_masks:
        if any(chunk_outputs[start][2] is None for start in ordered_starts):
            raise RuntimeError("Renderer returned valid masks inconsistently across devices.")
        valid_masks = torch.cat([chunk_outputs[start][2] for start in ordered_starts], dim=0)
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
    for batch_idx in range(batch):
        for view_idx in range(views):
            depth_map = depth_cpu[batch_idx, view_idx, ..., 0]
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
            suffix = f"b{batch_idx}_view_{view_idx}" if batch > 1 else f"view_{view_idx}"
            save_path = depth_dir / f"{pair_name}_{suffix}.png"
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
                mode="markers",
                marker=dict(
                    size=5,
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
                hovertext=view_labels,
                name="Second-view candidates",
                hovertemplate="view %{hovertext}<extra></extra>",
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
                mode="markers",
                marker=dict(size=8, color="gray", symbol="x"),
                hovertext=[f"view {idx} (no chamfer)" for idx in missing_indices],
                hovertemplate="%{hovertext}<extra></extra>",
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


def _infer_image_hw_from_summary(output_dir: Path, summary: Dict[str, Any]) -> Tuple[int, int]:
    image_hw = summary.get("image_hw")
    if image_hw and len(image_hw) == 2:
        return int(image_hw[0]), int(image_hw[1])

    depth_dir = output_dir / "views" / "depth"
    if depth_dir.exists():
        for depth_file in sorted(depth_dir.glob("view_*.pt")):
            try:
                depth_tensor = torch.load(depth_file, map_location="cpu")
            except Exception:
                continue
            if depth_tensor.ndim >= 2:
                return int(depth_tensor.shape[0]), int(depth_tensor.shape[1])

    image_dir = output_dir / "views" / "images"
    if image_dir.exists():
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            Image = None
        else:
            for image_file in sorted(image_dir.glob("view_*.png")):
                try:
                    with Image.open(image_file) as img:
                        return int(img.height), int(img.width)
                except Exception:
                    continue

    raise FileNotFoundError(
        "Could not infer image resolution. Ensure 'image_hw' is recorded or keep depth/images under outputs/views."
    )


def regenerate_visualisations(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    summary_path = output_dir / "chamfer_results.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Cannot regenerate visualisations: {summary_path} not found.")

    with open(summary_path, "r", encoding="utf-8") as handle:
        summary_data = json.load(handle)

    mesh_path = summary_data.get("mesh_path")
    if mesh_path is None:
        raise ValueError("Summary JSON is missing 'mesh_path'; cannot regenerate visualisations.")

    normalize_method = summary_data.get("normalize_method", args.normalize_method)
    num_samples = int(summary_data.get("num_gt_samples", args.num_gt_samples))
    mesh_data = load_and_normalize_mesh(
        mesh_path=str(mesh_path),
        normalize_method=normalize_method,
        num_samples=num_samples,
    )
    mesh = mesh_data["normalized_mesh"]

    num_views = int(summary_data.get("num_views", 0))
    if num_views <= 0:
        raise ValueError("Summary JSON reports zero candidate views; nothing to regenerate.")

    if not summary_data.get("view_artifacts_saved", True):
        raise ValueError(
            "View artifacts were skipped in the original run (--skip_view_artifacts). "
            "Regeneration requires stored poses/images; rerun without skipping or provide poses manually."
        )

    pose_dir = output_dir / "views" / "poses"
    pose_tensors: List[torch.Tensor] = []
    for view_idx in range(num_views):
        pose_path = pose_dir / f"view_{view_idx:03d}.json"
        if not pose_path.exists():
            raise FileNotFoundError(f"Missing pose file for view {view_idx:03d}: {pose_path}")
        with open(pose_path, "r", encoding="utf-8") as handle:
            pose_dict = json.load(handle)
        pose_tensors.append(pose_dict_to_tensor(pose_dict))
    pose_tensor = torch.stack(pose_tensors, dim=0)

    results = summary_data.get("results", [])
    first_view = int(summary_data.get("fixed_first_view", 0))
    image_hw = _infer_image_hw_from_summary(output_dir, summary_data)

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
            image_hw=image_hw,
        )

    print(f"Regenerated visualisations in {output_dir}.")

def main() -> None:
    args = parse_args()
    if args.regenerate_html_only:
        regenerate_visualisations(args)
        return

    if args.mesh_path is None:
        raise ValueError("--mesh_path is required unless --regenerate_html_only is specified.")

    device_strings = resolve_device_strings(args.device, args.device_ids, args.max_devices)
    if not device_strings:
        raise ValueError("No devices resolved from CLI arguments.")
    set_random_seed(args.seed)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_data = load_and_normalize_mesh(
        mesh_path=str(args.mesh_path),
        normalize_method=args.normalize_method,
        num_samples=args.num_gt_samples,
    )
    mesh_cpu = mesh_data["normalized_mesh"]
    gt_points_tensor = mesh_data["gt_points"].to(dtype=torch.float32)

    pose_generator = CameraPoseGenerator()
    pose_dicts = pose_generator.generate_camera_poses(
        args.num_views,
        seed=args.seed,
        hemisphere=args.hemisphere,
        base_radius=args.camera_base_radius,
        radius_variation=args.camera_radius_variation,
        radius_mode=args.camera_radius_mode,
        radius_layers=args.camera_radius_layers,
    )
    pose_tensor = torch.stack([pose_dict_to_tensor(p) for p in pose_dicts])

    images, point_maps, valid_masks = render_candidate_views(
        mesh_cpu,
        pose_tensor,
        image_size=args.image_size,
        fov_degrees=args.fov_degrees,
        batch_size=args.render_batch_size,
        device_strings=device_strings,
        show_progress=True,
    )
    depth_maps = compute_depth_maps(point_maps, pose_tensor, valid_masks)

    if not args.skip_view_artifacts:
        save_candidate_artifacts(output_dir / "views", images, depth_maps, pose_dicts)
    else:
        print("Skipping view artifact export as requested; outputs/views/* will not be created.")

    point_maps_hw = point_maps.permute(0, 2, 3, 1).contiguous()
    valid_masks_hw = valid_masks.squeeze(1).bool() if valid_masks is not None else None

    first_view = random.randrange(args.num_views)
    print(f"Fixed first view index: {first_view}")
    print(f"Using devices: {', '.join(device_strings)}")

    artifact_queue: Optional[queue.Queue[Optional[ArtifactTask]]] = None
    artifact_workers: List[threading.Thread] = []
    if not (args.skip_glb and args.skip_depth_png):
        artifact_queue = queue.Queue()
        num_artifact_workers = max(1, min(4, len(device_strings)))
        for _ in range(num_artifact_workers):
            worker = threading.Thread(target=_artifact_worker, args=(artifact_queue,), daemon=True)
            worker.start()
            artifact_workers.append(worker)

    glb_export_counter = {"count": 0}
    glb_counter_lock = threading.Lock()

    initial_image_cpu = images[first_view]
    initial_pose_cpu = pose_tensor[first_view]
    initial_depth_cpu = depth_maps[first_view]
    initial_point_map_cpu = point_maps_hw[first_view]
    initial_mask_cpu = valid_masks_hw[first_view] if valid_masks_hw is not None else None

    point_cloud_dir = output_dir / "loss_pointclouds"
    point_cloud_dir.mkdir(parents=True, exist_ok=True)

    device_contexts: List[DeviceContext] = []
    for device_str in device_strings:
        device_obj = torch.device(device_str)
        mapanything = MapAnythingWrapper(device=device_str)
        mapanything.eval()
        mapanything.to(device_obj)
        renderer_ctx = DifferentiableRenderer(image_size=args.image_size, device=device_str)
        loss_fn = ReconstructionLoss(
            renderer=renderer_ctx,
            save_point_clouds=False,
            default_device=device_obj,
            tensor_dtype=torch.float32,
        )
        loss_fn.eval()

        mesh_device = mesh_cpu.to(device_obj)
        gt_points_device = gt_points_tensor.unsqueeze(0).to(device_obj)
        initial_image_device = initial_image_cpu.unsqueeze(0).unsqueeze(0).to(device_obj)
        initial_pose_device = initial_pose_cpu.unsqueeze(0).unsqueeze(0).to(device_obj)
        initial_depth_device = initial_depth_cpu.unsqueeze(0).unsqueeze(0).to(device_obj)
        initial_point_map_device = initial_point_map_cpu.unsqueeze(0).unsqueeze(0).to(device_obj)
        if initial_mask_cpu is not None:
            initial_mask_device = initial_mask_cpu.unsqueeze(0).unsqueeze(0).to(device_obj)
        else:
            initial_mask_device = None

        device_contexts.append(
            DeviceContext(
                device=device_obj,
                device_str=device_str,
                mapanything=mapanything,
                loss_fn=loss_fn,
                mesh=mesh_device,
                gt_points=gt_points_device,
                initial_image=initial_image_device,
                initial_pose=initial_pose_device,
                initial_depth=initial_depth_device,
                initial_point_map=initial_point_map_device,
                initial_mask=initial_mask_device,
            )
        )

    if not device_contexts:
        raise RuntimeError("No valid devices resolved for inference.")

    indices_to_eval = list(range(args.num_views))
    if args.skip_same_view and first_view in indices_to_eval:
        indices_to_eval.remove(first_view)

    results: List[Dict[str, Optional[float]]] = []

    point_maps_hw_cpu = point_maps_hw
    valid_masks_hw_cpu = valid_masks_hw
    images_cpu = images
    depth_maps_cpu = depth_maps
    pose_tensor_cpu = pose_tensor

    def process_indices(index_subset: Sequence[int], ctx: DeviceContext) -> List[Dict[str, Optional[float]]]:
        local_results: List[Dict[str, Optional[float]]] = []
        if not index_subset:
            return local_results
        device = ctx.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        device_ctx = torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()
        with device_ctx, torch.no_grad():
            for idx in index_subset:
                new_image = images_cpu[idx].unsqueeze(0).unsqueeze(0).to(ctx.device)
                new_pose = pose_tensor_cpu[idx].unsqueeze(0).unsqueeze(0).to(ctx.device)
                new_depth = depth_maps_cpu[idx].unsqueeze(0).unsqueeze(0).to(ctx.device)

                combined_images_batch = torch.cat([ctx.initial_image, new_image], dim=1).contiguous()
                combined_camera_poses = torch.cat([ctx.initial_pose, new_pose], dim=1).contiguous()
                combined_depth = torch.cat([ctx.initial_depth, new_depth], dim=1).contiguous()

                processed_views, normalized_views = prepare_mapanything_views(
                    combined_images_batch,
                    combined_camera_poses,
                    data_norm_type=ctx.mapanything.data_norm_type,
                    device=ctx.mapanything.device,
                    fov_degrees=args.fov_degrees,
                    is_metric_scale=False,
                    depth_z=combined_depth,
                )

                ctx.mapanything._configure_geometric_inputs(
                    use_calibration=True,
                    use_pose=True,
                    use_depth=True,
                )
                try:
                    raw_predictions = ctx.mapanything.base_model.forward(
                        processed_views,
                        memory_efficient_inference=ctx.mapanything.memory_efficient_inference,
                    )
                finally:
                    ctx.mapanything._restore_geometric_inputs()

                recon = ctx.mapanything._stack_predictions(raw_predictions)
                ctx.mapanything._maybe_retain_grad_from_result(recon, normalized_views)

                with POSTPROCESS_LOCK:
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

                new_point_map = point_maps_hw_cpu[idx].unsqueeze(0).unsqueeze(0).to(ctx.device)
                gt_point_maps = torch.cat([ctx.initial_point_map, new_point_map], dim=1).contiguous()

                if valid_masks_hw_cpu is not None:
                    if ctx.initial_mask is None:
                        raise RuntimeError("Missing initial valid mask for device context.")
                    new_mask = valid_masks_hw_cpu[idx].unsqueeze(0).unsqueeze(0).to(ctx.device)
                    gt_valid_masks = torch.cat([ctx.initial_mask, new_mask], dim=1).contiguous()
                else:
                    gt_valid_masks = torch.ones(
                        gt_point_maps.shape[:-1],
                        dtype=torch.bool,
                        device=ctx.device,
                    ).contiguous()

                gt_mesh_data_pair: Dict[str, torch.Tensor] = {
                    "normalized_mesh": ctx.mesh,
                    "gt_points": ctx.gt_points,
                    "gt_point_maps": gt_point_maps,
                    "gt_valid_masks": gt_valid_masks,
                    "depth_z": combined_depth,
                }

                total_loss, loss_components = ctx.loss_fn(
                    recon,
                    gt_mesh_data_pair,
                    combined_images_batch,
                    combined_camera_poses,
                    return_components=True,
                    writer=None,
                    step=None,
                    train_flag=False,
                    point_cloud_dir=str(point_cloud_dir),
                )

                chamfer_value = loss_components.get("chamfer_loss")
                chamfer_value = float(chamfer_value) if chamfer_value is not None else None

                local_results.append(
                    {
                        "second_view_index": idx,
                        "chamfer": chamfer_value,
                        "total_loss": float(total_loss.item()),
                    }
                )

                depth_task_payload = None
                if artifact_queue is not None and not args.skip_depth_png:
                    depth_task_payload = {
                        "output_dir": output_dir,
                        "depth_batch": combined_depth.detach().cpu(),
                        "pair_name": pair_name,
                    }

                glb_task_payload = None
                if artifact_queue is not None and not args.skip_glb:
                    should_export_glb = True
                    with glb_counter_lock:
                        if args.max_glb is not None and glb_export_counter["count"] >= args.max_glb:
                            should_export_glb = False
                        else:
                            glb_export_counter["count"] += 1
                    if should_export_glb:
                        predictions_cpu = _to_cpu_recursive(postprocessed_predictions)
                        glb_task_payload = {
                            "output_path": (output_dir / "glb") / f"{pair_name}.glb",
                            "predictions": predictions_cpu,
                            "as_mesh": args.glb_as_mesh,
                            "confidence_threshold": args.confidence_threshold,
                            "filter_black_bg": True,
                            "black_bg_threshold": 0.2,
                        }

                if depth_task_payload is not None:
                    artifact_queue.put(ArtifactTask(kind="depth_png", payload=depth_task_payload))
                if glb_task_payload is not None:
                    artifact_queue.put(ArtifactTask(kind="glb", payload=glb_task_payload))
                del postprocessed_predictions
                del raw_predictions
        return local_results

    num_contexts = len(device_contexts)
    chunks: List[List[int]] = [[] for _ in range(num_contexts)]
    for order, idx in enumerate(indices_to_eval):
        chunks[order % num_contexts].append(idx)

    if indices_to_eval:
        with ThreadPoolExecutor(max_workers=num_contexts) as executor:
            futures = []
            for ctx, subset in zip(device_contexts, chunks):
                if not subset:
                    continue
                futures.append(executor.submit(process_indices, subset, ctx))
            for future in futures:
                results.extend(future.result())

    results.sort(key=lambda item: item["second_view_index"])

    summary_path = output_dir / "chamfer_results.json"
    summary_payload = {
        "mesh_path": str(args.mesh_path),
        "num_views": args.num_views,
        "fixed_first_view": first_view,
        "confidence_threshold": args.confidence_threshold,
        "results": results,
        "image_hw": [int(images.shape[-2]), int(images.shape[-1])],
        "normalize_method": args.normalize_method,
        "num_gt_samples": args.num_gt_samples,
        "camera_base_radius": args.camera_base_radius,
        "camera_radius_variation": args.camera_radius_variation,
        "camera_radius_mode": args.camera_radius_mode,
        "camera_radius_layers": args.camera_radius_layers,
        "max_glb_exported": glb_export_counter["count"],
        "view_artifacts_saved": not args.skip_view_artifacts,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    plot_path = output_dir / "chamfer_per_second_view.png"
    plot_results(results, first_view, plot_path)
    if not args.skip_visualization:
        interactive_path = output_dir / "camera_heatmap.html"
        export_camera_heatmap_visualisation(
            mesh=mesh_cpu,
            pose_tensor=pose_tensor,
            chamfer_results=results,
            first_view=first_view,
            output_path=interactive_path,
            image_hw=tuple(int(dim) for dim in images.shape[-2:]),
        )

    if artifact_queue is not None:
        for _ in artifact_workers:
            artifact_queue.put(None)
        artifact_queue.join()
        for worker in artifact_workers:
            worker.join()

    print(f"Experiment complete. Results saved to {summary_path} and {plot_path}.")


if __name__ == "__main__":
    main()
