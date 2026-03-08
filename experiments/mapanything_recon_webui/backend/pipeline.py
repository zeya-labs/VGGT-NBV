"""Core backend pipeline for input preparation and MapAnything reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import os

# Keep runtime caches in writable temp paths for sandboxed environments.
_DEFAULT_KEOPS_CACHE = Path("/tmp/keops_cache_mapanything_webui")
_DEFAULT_TORCH_EXT = Path("/tmp/torch_extensions_mapanything_webui")
_DEFAULT_KEOPS_CACHE.mkdir(parents=True, exist_ok=True)
_DEFAULT_TORCH_EXT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("KEOPS_CACHE_FOLDER", str(_DEFAULT_KEOPS_CACHE))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(_DEFAULT_TORCH_EXT))

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from nbv_framework.workflows import BatchPreparationUseCase
from nbv_framework.reconstruction import ReconstructionData
from nbv_framework.adapters.depth.depth_visualization_adapter import (
    DepthVisualizationAdapter,
)
from nbv_framework.adapters.mesh_repository.pytorch3d_mesh_repository_adapter import (
    PyTorch3DMeshRepositoryAdapter,
)
from nbv_framework.adapters.renderer.pytorch3d_renderer_adapter import (
    PyTorch3DRendererAdapter,
)
from nbv_framework.data.collate_functions import custom_nbv_collate_fn
from nbv_framework.data.house3k_camera import (
    House3KCameraConfig,
    House3KCameraPlanner,
)
from nbv_framework.data.house3k_sample_builder import build_house3k_sample
from nbv_framework.models.scene_encoder.mapanything_encoder import MapAnythingWrapper
from nbv_framework.infrastructure.rendering.differentiable_renderer import DifferentiableRenderer
from nbv_framework.infrastructure.utils.mesh_utils import load_and_normalize_mesh

from cache import PreparedRun
from schemas import PrepareInputsRequest


@dataclass(frozen=True)
class PrepareResult:
    prepared_run: PreparedRun
    rgb_urls: List[str]
    depth_urls: List[str]
    camera_poses: List[List[float]]
    timings: Dict[str, float]


@dataclass(frozen=True)
class ReconstructResult:
    ply_url: str
    num_points: int
    num_points_before_sampling: int
    num_points_gt: int
    num_points_recon: int
    display_mode: DisplayMode
    timings: Dict[str, float]


DisplayMode = Literal["both", "gt_only", "recon_only"]


_MODEL_LOCK = Lock()
_MODEL_INSTANCE: Optional[MapAnythingWrapper] = None
_MODEL_DEVICE: Optional[str] = None


def discover_mesh_roots(repo_root: Path) -> List[Path]:
    env_roots = os.environ.get("NBV_MESH_ROOTS")
    roots: List[Path] = []
    if env_roots:
        for item in env_roots.split(os.pathsep):
            item = item.strip()
            if not item:
                continue
            roots.append(Path(item).expanduser().resolve())

    default_roots = [
        repo_root / "models" / "House3K_obj",
        repo_root / "models",
        repo_root,
    ]
    roots.extend(default_roots)

    unique: List[Path] = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        key = str(resolved)
        if key in seen:
            continue
        if not resolved.exists() or not resolved.is_dir():
            continue
        seen.add(key)
        unique.append(resolved)

    return unique


def resolve_mesh_root(root: Optional[str], repo_root: Path, candidates: Sequence[Path]) -> Path:
    if root is None or not root.strip():
        if not candidates:
            raise FileNotFoundError("No mesh roots available")
        return candidates[0]

    root_path = Path(root).expanduser()
    if not root_path.is_absolute():
        root_path = (repo_root / root_path).resolve()
    else:
        root_path = root_path.resolve()

    if not root_path.exists() or not root_path.is_dir():
        raise FileNotFoundError(f"Mesh root does not exist: {root_path}")
    return root_path


def list_mesh_files(
    root: Path,
    *,
    extensions: Sequence[str],
    limit: int,
) -> Tuple[List[Path], bool]:
    normalized_ext = set()
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        normalized_ext.add(ext)
    if not normalized_ext:
        normalized_ext = {".obj", ".ply"}

    result: List[Path] = []
    truncated = False

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in normalized_ext:
            continue
        result.append(path)
        if len(result) >= limit:
            truncated = True
            break

    result.sort()
    return result, truncated


def resolve_mesh_path(mesh_path: str, repo_root: Path) -> Path:
    path = Path(mesh_path).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Mesh not found: {path}")
    if path.suffix.lower() not in {".obj", ".ply"}:
        raise ValueError(f"Unsupported mesh type: {path.suffix}")
    return path


def path_to_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _build_house3k_data_item(mesh_path: Path) -> Dict[str, str]:
    model_name = mesh_path.stem
    set_name = mesh_path.parent.name if mesh_path.parent is not None else "unknown_set"
    batch_name = (
        mesh_path.parent.parent.name if mesh_path.parent is not None and mesh_path.parent.parent is not None else "unknown_batch"
    )
    return {
        "obj_path": str(mesh_path),
        "model_name": model_name,
        "set_name": set_name,
        "batch_name": batch_name,
    }


def _build_preparation_use_case(
    *,
    image_size: int,
    fov: float,
    num_views: int,
) -> BatchPreparationUseCase:
    renderer = DifferentiableRenderer(image_size=image_size, fov=fov)
    return BatchPreparationUseCase(
        renderer=PyTorch3DRendererAdapter(renderer),
        mesh_repository=PyTorch3DMeshRepositoryAdapter(),
        depth_visualizer=DepthVisualizationAdapter(),
        mesh_load_workers=0,
        min_initial_views=num_views,
        max_initial_views=num_views,
        randomize_initial_views=False,
    )


def _to_results_url(path: Path, results_dir: Path) -> str:
    rel = path.resolve().relative_to(results_dir.resolve()).as_posix()
    return f"/results/{rel}"


def _save_prepare_previews(
    *,
    output_dir: Path,
    results_dir: Path,
    images: torch.Tensor,
    depth_viz: Optional[torch.Tensor],
    show_depth: bool,
) -> Tuple[List[str], List[str]]:
    preview_dir = output_dir / "inputs"
    preview_dir.mkdir(parents=True, exist_ok=True)

    rgb_urls: List[str] = []
    depth_urls: List[str] = []

    batch_images = images[0]
    num_views = batch_images.shape[0]

    for view_idx in range(num_views):
        rgb_path = preview_dir / f"view_{view_idx:03d}_rgb.png"
        save_image(batch_images[view_idx].clamp(0.0, 1.0), rgb_path)
        rgb_urls.append(_to_results_url(rgb_path, results_dir))

    if show_depth and depth_viz is not None:
        batch_depth = depth_viz[0]
        for view_idx in range(batch_depth.shape[0]):
            depth_path = preview_dir / f"view_{view_idx:03d}_depth.png"
            depth_img = batch_depth[view_idx].unsqueeze(0).clamp(0.0, 1.0)
            save_image(depth_img, depth_path)
            depth_urls.append(_to_results_url(depth_path, results_dir))

    return rgb_urls, depth_urls


def prepare_inputs_for_run(
    *,
    request: PrepareInputsRequest,
    run_id: str,
    created_at: str,
    output_dir: Path,
    results_dir: Path,
    repo_root: Path,
) -> PrepareResult:
    timings: Dict[str, float] = {}
    image_size = int(request.render.image_size)
    if image_size % 14 != 0:
        raise ValueError(
            f"render.image_size must be divisible by 14 for MapAnything (got {image_size})."
        )

    resolve_started = perf_counter()
    mesh_path = resolve_mesh_path(request.mesh_path, repo_root)
    timings["resolve_mesh_path_s"] = perf_counter() - resolve_started

    model_data_item = _build_house3k_data_item(mesh_path)

    mesh_started = perf_counter()
    mesh_data = load_and_normalize_mesh(
        str(mesh_path),
        normalize_method=request.render.normalize_method,
        num_samples=request.render.num_samples,
    )
    timings["load_and_normalize_mesh_s"] = perf_counter() - mesh_started

    planner_started = perf_counter()
    planner = House3KCameraPlanner(
        House3KCameraConfig(
            up_axis=str(request.sampling.up_axis).upper(),
            seed=int(request.sampling.seed),
            view_sampling_mode=str(request.sampling.view_sampling_mode),
            camera_radius=float(request.sampling.camera_radius),
            camera_radius_variation=float(request.sampling.camera_radius_variation),
            camera_radius_mode=str(request.sampling.camera_radius_mode),
            use_manual_camera=bool(request.sampling.use_manual_camera),
            manual_camera_position=request.sampling.manual_camera_position,
            manual_camera_look_at=request.sampling.manual_camera_look_at,
        )
    )
    camera_poses_tensor, _ = planner.build_camera_poses(
        idx=int(request.sampling.scene_index),
        data_item=model_data_item,
        model_name=model_data_item["model_name"],
        num_views=int(request.num_views),
    )
    timings["sample_camera_poses_s"] = perf_counter() - planner_started

    sample = build_house3k_sample(
        data_item=model_data_item,
        mesh_path=str(mesh_path),
        gt_mesh_data=mesh_data,
        camera_poses_tensor=camera_poses_tensor,
    )

    collate_started = perf_counter()
    batch = custom_nbv_collate_fn([sample])
    timings["collate_s"] = perf_counter() - collate_started

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch["inputs"]["camera_poses"] = batch["inputs"]["camera_poses"].to(device=device, dtype=torch.float32)
    mesh_batch = batch.get("mesh", {}).get("normalized")
    if mesh_batch is not None:
        batch["mesh"]["normalized"] = mesh_batch.to(device)

    prepare_started = perf_counter()
    use_case = _build_preparation_use_case(
        image_size=image_size,
        fov=float(request.render.fov),
        num_views=int(request.num_views),
    )
    with torch.inference_mode():
        prepared = use_case.prepare(batch, randomize=False)
    timings["batch_prepare_s"] = perf_counter() - prepare_started

    export_started = perf_counter()
    initial_images = prepared.initial_images.detach().cpu().to(dtype=torch.float32).contiguous()
    camera_poses = prepared.camera_poses.detach().cpu().to(dtype=torch.float32).contiguous()

    depth_z = prepared.depth_z
    if depth_z is not None:
        depth_z = depth_z.detach().cpu().to(dtype=torch.float32).contiguous()

    depth_z_viz = prepared.gt_mesh_data.get("depth_z_viz")
    if torch.is_tensor(depth_z_viz):
        depth_z_viz = depth_z_viz.detach().cpu().to(dtype=torch.float32).contiguous()
    else:
        depth_z_viz = None

    gt_point_maps = prepared.trimmed_gt_mesh_data.get("gt_point_maps")
    if not torch.is_tensor(gt_point_maps):
        raise RuntimeError("Prepared batch missing tensor gt_point_maps for WebUI reconstruction.")
    gt_point_maps = gt_point_maps.detach().cpu().to(dtype=torch.float32).contiguous()

    gt_valid_masks = prepared.trimmed_gt_mesh_data.get("gt_valid_masks")
    if not torch.is_tensor(gt_valid_masks):
        raise RuntimeError("Prepared batch missing tensor gt_valid_masks for WebUI reconstruction.")
    gt_valid_masks = gt_valid_masks.detach().cpu().to(dtype=torch.bool).contiguous()

    rgb_urls, depth_urls = _save_prepare_previews(
        output_dir=output_dir,
        results_dir=results_dir,
        images=initial_images,
        depth_viz=depth_z_viz,
        show_depth=bool(request.show_depth),
    )
    timings["export_prepare_previews_s"] = perf_counter() - export_started

    prepared_run = PreparedRun(
        run_id=run_id,
        mesh_path=str(mesh_path),
        model_name=model_data_item["model_name"],
        created_at=created_at,
        output_dir=output_dir,
        image_size=image_size,
        fov=float(request.render.fov),
        device_used=str(device),
        initial_images=initial_images,
        camera_poses=camera_poses,
        depth_z=depth_z,
        depth_z_viz=depth_z_viz,
        gt_point_maps=gt_point_maps,
        gt_valid_masks=gt_valid_masks,
    )

    camera_pose_list = camera_poses[0].tolist()
    return PrepareResult(
        prepared_run=prepared_run,
        rgb_urls=rgb_urls,
        depth_urls=depth_urls,
        camera_poses=camera_pose_list,
        timings=timings,
    )


def _get_mapanything_model(device: torch.device) -> MapAnythingWrapper:
    global _MODEL_INSTANCE
    global _MODEL_DEVICE

    with _MODEL_LOCK:
        if _MODEL_INSTANCE is None:
            _MODEL_INSTANCE = MapAnythingWrapper(
                model_name="facebook/map-anything",
                revision="6f3a25bfbb8fcc799176bb01e9d07dfb49d5416a",
                local_files_only=True,
            )
            _MODEL_INSTANCE.eval()
            _MODEL_INSTANCE.to(device)
            _MODEL_DEVICE = str(device)
            return _MODEL_INSTANCE

        if _MODEL_DEVICE != str(device):
            _MODEL_INSTANCE.to(device)
            _MODEL_DEVICE = str(device)

        return _MODEL_INSTANCE


def _normalize_map_to_bshw(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
    target_hw: Tuple[int, int],
    as_mask: bool,
) -> torch.Tensor:
    out = tensor
    if out.dim() == 5 and out.shape[-1] == 1:
        out = out.squeeze(-1)
    elif out.dim() == 5 and out.shape[2] == 1:
        out = out.squeeze(2)

    if out.dim() != 4:
        raise ValueError(f"Expected map tensor with 4 dims after squeeze, got {tuple(tensor.shape)}")
    if out.shape[0] != batch_size or out.shape[1] != num_views:
        raise ValueError(
            "Map tensor batch/view shape mismatch: "
            f"{tuple(out.shape[:2])} vs {(batch_size, num_views)}"
        )

    if out.shape[2:] != target_hw:
        flat = out.reshape(batch_size * num_views, 1, out.shape[2], out.shape[3])
        if as_mask:
            flat = flat.to(dtype=torch.float32)
            resized = F.interpolate(flat, size=target_hw, mode="nearest")
            out = resized.reshape(batch_size, num_views, target_hw[0], target_hw[1]) >= 0.5
        else:
            resized = F.interpolate(flat.to(dtype=torch.float32), size=target_hw, mode="bilinear", align_corners=False)
            out = resized.reshape(batch_size, num_views, target_hw[0], target_hw[1])

    if as_mask:
        return out.to(dtype=torch.bool)
    return out


def _sample_points(points: torch.Tensor, max_points: int) -> torch.Tensor:
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected points shape [N, 3], got {tuple(points.shape)}")
    if max_points <= 0 or points.shape[0] == 0:
        return points.new_empty((0, 3))
    if points.shape[0] <= max_points:
        return points
    indices = torch.randperm(points.shape[0], device=points.device)[:max_points]
    return points.index_select(0, indices)


def _extract_recon_points(
    recon_data: ReconstructionData | Dict[str, torch.Tensor],
    *,
    conf_threshold: float,
) -> Tuple[torch.Tensor, int]:
    if isinstance(recon_data, ReconstructionData):
        world_points = recon_data.recon_world_points
        conf = recon_data.recon_conf
        non_ambiguous = recon_data.recon_mask
    else:
        world_points = recon_data.get("world_points")
        if world_points is None:
            world_points = recon_data.get("world_points_from_depth")
        if world_points is None:
            raise KeyError("reconstruct output does not contain world_points/world_points_from_depth")

        conf = recon_data.get("world_points_conf")
        if conf is None:
            conf = recon_data.get("depth_conf")
        if conf is None:
            conf = recon_data.get("conf")
        non_ambiguous = recon_data.get("non_ambiguous_mask")

    if world_points.dim() != 5 or world_points.shape[-1] != 3:
        raise ValueError(f"Expected world_points shape [B, S, H, W, 3], got {tuple(world_points.shape)}")

    world_points = world_points.to(dtype=torch.float32)
    batch_size, num_views, h_pts, w_pts, _ = world_points.shape
    mask = torch.isfinite(world_points).all(dim=-1)

    if torch.is_tensor(non_ambiguous):
        nb_mask = _normalize_map_to_bshw(
            non_ambiguous,
            batch_size=batch_size,
            num_views=num_views,
            target_hw=(h_pts, w_pts),
            as_mask=True,
        )
        mask &= nb_mask

    if torch.is_tensor(conf):
        conf_map = _normalize_map_to_bshw(
            conf,
            batch_size=batch_size,
            num_views=num_views,
            target_hw=(h_pts, w_pts),
            as_mask=False,
        )
        mask &= torch.isfinite(conf_map)
        mask &= conf_map >= float(conf_threshold)

    if not mask.any():
        mask = torch.isfinite(world_points).all(dim=-1)

    selected_points = world_points[mask]
    return selected_points, int(selected_points.shape[0])


def _extract_gt_points(
    gt_point_maps: torch.Tensor,
    gt_valid_masks: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    if gt_point_maps.dim() != 5 or gt_point_maps.shape[-1] != 3:
        raise ValueError(
            "Expected gt_point_maps shape [B, S, H, W, 3], "
            f"got {tuple(gt_point_maps.shape)}"
        )

    points = gt_point_maps.to(dtype=torch.float32)
    batch_size, num_views, h_pts, w_pts, _ = points.shape
    mask = torch.isfinite(points).all(dim=-1)

    valid_masks = _normalize_map_to_bshw(
        gt_valid_masks,
        batch_size=batch_size,
        num_views=num_views,
        target_hw=(h_pts, w_pts),
        as_mask=True,
    )
    mask &= valid_masks
    selected_points = points[mask]
    return selected_points, int(selected_points.shape[0])


def _build_display_cloud(
    *,
    recon_points: torch.Tensor,
    gt_points: torch.Tensor,
    display_mode: DisplayMode,
    max_points: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    if max_points <= 0:
        raise ValueError(f"max_points must be > 0, got {max_points}")

    device = recon_points.device if recon_points.numel() > 0 else gt_points.device
    dtype = recon_points.dtype if recon_points.numel() > 0 else gt_points.dtype
    red = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    green = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

    recon_before = int(recon_points.shape[0])
    gt_before = int(gt_points.shape[0])

    if display_mode == "recon_only":
        recon_selected = _sample_points(recon_points, max_points)
        gt_selected = gt_points.new_empty((0, 3))
        num_before = recon_before
    elif display_mode == "gt_only":
        gt_selected = _sample_points(gt_points, max_points)
        recon_selected = recon_points.new_empty((0, 3))
        num_before = gt_before
    elif display_mode == "both":
        recon_budget = min(recon_before, max_points // 2)
        gt_budget = min(gt_before, max_points - recon_budget)
        remaining = max_points - (recon_budget + gt_budget)
        if remaining > 0 and recon_before > recon_budget:
            recon_extra = min(recon_before - recon_budget, remaining)
            recon_budget += recon_extra
            remaining -= recon_extra
        if remaining > 0 and gt_before > gt_budget:
            gt_extra = min(gt_before - gt_budget, remaining)
            gt_budget += gt_extra

        recon_selected = _sample_points(recon_points, recon_budget)
        gt_selected = _sample_points(gt_points, gt_budget)
        num_before = gt_before + recon_before
    else:
        raise ValueError(
            f"Unsupported display_mode `{display_mode}`. Expected one of: both, gt_only, recon_only."
        )

    point_chunks: List[torch.Tensor] = []
    color_chunks: List[torch.Tensor] = []
    if gt_selected.shape[0] > 0:
        point_chunks.append(gt_selected)
        color_chunks.append(green.unsqueeze(0).repeat(gt_selected.shape[0], 1))
    if recon_selected.shape[0] > 0:
        point_chunks.append(recon_selected)
        color_chunks.append(red.unsqueeze(0).repeat(recon_selected.shape[0], 1))

    if not point_chunks:
        raise RuntimeError(
            f"No valid points to visualize for mode={display_mode}. "
            f"gt_before={gt_before}, recon_before={recon_before}"
        )

    points = torch.cat(point_chunks, dim=0)
    colors = torch.cat(color_chunks, dim=0)
    return (
        points,
        colors,
        num_before,
        int(gt_selected.shape[0]),
        int(recon_selected.shape[0]),
    )


def _write_colored_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors_uint8 = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    data = np.concatenate([points.astype(np.float32), colors_uint8.astype(np.float32)], axis=1)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        np.savetxt(handle, data, fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"])


def reconstruct_and_export(
    *,
    prepared_run: PreparedRun,
    results_dir: Path,
    conf_threshold: float,
    max_points: int,
    use_depth_input: bool,
    display_mode: DisplayMode,
) -> ReconstructResult:
    timings: Dict[str, float] = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_started = perf_counter()
    model = _get_mapanything_model(device)
    timings["load_model_s"] = perf_counter() - model_started

    run_started = perf_counter()
    images = prepared_run.initial_images.to(device=device, dtype=torch.float32)
    camera_poses = prepared_run.camera_poses.to(device=device, dtype=torch.float32)
    depth_z = None
    if use_depth_input and prepared_run.depth_z is not None:
        depth_z = prepared_run.depth_z.to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        recon_data = model.reconstruct_and_evaluate(
            images,
            camera_poses,
            depth_z=depth_z,
            is_metric_scale=True,
            fov_degrees=float(prepared_run.fov),
        )
    timings["reconstruct_s"] = perf_counter() - run_started

    cloud_started = perf_counter()
    recon_points, _ = _extract_recon_points(
        recon_data,
        conf_threshold=float(conf_threshold),
    )
    gt_points, _ = _extract_gt_points(
        prepared_run.gt_point_maps.to(device=device, dtype=torch.float32),
        prepared_run.gt_valid_masks.to(device=device, dtype=torch.bool),
    )
    points, colors, num_before, num_points_gt, num_points_recon = _build_display_cloud(
        recon_points=recon_points,
        gt_points=gt_points,
        display_mode=display_mode,
        max_points=int(max_points),
    )
    points_np = points.detach().cpu().numpy()
    colors_np = colors.detach().cpu().numpy()

    depth_tag = "with_depth" if use_depth_input else "without_depth"
    ply_path = (
        prepared_run.output_dir
        / "reconstruction"
        / f"point_cloud_{display_mode}_{depth_tag}.ply"
    )
    _write_colored_ply(ply_path, points_np, colors_np)
    timings["export_point_cloud_s"] = perf_counter() - cloud_started

    return ReconstructResult(
        ply_url=_to_results_url(ply_path, results_dir),
        num_points=int(points_np.shape[0]),
        num_points_before_sampling=num_before,
        num_points_gt=num_points_gt,
        num_points_recon=num_points_recon,
        display_mode=display_mode,
        timings=timings,
    )
