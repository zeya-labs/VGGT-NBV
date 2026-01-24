from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from pytorch3d.structures import Meshes

from nbv_framework.models.direct_reconstruction import build_recon_from_point_maps
from nbv_framework.rendering.differentiable_renderer import DifferentiableRenderer
from nbv_framework.training.loss.reconstruction import ReconstructionLoss
from nbv_framework.utils.camera_utils import position_to_pose_tensor
from nbv_framework.utils.mesh_utils import load_and_normalize_mesh, load_mesh_as_pytorch3d
from nbv_framework.utils.render_utils import render_mesh_views


@dataclass(frozen=True)
class CameraInput:
    position: List[float]
    target: Optional[List[float]] = None


@dataclass(frozen=True)
class MeshInfo:
    centroid: List[float]
    scale: float


@dataclass(frozen=True)
class ChamferRecord:
    created_at: str
    loss_chamfer: float
    views: List[dict]


def resolve_mesh_path(mesh_path: str, repo_root: Path) -> Path:
    path = Path(mesh_path)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Mesh not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Mesh path is not a file: {path}")
    return path


def _compute_quantile_normalization(mesh: Meshes) -> MeshInfo:
    verts = mesh.verts_packed()
    centroid = verts.mean(dim=0)
    centered = verts - centroid
    distances = torch.norm(centered, p=2, dim=1)
    scale = torch.quantile(distances, q=0.95).item()
    if scale < 1e-8:
        scale = 1.0
    return MeshInfo(centroid=centroid.tolist(), scale=float(scale))


def compute_mesh_info(mesh_path: Path) -> MeshInfo:
    mesh = load_mesh_as_pytorch3d(str(mesh_path))
    return _compute_quantile_normalization(mesh)


def _camera_poses_from_inputs(
    cameras: List[CameraInput],
    device: torch.device,
    up_axis: str = "Y",
) -> torch.Tensor:
    positions = torch.tensor([camera.position for camera in cameras], dtype=torch.float32, device=device)
    targets = [camera.target if camera.target is not None else [0.0, 0.0, 0.0] for camera in cameras]
    look_at = torch.tensor(targets, dtype=torch.float32, device=device)
    poses = position_to_pose_tensor(positions, up_axis=up_axis, look_at=look_at)
    return poses.unsqueeze(0)


def _point_map_to_image(point_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if point_map.ndim != 3 or point_map.shape[-1] != 3:
        raise ValueError(f"point_map must be [H, W, 3], got {tuple(point_map.shape)}")

    mask = mask.bool()
    if mask.any():
        valid_points = point_map[mask]
        min_vals = valid_points.min(dim=0).values
        max_vals = valid_points.max(dim=0).values
        scale = (max_vals - min_vals).clamp(min=1e-6)
        normalized = (point_map - min_vals) / scale
    else:
        normalized = torch.zeros_like(point_map)

    normalized = normalized.clamp(0.0, 1.0)
    normalized[~mask] = 0.0
    return normalized


def compute_chamfer_record(
    *,
    mesh_path: Path,
    cameras: List[CameraInput],
    output_dir: Path,
    image_size: int = 256,
    fov: float = 60.0,
) -> ChamferRecord:
    if not cameras:
        raise ValueError("At least one camera is required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh_data = load_and_normalize_mesh(
        str(mesh_path),
        normalize_method="quantile",
        num_samples=32768,
    )
    normalized_mesh = mesh_data["normalized_mesh"]
    gt_points = mesh_data["gt_points"]

    normalized_mesh = normalized_mesh.to(device)
    gt_points = gt_points.to(device)

    camera_poses = _camera_poses_from_inputs(cameras, device=device)

    renderer = DifferentiableRenderer(image_size=image_size, fov=fov)
    render_out = render_mesh_views(
        renderer,
        normalized_mesh,
        camera_poses,
        out_rgb=True,
        out_points=True,
        out_mask=True,
    )

    rgb = render_out["rgb"].contiguous()
    point_maps = render_out["points"].contiguous()
    masks = render_out["mask"].contiguous()

    recon_data = build_recon_from_point_maps(
        point_maps,
        camera_poses,
        valid_masks=masks,
    )

    gt_mesh_data = {
        "gt_point_maps": point_maps,
        "gt_valid_masks": masks,
        "gt_points": gt_points,
    }

    loss_fn = ReconstructionLoss(
        chamfer_weight=1.0,
        confidence_weight=0.0,
        viewpoint_weight=0.0,
        pose_penalty_weight=0.0,
        save_point_clouds=True,
    )

    total_loss, components = loss_fn(
        recon_data,
        gt_mesh_data,
        rgb,
        camera_poses,
        return_components=True,
    )

    loss_chamfer = float(components.get("chamfer_loss", total_loss.detach().item()))

    output_dir.mkdir(parents=True, exist_ok=True)
    views: List[dict] = []

    rgb_cpu = rgb.detach().cpu()
    point_cpu = point_maps.detach().cpu()
    masks_cpu = masks.detach().cpu()

    batch_rgb = rgb_cpu[0]
    batch_points = point_cpu[0]
    batch_masks = masks_cpu[0]

    from torchvision.utils import save_image

    for idx in range(batch_rgb.shape[0]):
        rgb_path = output_dir / f"view_{idx:03d}_rgb.png"
        points_path = output_dir / f"view_{idx:03d}_points.png"

        rgb_image = batch_rgb[idx]
        save_image(rgb_image, rgb_path)

        point_image = _point_map_to_image(batch_points[idx], batch_masks[idx])
        save_image(point_image.permute(2, 0, 1), points_path)

        views.append(
            {
                "rgb": f"/results/{output_dir.name}/{rgb_path.name}",
                "points": f"/results/{output_dir.name}/{points_path.name}",
            }
        )

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return ChamferRecord(
        created_at=created_at,
        loss_chamfer=loss_chamfer,
        views=views,
    )


__all__ = [
    "CameraInput",
    "MeshInfo",
    "ChamferRecord",
    "compute_chamfer_record",
    "compute_mesh_info",
    "resolve_mesh_path",
]
