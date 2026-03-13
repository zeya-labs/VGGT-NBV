from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import threading
from typing import Dict, List, Optional, Sequence
import uuid

import imageio.v2 as imageio
import torch

from nbv_framework.adapters.renderer.pytorch3d_renderer_adapter import (
    PyTorch3DRendererAdapter,
)
from nbv_framework.geometry.camera_pose import position_to_pose_tensor
from nbv_framework.infrastructure.rendering.differentiable_renderer import (
    DifferentiableRenderer,
)
from nbv_framework.infrastructure.utils.mesh_utils import load_mesh_as_pytorch3d, normalize_mesh


SUPPORTED_SUFFIXES = {".obj", ".ply"}
MESH_ROOT_RELATIVE = Path("models/House3K_obj")
DEFAULT_MESH_RELATIVE_PATH = (
    Path("models/House3K_obj/BATCH_1/Set_A/BAT1_SETA_HOUSE1.obj")
)
DEFAULT_NORMALIZE_METHOD = "quantile"
DEFAULT_UP_AXIS = "Y"
VIDEO_FRAME_CHUNK = 24
SWING_AMPLITUDE_DEGREES = 30.0


@dataclass(frozen=True)
class MeshInfo:
    centroid: List[float]
    scale: float


@dataclass(frozen=True)
class ViewerCameraSpec:
    position: List[float]
    target: List[float]


@dataclass(frozen=True)
class CaptureRecord:
    record_id: str
    kind: str
    mesh_path: str
    created_at: str
    preview_url: Optional[str]
    image_url: Optional[str]
    video_url: Optional[str]
    metadata_url: str


@dataclass(frozen=True)
class VideoRenderSettings:
    trajectory_mode: str
    duration_sec: float
    fps: int
    image_size: int
    fov: float


@dataclass
class _MeshBundle:
    mesh_info: MeshInfo
    normalized_mesh_cpu: torch.nn.Module


_MESH_BUNDLE_CACHE: Dict[str, _MeshBundle] = {}
_MESH_BUNDLE_CACHE_LOCK = threading.Lock()


def resolve_mesh_path(mesh_path: str, repo_root: Path) -> Path:
    repo_root = repo_root.resolve()
    path = Path(mesh_path)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()

    if not path.is_relative_to(repo_root):
        raise ValueError(f"Mesh path must stay inside repository root: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Mesh not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Mesh path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported mesh type: {path.suffix}. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )
    return path


def list_meshes(repo_root: Path, mesh_root: Optional[Path] = None) -> List[dict]:
    repo_root = repo_root.resolve()
    mesh_root = (mesh_root or repo_root / MESH_ROOT_RELATIVE).resolve()
    if not mesh_root.exists():
        return []

    mesh_paths = [
        path
        for path in mesh_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    mesh_paths.sort(key=lambda path: path.relative_to(repo_root).as_posix())

    return [
        {
            "id": path.relative_to(repo_root).as_posix(),
            "relative_path": path.relative_to(repo_root).as_posix(),
            "name": path.name,
        }
        for path in mesh_paths
    ]


def _compute_quantile_mesh_info(mesh) -> MeshInfo:
    verts = mesh.verts_packed()
    centroid = verts.mean(dim=0)
    centered = verts - centroid
    distances = torch.norm(centered, dim=1, p=2)
    scale = float(torch.quantile(distances, q=0.95).item())
    if scale < 1e-8:
        scale = 1.0
    return MeshInfo(centroid=centroid.tolist(), scale=scale)


def _get_mesh_bundle(mesh_path: Path) -> _MeshBundle:
    cache_key = str(mesh_path.resolve())
    with _MESH_BUNDLE_CACHE_LOCK:
        cached = _MESH_BUNDLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mesh = load_mesh_as_pytorch3d(str(mesh_path))
    mesh_info = _compute_quantile_mesh_info(mesh)
    normalized_mesh = normalize_mesh(mesh, DEFAULT_NORMALIZE_METHOD)
    bundle = _MeshBundle(mesh_info=mesh_info, normalized_mesh_cpu=normalized_mesh)

    with _MESH_BUNDLE_CACHE_LOCK:
        _MESH_BUNDLE_CACHE[cache_key] = bundle
    return bundle


def compute_mesh_info(mesh_path: Path) -> MeshInfo:
    return _get_mesh_bundle(mesh_path).mesh_info


def _to_viewer_camera_spec(camera: Dict[str, Sequence[float]]) -> ViewerCameraSpec:
    return ViewerCameraSpec(
        position=[float(value) for value in camera["position"]],
        target=[float(value) for value in camera["target"]],
    )


def _camera_pose_tensor(camera: ViewerCameraSpec, device: torch.device) -> torch.Tensor:
    position = torch.tensor(camera.position, dtype=torch.float32, device=device)
    target = torch.tensor(camera.target, dtype=torch.float32, device=device)
    return position_to_pose_tensor(
        position.unsqueeze(0),
        up_axis=DEFAULT_UP_AXIS,
        look_at=target.unsqueeze(0),
    )


def _rotation_matrix_y(angle_radians: torch.Tensor, device: torch.device) -> torch.Tensor:
    cos_theta = torch.cos(angle_radians)
    sin_theta = torch.sin(angle_radians)
    zeros = torch.zeros_like(cos_theta)
    ones = torch.ones_like(cos_theta)
    return torch.stack(
        [
            torch.stack([cos_theta, zeros, sin_theta], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sin_theta, zeros, cos_theta], dim=-1),
        ],
        dim=-2,
    ).to(device=device)


def _stabilize_orbit_offset(offset: torch.Tensor) -> torch.Tensor:
    horizontal = torch.linalg.norm(offset[[0, 2]])
    if float(horizontal) >= 1e-4:
        return offset

    adjusted = offset.clone()
    radius = float(torch.linalg.norm(offset))
    adjusted[0] = radius if radius > 1e-4 else 1.0
    adjusted[2] = 0.0
    return adjusted


def build_camera_path(
    camera: ViewerCameraSpec,
    *,
    trajectory_mode: str,
    frame_count: int,
) -> List[ViewerCameraSpec]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")

    position = torch.tensor(camera.position, dtype=torch.float32)
    target = torch.tensor(camera.target, dtype=torch.float32)
    offset = position - target
    offset = _stabilize_orbit_offset(offset)

    if trajectory_mode == "orbit":
        angles = torch.arange(frame_count, dtype=torch.float32) * (2.0 * math.pi / frame_count)
    elif trajectory_mode == "swing":
        if frame_count == 1:
            angles = torch.zeros(1, dtype=torch.float32)
        else:
            phase = torch.arange(frame_count, dtype=torch.float32) * (
                2.0 * math.pi / (frame_count - 1)
            )
            amplitude = math.radians(SWING_AMPLITUDE_DEGREES)
            angles = amplitude * torch.sin(phase)
    else:
        raise ValueError(f"Unsupported trajectory_mode: {trajectory_mode}")

    rotation = _rotation_matrix_y(angles, device=offset.device)
    rotated_offsets = torch.einsum("nij,j->ni", rotation, offset)
    positions = rotated_offsets + target.unsqueeze(0)
    targets = target.unsqueeze(0).expand(frame_count, -1)

    return [
        ViewerCameraSpec(
            position=positions[index].tolist(),
            target=targets[index].tolist(),
        )
        for index in range(frame_count)
    ]


def _render_rgb_frames(
    mesh_path: Path,
    camera_path: Sequence[ViewerCameraSpec],
    *,
    image_size: int,
    fov: float,
) -> List[torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_bundle = _get_mesh_bundle(mesh_path)
    mesh_batch = mesh_bundle.normalized_mesh_cpu.to(device)
    renderer = PyTorch3DRendererAdapter(
        DifferentiableRenderer(image_size=image_size, fov=fov).to(device)
    )

    frames: List[torch.Tensor] = []
    pose_tensors = [
        _camera_pose_tensor(camera, device=device).squeeze(0)
        for camera in camera_path
    ]
    all_poses = torch.stack(pose_tensors, dim=0)

    with torch.no_grad():
        for start in range(0, all_poses.shape[0], VIDEO_FRAME_CHUNK):
            pose_chunk = all_poses[start : start + VIDEO_FRAME_CHUNK].unsqueeze(0)
            render_out = renderer.render_views(
                mesh_batch=mesh_batch,
                camera_poses=pose_chunk,
                out_rgb=True,
                out_points=False,
                out_mask=False,
                out_depth=False,
            )
            chunk_rgb = render_out["rgb"][0].detach().cpu()
            frames.extend(chunk_rgb[frame_idx] for frame_idx in range(chunk_rgb.shape[0]))

    return frames


def _tensor_to_uint8_image(frame: torch.Tensor) -> torch.Tensor:
    return (
        frame.permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .to(torch.uint8)
        .contiguous()
    )


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _new_record_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _write_metadata(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def capture_image_record(
    *,
    mesh_path: Path,
    camera: ViewerCameraSpec,
    output_root: Path,
    image_size: int,
    fov: float,
) -> CaptureRecord:
    record_id = _new_record_id("image")
    record_dir = output_root / record_id
    record_dir.mkdir(parents=True, exist_ok=True)

    frames = _render_rgb_frames(
        mesh_path,
        [camera],
        image_size=image_size,
        fov=fov,
    )
    image_path = record_dir / "capture.png"
    imageio.imwrite(image_path, _tensor_to_uint8_image(frames[0]).numpy())

    metadata = {
        "record_id": record_id,
        "kind": "image",
        "mesh_path": str(mesh_path),
        "created_at": _timestamp(),
        "camera": asdict(camera),
        "render": {
            "image_size": image_size,
            "fov": fov,
            "normalize_method": DEFAULT_NORMALIZE_METHOD,
        },
    }
    metadata_path = record_dir / "metadata.json"
    _write_metadata(metadata_path, metadata)

    created_at = metadata["created_at"]
    return CaptureRecord(
        record_id=record_id,
        kind="image",
        mesh_path=str(mesh_path),
        created_at=created_at,
        preview_url=f"/results/{record_id}/capture.png",
        image_url=f"/results/{record_id}/capture.png",
        video_url=None,
        metadata_url=f"/results/{record_id}/metadata.json",
    )


def render_video_record(
    *,
    mesh_path: Path,
    camera: ViewerCameraSpec,
    output_root: Path,
    settings: VideoRenderSettings,
) -> CaptureRecord:
    frame_count = max(1, int(round(settings.duration_sec * settings.fps)))
    camera_path = build_camera_path(
        camera,
        trajectory_mode=settings.trajectory_mode,
        frame_count=frame_count,
    )

    record_id = _new_record_id("video")
    record_dir = output_root / record_id
    frames_dir = record_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frames = _render_rgb_frames(
        mesh_path,
        camera_path,
        image_size=settings.image_size,
        fov=settings.fov,
    )

    frame_arrays = []
    preview_relpath = f"/results/{record_id}/frames/frame_0000.png"
    for index, frame in enumerate(frames):
        frame_array = _tensor_to_uint8_image(frame).numpy()
        frame_arrays.append(frame_array)
        imageio.imwrite(frames_dir / f"frame_{index:04d}.png", frame_array)

    video_path = record_dir / "video.mp4"
    imageio.mimwrite(
        video_path,
        frame_arrays,
        fps=settings.fps,
        codec="libx264",
        quality=8,
    )

    metadata = {
        "record_id": record_id,
        "kind": "video",
        "mesh_path": str(mesh_path),
        "created_at": _timestamp(),
        "camera": asdict(camera),
        "trajectory": [asdict(frame_camera) for frame_camera in camera_path],
        "render": {
            "trajectory_mode": settings.trajectory_mode,
            "duration_sec": settings.duration_sec,
            "fps": settings.fps,
            "image_size": settings.image_size,
            "fov": settings.fov,
            "normalize_method": DEFAULT_NORMALIZE_METHOD,
        },
    }
    metadata_path = record_dir / "metadata.json"
    _write_metadata(metadata_path, metadata)

    created_at = metadata["created_at"]
    return CaptureRecord(
        record_id=record_id,
        kind="video",
        mesh_path=str(mesh_path),
        created_at=created_at,
        preview_url=preview_relpath,
        image_url=preview_relpath,
        video_url=f"/results/{record_id}/video.mp4",
        metadata_url=f"/results/{record_id}/metadata.json",
    )


__all__ = [
    "CaptureRecord",
    "DEFAULT_MESH_RELATIVE_PATH",
    "MeshInfo",
    "MESH_ROOT_RELATIVE",
    "VideoRenderSettings",
    "ViewerCameraSpec",
    "build_camera_path",
    "capture_image_record",
    "compute_mesh_info",
    "list_meshes",
    "render_video_record",
    "resolve_mesh_path",
]

