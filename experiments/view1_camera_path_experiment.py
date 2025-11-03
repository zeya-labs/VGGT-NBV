#!/usr/bin/env python3
"""
Visualise the trajectory of a specific view's camera poses across optimisation steps.

The script scans a MapAnything run directory, extracts cam2world matrices from
stored view tensors, and renders an interactive Plotly scene showing:
    - the target mesh,
    - the trajectory of the requested view (default: view 01),
    - an optional reference camera (default: view 00 at the first step).

Example:
    python experiments/view1_camera_path_experiment.py \\
        --run_dir runs/dataset-house3k_bs-1_initv-1_pom-position_only_20251103-142858 \\
        --mesh_path models/House3K_obj/BATCH_7/SET_B/BAT7_SETB_HOUSE33_WTR.obj \\
        --max_steps 120
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from nbv_framework.utils.mesh_utils import load_mesh_as_pytorch3d, normalize_mesh


@dataclass
class CameraSample:
    """Container for a single cam2world matrix identified by step index."""

    step_name: str
    step_number: int
    cam2world: np.ndarray  # (4, 4)

    @property
    def position(self) -> np.ndarray:
        """Return the camera origin in world coordinates."""
        return self.cam2world[:3, 3]

    @property
    def forward(self) -> np.ndarray:
        """Return the forward direction (positive Z axis) in world coordinates."""
        return self.cam2world[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the evolution of a camera view across optimisation steps."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Path to the run directory containing the images/step_XXXXX hierarchy.",
    )
    parser.add_argument(
        "--mesh_path",
        type=Path,
        required=True,
        help="Path to the mesh (.obj or .ply) to render in the centre of the scene.",
    )
    parser.add_argument(
        "--output_html",
        type=Path,
        default=None,
        help="Optional output path for the HTML visualisation. Defaults to <run_dir>/viewXX_trajectory.html.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional maximum number of steps to load starting from step_000000.",
    )
    parser.add_argument(
        "--batch_index",
        type=int,
        default=0,
        help="Batch index to read within each step directory (zero-based).",
    )
    parser.add_argument(
        "--view_index",
        type=int,
        default=1,
        help="View index whose trajectory will be rendered (default: 1 for view_01).",
    )
    parser.add_argument(
        "--reference_view_index",
        type=int,
        default=0,
        help="Index of a reference view to display once (default: 0 for view_00).",
    )
    parser.add_argument(
        "--normalize_method",
        type=str,
        default="none",
        help="Normalization method applied to the mesh (see nbv_framework.utils.mesh_utils.normalize_mesh).",
    )
    parser.add_argument(
        "--orientation_stride",
        type=int,
        default=0,
        help="Stride for drawing camera orientation cones along the trajectory (<=0 disables cones).",
    )
    parser.add_argument(
        "--cone_scale",
        type=float,
        default=0.2,
        help="Relative scale factor for camera orientation cones.",
    )
    parser.add_argument(
        "--marker_size",
        type=float,
        default=2.0,
        help="Marker size for trajectory points.",
    )
    parser.add_argument(
        "--line_width",
        type=float,
        default=1,
        help="Line width for the trajectory polyline.",
    )
    return parser.parse_args()


def safe_torch_load(path: Path) -> torch.Tensor:
    """Load a torch file and return the cam2world tensor."""
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict in {path}, received {type(data).__name__}.")

    cam2world = data.get("camera_poses")
    if not isinstance(cam2world, torch.Tensor):
        raise KeyError(f"'camera_poses' missing or not a tensor in {path}.")
    if cam2world.shape[-2:] != (4, 4):
        raise ValueError(f"Expected cam2world of shape (4, 4); got {cam2world.shape} in {path}.")
    return cam2world.detach().to(dtype=torch.float64, device="cpu")


def iter_step_dirs(images_dir: Path) -> Iterable[Path]:
    """Yield step directories in lexicographic order."""
    if not images_dir.exists():
        raise FileNotFoundError(f"{images_dir} does not exist.")

    step_dirs = sorted(
        (path for path in images_dir.iterdir() if path.is_dir() and path.name.startswith("step_"))
    )
    if not step_dirs:
        raise FileNotFoundError(f"No step_* directories found under {images_dir}.")
    return step_dirs


def collect_camera_samples(
    images_dir: Path,
    batch_index: int,
    view_index: int,
    max_steps: Optional[int],
) -> List[CameraSample]:
    """Collect cam2world matrices for a given view across steps."""
    samples: List[CameraSample] = []
    missing: List[Path] = []

    for idx, step_dir in enumerate(iter_step_dirs(images_dir)):
        if max_steps is not None and idx >= max_steps:
            break

        batch_dir = step_dir / f"batch_{batch_index:03d}"
        pose_path = batch_dir / f"view_{view_index:02d}.pt"
        if not pose_path.exists():
            missing.append(pose_path)
            continue

        cam_tensor = safe_torch_load(pose_path)
        step_number = _parse_step_number(step_dir.name)
        samples.append(
            CameraSample(
                step_name=step_dir.name,
                step_number=step_number,
                cam2world=cam_tensor.numpy(),
            )
        )

    if missing:
        print(f"[WARN] Skipped {len(missing)} missing pose files for view_{view_index:02d}.")
    if not samples:
        raise FileNotFoundError("No camera poses collected; check run_dir, batch_index, and view_index.")

    return samples


def _parse_step_number(step_name: str) -> int:
    """Extract the numeric suffix from a step directory name."""
    try:
        return int(step_name.split("_")[-1])
    except ValueError:
        return 0


def load_reference_pose(
    images_dir: Path,
    step_name: str,
    batch_index: int,
    view_index: int,
) -> Optional[np.ndarray]:
    """Load a reference view's cam2world matrix from the specified step."""
    pose_path = images_dir / step_name / f"batch_{batch_index:03d}" / f"view_{view_index:02d}.pt"
    if not pose_path.exists():
        print(f"[INFO] Reference pose {pose_path} not found; skipping reference view.")
        return None

    cam_tensor = safe_torch_load(pose_path)
    return cam_tensor.numpy()


def mesh_to_numpy(mesh_path: Path, normalize_method: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the mesh and return vertices/faces as numpy arrays."""
    mesh = load_mesh_as_pytorch3d(str(mesh_path))
    if normalize_method != "none":
        mesh = normalize_mesh(mesh, normalize_method)

    verts = mesh.verts_list()[0].detach().cpu().numpy()
    faces = mesh.faces_list()[0].detach().cpu().numpy()
    return verts, faces


def build_plotly_figure(
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    samples: Sequence[CameraSample],
    reference_pose: Optional[np.ndarray],
    *,
    marker_size: float,
    line_width: float,
    orientation_stride: int,
    cone_scale: float,
    view_index: int,
    reference_view_index: int,
):
    """Construct the Plotly figure for the camera trajectory and mesh."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for HTML export. Install it via 'pip install plotly'."
        ) from exc

    positions = np.stack([sample.position for sample in samples], axis=0)
    step_numbers = np.array([sample.step_number for sample in samples], dtype=np.int32)

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=mesh_verts[:, 0],
            y=mesh_verts[:, 1],
            z=mesh_verts[:, 2],
            i=mesh_faces[:, 0],
            j=mesh_faces[:, 1],
            k=mesh_faces[:, 2],
            color="lightgray",
            opacity=0.35,
            name="Mesh",
            hoverinfo="skip",
        )
    )

    hover_texts = [
        f"{sample.step_name} | t=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
        for sample, pos in zip(samples, positions)
    ]

    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="lines",
            line=dict(color="#1f77b4", width=line_width),
            name=f"View {view_index:02d} path",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=step_numbers,
                colorscale="Viridis",
                cmin=float(step_numbers.min()),
                cmax=float(step_numbers.max()),
                colorbar=dict(title="step"),
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            name=f"View {view_index:02d} poses",
        )
    )

    # Highlight start and end positions.
    start_pos = positions[0]
    end_pos = positions[-1]
    fig.add_trace(
        go.Scatter3d(
            x=[start_pos[0]],
            y=[start_pos[1]],
            z=[start_pos[2]],
            mode="markers+text",
            marker=dict(color="#d62728", size=marker_size * 1.5, symbol="diamond"),
            text=[f"start ({samples[0].step_name})"],
            textposition="top center",
            name="Start",
        )
    )
    if samples[-1].step_name != samples[0].step_name:
        fig.add_trace(
            go.Scatter3d(
                x=[end_pos[0]],
                y=[end_pos[1]],
                z=[end_pos[2]],
                mode="markers+text",
                marker=dict(color="#2ca02c", size=marker_size * 1.5, symbol="diamond-open"),
                text=[f"end ({samples[-1].step_name})"],
                textposition="top center",
                name="End",
            )
        )

    # Reference view (assumed fixed).
    if reference_pose is not None:
        ref_pos = reference_pose[:3, 3]
        fig.add_trace(
            go.Scatter3d(
                x=[ref_pos[0]],
                y=[ref_pos[1]],
                z=[ref_pos[2]],
                mode="markers+text",
                marker=dict(color="#ff7f0e", size=marker_size * 1.4, symbol="x"),
                text=[f"view {reference_view_index:02d}"],
                textposition="bottom center",
                name=f"View {reference_view_index:02d}",
            )
        )

    if orientation_stride > 0:
        cone_positions: List[np.ndarray] = []
        cone_vectors: List[np.ndarray] = []
        stride = max(orientation_stride, 1)
        for sample in samples[::stride]:
            forward = sample.forward
            norm = np.linalg.norm(forward)
            if norm < 1e-6:
                continue
            cone_positions.append(sample.position)
            cone_vectors.append(forward / norm)

        if cone_positions and cone_vectors:
            cone_positions_np = np.stack(cone_positions, axis=0)
            cone_vectors_np = np.stack(cone_vectors, axis=0)
            scene_extent = _compute_scene_extent(mesh_verts, positions)
            size_ref = max(scene_extent * cone_scale, 1e-3)
            fig.add_trace(
                go.Cone(
                    x=cone_positions_np[:, 0],
                    y=cone_positions_np[:, 1],
                    z=cone_positions_np[:, 2],
                    u=cone_vectors_np[:, 0],
                    v=cone_vectors_np[:, 1],
                    w=cone_vectors_np[:, 2],
                    sizemode="absolute",
                    sizeref=size_ref,
                    colorscale=[[0.0, "#1f77b4"], [1.0, "#1f77b4"]],
                    showscale=False,
                    name="Orientation",
                    anchor="tip",
                )
            )

    axis_cfg = dict(showbackground=False, showgrid=False, zeroline=False)
    fig.update_layout(
        title=f"Trajectory of view_{view_index:02d}",
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="center", x=0.5),
        scene=dict(
            xaxis=dict(title="X", **axis_cfg),
            yaxis=dict(title="Y", **axis_cfg),
            zaxis=dict(title="Z", **axis_cfg),
            aspectmode="data",
        ),
    )
    return fig


def _compute_scene_extent(mesh_verts: np.ndarray, camera_positions: np.ndarray) -> float:
    """Estimate a characteristic length scale for sizing cones."""
    mesh_extent = float(np.max(np.linalg.norm(mesh_verts, axis=1))) if mesh_verts.size else 1.0
    cam_extent = float(np.max(np.linalg.norm(camera_positions, axis=1))) if camera_positions.size else 1.0
    return max(mesh_extent, cam_extent, 1e-3)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    images_dir = run_dir / "images"

    samples = collect_camera_samples(
        images_dir=images_dir,
        batch_index=args.batch_index,
        view_index=args.view_index,
        max_steps=args.max_steps,
    )

    reference_pose = load_reference_pose(
        images_dir=images_dir,
        step_name=samples[0].step_name,
        batch_index=args.batch_index,
        view_index=args.reference_view_index,
    )

    mesh_verts, mesh_faces = mesh_to_numpy(args.mesh_path, args.normalize_method)

    fig = build_plotly_figure(
        mesh_verts=mesh_verts,
        mesh_faces=mesh_faces,
        samples=samples,
        reference_pose=reference_pose,
        marker_size=args.marker_size,
        line_width=args.line_width,
        orientation_stride=args.orientation_stride,
        cone_scale=args.cone_scale,
        view_index=args.view_index,
        reference_view_index=args.reference_view_index,
    )

    output_html = (
        args.output_html
        if args.output_html is not None
        else run_dir / f"view{args.view_index:02d}_trajectory.html"
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")
    print(f"[INFO] Saved trajectory visualisation to {output_html}")


if __name__ == "__main__":
    main()
