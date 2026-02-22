"""Camera pose sampling and serialization utilities."""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import matrix_to_quaternion

from .coordinate_utils import (
    generate_fibonacci_sphere_points,
    generate_fibonacci_upper_hemisphere_points,
    get_up_vector,
)


class CameraPoseGenerator:
    """Generate camera poses from sphere/hemisphere sampling."""

    def __init__(self, up_axis: str = "Y"):
        self.up_axis = up_axis
        self.up_vector = get_up_vector(up_axis)

    def _generate_poses_from_positions(
        self,
        sphere_positions: np.ndarray,
        seed: int = 0,
        base_radius: float = 2.5,
        radius_variation: float = 0,
        radii: Optional[np.ndarray] = None,
        to_c2w: bool = False,
    ) -> List[Dict[str, List[float]]]:
        num_views = len(sphere_positions)
        rng = np.random.RandomState(seed)

        if radii is not None:
            if len(radii) != num_views:
                raise ValueError("radii length mismatch")
            batch_radii = np.asarray(radii, dtype=np.float32)
        else:
            batch_radii = base_radius + rng.uniform(-radius_variation, radius_variation, size=num_views)
            batch_radii = batch_radii.astype(np.float32)

        positions = sphere_positions * batch_radii[:, np.newaxis]

        eye = torch.tensor(positions, dtype=torch.float32)
        at = torch.zeros((1, 3), dtype=torch.float32)
        up = torch.tensor(self.up_vector, dtype=torch.float32).view(1, 3)
        rotation, _ = look_at_view_transform(eye=eye, at=at, up=up)

        if to_c2w:
            rotation = rotation.transpose(1, 2)

        quaternions_wxyz = matrix_to_quaternion(rotation)
        quats_np = quaternions_wxyz.detach().cpu().numpy()

        poses = []
        for i in range(num_views):
            q = quats_np[i]  # [w, x, y, z]
            quaternion_xyzw = [float(q[1]), float(q[2]), float(q[3]), float(q[0])]
            poses.append(
                {
                    "position": positions[i].tolist(),
                    "quaternion": quaternion_xyzw,
                }
            )
        return poses

    def generate_camera_poses(
        self,
        num_views: int,
        seed: int = 0,
        base_radius: float = 2.6,
        radius_variation: float = 0,
        hemisphere: str = "full",
        radius_mode: str = "random",
        radius_layers: int = 1,
    ) -> List[Dict[str, List[float]]]:
        radius_mode = (radius_mode or "random").lower()
        radii: Optional[np.ndarray] = None
        if radius_mode == "layered" and radius_layers > 1 and radius_variation > 0:
            layers = max(1, int(radius_layers))
            radius_min = max(1e-6, base_radius - radius_variation)
            radius_max = max(radius_min, base_radius + radius_variation)
            layer_radii = np.linspace(radius_min, radius_max, layers, dtype=np.float32)
            counts = np.full(layers, num_views // layers, dtype=int)
            counts[: num_views % layers] += 1
            positions_list: List[np.ndarray] = []
            radii_list: List[np.ndarray] = []
            for layer_idx, count in enumerate(counts):
                if count <= 0:
                    continue
                if hemisphere == "upper":
                    layer_positions, _ = generate_fibonacci_upper_hemisphere_points(
                        count, radius=1.0, up_axis=self.up_axis
                    )
                else:
                    layer_positions, _ = generate_fibonacci_sphere_points(count, radius=1.0)
                positions_list.append(layer_positions)
                radii_list.append(np.full(count, layer_radii[layer_idx], dtype=np.float32))
            if positions_list:
                sphere_positions = np.concatenate(positions_list, axis=0)
                radii = np.concatenate(radii_list, axis=0)
            else:
                if hemisphere == "upper":
                    sphere_positions, _ = generate_fibonacci_upper_hemisphere_points(
                        num_views, radius=1.0, up_axis=self.up_axis
                    )
                else:
                    sphere_positions, _ = generate_fibonacci_sphere_points(num_views, radius=1.0)
                radii = np.full(num_views, base_radius, dtype=np.float32)
        else:
            if hemisphere == "upper":
                sphere_positions, _ = generate_fibonacci_upper_hemisphere_points(
                    num_views, radius=1.0, up_axis=self.up_axis
                )
            else:
                sphere_positions, _ = generate_fibonacci_sphere_points(num_views, radius=1.0)

            if radius_mode == "constant" or radius_variation <= 0:
                radii = np.full(num_views, base_radius, dtype=np.float32)
            elif radius_mode == "random":
                radii = None
            else:
                radii = np.full(num_views, base_radius, dtype=np.float32)

        if radius_mode not in {"constant", "random", "layered"}:
            raise ValueError(
                f"Unsupported radius_mode '{radius_mode}'. Expected 'constant', 'random', or 'layered'."
            )

        return self._generate_poses_from_positions(
            sphere_positions,
            seed,
            base_radius,
            radius_variation,
            radii=radii,
        )

    def generate_random_camera_poses(
        self,
        num_views: int,
        seed: Optional[int] = None,
        base_radius: float = 2.6,
        radius_variation: float = 0.0,
        hemisphere: str = "upper",
        radius_mode: str = "random",
    ) -> List[Dict[str, List[float]]]:
        hemisphere = (hemisphere or "upper").lower()
        if hemisphere not in {"upper", "full"}:
            raise ValueError(f"Unsupported hemisphere '{hemisphere}'. Expected 'upper' or 'full'.")

        radius_mode = (radius_mode or "random").lower()
        if radius_mode not in {"random", "constant"}:
            raise ValueError(f"Unsupported radius_mode '{radius_mode}'. Expected 'random' or 'constant'.")

        rng = np.random.RandomState() if seed is None else np.random.RandomState(seed)
        theta = rng.uniform(0.0, 2.0 * math.pi, size=num_views)
        if hemisphere == "upper":
            up = rng.uniform(0.0, 1.0, size=num_views)
        else:
            up = rng.uniform(-1.0, 1.0, size=num_views)
        radius_plane = np.sqrt(np.clip(1.0 - up**2, 0.0, 1.0))

        axis_index = {"X": 0, "Y": 1, "Z": 2}.get(self.up_axis, 1)
        other_axes = [0, 1, 2]
        other_axes.remove(axis_index)

        directions = np.zeros((num_views, 3), dtype=np.float32)
        directions[:, axis_index] = up.astype(np.float32)
        directions[:, other_axes[0]] = (radius_plane * np.cos(theta)).astype(np.float32)
        directions[:, other_axes[1]] = (radius_plane * np.sin(theta)).astype(np.float32)

        if radius_mode == "constant" or radius_variation <= 0:
            radii = np.full(num_views, base_radius, dtype=np.float32)
        else:
            radius_min = max(1e-6, base_radius - radius_variation)
            radius_max = max(radius_min, base_radius + radius_variation)
            radii = rng.uniform(radius_min, radius_max, size=num_views).astype(np.float32)

        return self._generate_poses_from_positions(
            directions,
            seed=0 if seed is None else seed,
            base_radius=base_radius,
            radius_variation=radius_variation,
            radii=radii,
        )

    def save_camera_poses(self, camera_poses: List[Dict], filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(camera_poses, f, indent=2)

    def load_camera_poses(self, filepath: str) -> List[Dict]:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
