"""Camera-pose planning helpers for House3K dataset samples."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from nbv_framework.infrastructure.utils.camera_utils import CameraPoseGenerator, position_to_pose_tensor

ManualCameraArray = Union[Sequence[float], Sequence[Sequence[float]]]
ManualCameraValue = Optional[
    Union[
        ManualCameraArray,
        Dict[Union[str, int, Tuple[str, str, str]], ManualCameraArray],
    ]
]


@dataclass(frozen=True)
class House3KCameraConfig:
    up_axis: str
    seed: int
    view_sampling_mode: str
    camera_radius: float
    camera_radius_variation: float
    camera_radius_mode: str
    use_manual_camera: bool
    manual_camera_position: ManualCameraValue
    manual_camera_look_at: ManualCameraValue


class House3KCameraPlanner:
    """Encapsulates House3K camera-pose sampling rules."""

    def __init__(self, config: House3KCameraConfig) -> None:
        self.config = config
        self._camera_generator: Optional[CameraPoseGenerator] = None

    def build_camera_poses(
        self,
        *,
        idx: int,
        data_item: Dict[str, Any],
        model_name: str,
        num_views: int,
    ) -> Tuple[torch.Tensor, List[Dict[str, List[float]]]]:
        if self.config.use_manual_camera:
            manual_positions = self._resolve_manual_camera_positions(idx, data_item)
            if manual_positions is not None:
                manual_look_at = self._resolve_manual_camera_look_at(idx, data_item)
                manual_camera_pose = position_to_pose_tensor(
                    manual_positions,
                    up_axis=self.config.up_axis,
                    look_at=manual_look_at,
                )
                return manual_camera_pose.detach(), self._poses_tensor_to_list(manual_camera_pose)

        seed = self._resolve_view_seed(model_name, idx)
        camera_poses_list = self._generate_camera_poses(num_views, seed=seed)
        camera_pose_rows = [
            torch.tensor(pose["position"] + pose["quaternion"], dtype=torch.float32)
            for pose in camera_poses_list
        ]
        camera_poses_tensor = torch.stack(camera_pose_rows, dim=0)
        return camera_poses_tensor, camera_poses_list

    def _resolve_view_seed(self, model_name: str, idx: int) -> Optional[int]:
        mode = self.config.view_sampling_mode
        base_seed = self.config.seed

        if mode == "fixed":
            seed_material = f"{model_name}|{base_seed}"
        elif mode == "deterministic_per_call":
            seed_material = f"{model_name}|{idx}|{base_seed}"
        elif mode == "fully_random":
            return None
        else:
            raise ValueError(f"Unknown view_sampling_mode: {self.config.view_sampling_mode}")

        digest = hashlib.md5(seed_material.encode("utf-8")).hexdigest()
        return int(digest, 16) % (2**32 - 1)

    def _generate_camera_poses(
        self,
        num_views: int,
        seed: Optional[int] = None,
    ) -> List[Dict[str, List[float]]]:
        if self._camera_generator is None:
            self._camera_generator = CameraPoseGenerator(up_axis=self.config.up_axis)

        return self._camera_generator.generate_random_camera_poses(
            num_views,
            seed=seed,
            hemisphere="upper",
            base_radius=self.config.camera_radius,
            radius_variation=self.config.camera_radius_variation,
            radius_mode=self.config.camera_radius_mode,
        )

    @staticmethod
    def _poses_tensor_to_list(camera_poses: torch.Tensor) -> List[Dict[str, List[float]]]:
        return [
            {
                "position": camera_poses[i, :3].tolist(),
                "quaternion": camera_poses[i, 3:].tolist(),
            }
            for i in range(camera_poses.shape[0])
        ]

    def _resolve_manual_config(self, config: ManualCameraValue, idx: int, data_item: Dict[str, Any]):
        if config is None:
            return None
        if callable(config):
            return config(data_item, idx)
        if isinstance(config, dict):
            keys_to_try = [
                data_item.get("model_name"),
                (
                    data_item.get("batch_name"),
                    data_item.get("set_name"),
                    data_item.get("model_name"),
                ),
                idx,
            ]
            for key in keys_to_try:
                if key in config:
                    return config[key]
            return None
        return config

    def _resolve_manual_camera_positions(
        self,
        idx: int,
        data_item: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        value = self._resolve_manual_config(self.config.manual_camera_position, idx, data_item)
        if value is None:
            return None
        return _to_xyz_tensor(value, label="manual camera position")

    def _resolve_manual_camera_look_at(
        self,
        idx: int,
        data_item: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        value = self._resolve_manual_config(self.config.manual_camera_look_at, idx, data_item)
        if value is None:
            return None
        return _to_xyz_tensor(value, label="manual camera look_at")


def _to_xyz_tensor(value: Any, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 1:
        if tensor.numel() != 3:
            raise ValueError(f"{label} expects 3 values, but received shape {tuple(tensor.shape)}")
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2 and tensor.shape[1] == 3:
        pass
    else:
        raise ValueError(f"{label} must have shape [N, 3] or [3], got {tuple(tensor.shape)}")
    return tensor
