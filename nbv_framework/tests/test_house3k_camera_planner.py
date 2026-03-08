from __future__ import annotations

import pytest
import torch

from nbv_framework.data.house3k_camera import House3KCameraConfig, House3KCameraPlanner


def _base_config(**overrides) -> House3KCameraConfig:
    base = dict(
        up_axis="Y",
        seed=42,
        view_sampling_mode="fixed",
        camera_radius=1.6,
        camera_radius_variation=0.0,
        camera_radius_mode="constant",
        use_manual_camera=False,
        manual_camera_position=None,
        manual_camera_look_at=None,
    )
    base.update(overrides)
    return House3KCameraConfig(**base)


def test_fixed_sampling_mode_ignores_index_for_same_model() -> None:
    planner = House3KCameraPlanner(_base_config(view_sampling_mode="fixed"))
    data_item = {"model_name": "house_1", "batch_name": "BATCH_1", "set_name": "SetA"}

    pose_a, _ = planner.build_camera_poses(
        idx=0,
        data_item=data_item,
        model_name="house_1",
        num_views=4,
    )
    pose_b, _ = planner.build_camera_poses(
        idx=99,
        data_item=data_item,
        model_name="house_1",
        num_views=4,
    )

    assert torch.allclose(pose_a, pose_b)


def test_deterministic_per_call_sampling_changes_with_index() -> None:
    planner = House3KCameraPlanner(
        _base_config(
            view_sampling_mode="deterministic_per_call",
            camera_radius_mode="random",
            camera_radius_variation=0.2,
        )
    )
    data_item = {"model_name": "house_1", "batch_name": "BATCH_1", "set_name": "SetA"}

    pose_a, _ = planner.build_camera_poses(
        idx=0,
        data_item=data_item,
        model_name="house_1",
        num_views=4,
    )
    pose_b, _ = planner.build_camera_poses(
        idx=1,
        data_item=data_item,
        model_name="house_1",
        num_views=4,
    )

    assert not torch.allclose(pose_a, pose_b)


def test_manual_camera_position_overrides_sampling() -> None:
    planner = House3KCameraPlanner(
        _base_config(
            use_manual_camera=True,
            manual_camera_position=[1.0, 2.0, 3.0],
            manual_camera_look_at=[0.0, 0.0, 0.0],
        )
    )

    pose, _ = planner.build_camera_poses(
        idx=0,
        data_item={"model_name": "house_1", "batch_name": "BATCH_1", "set_name": "SetA"},
        model_name="house_1",
        num_views=8,
    )

    assert pose.shape == (1, 7)
    assert torch.allclose(pose[0, :3], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))


def test_invalid_manual_camera_position_shape_raises() -> None:
    planner = House3KCameraPlanner(
        _base_config(
            use_manual_camera=True,
            manual_camera_position=[1.0, 2.0],
        )
    )

    with pytest.raises(ValueError, match="manual camera position expects 3 values"):
        planner.build_camera_poses(
            idx=0,
            data_item={"model_name": "house_1", "batch_name": "BATCH_1", "set_name": "SetA"},
            model_name="house_1",
            num_views=2,
        )
