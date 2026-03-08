"""Lightweight smoke checks runnable without pytest."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Allow direct execution via:
# python nbv_framework/tests/smoke_checks.py
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from nbv_framework.config import NBVConfig, validate_config
from nbv_framework.data.house3k_camera import House3KCameraConfig, House3KCameraPlanner
from nbv_framework.training.test_metrics import summarize_values
from nbv_framework.tests.test_architecture_import_boundaries import test_architecture_import_boundaries


def run_smoke_checks() -> None:
    cfg = NBVConfig()
    validate_config(cfg)

    test_architecture_import_boundaries()

    mean, std, count = summarize_values([1.0, 2.0, 3.0])
    assert count == 3
    assert abs(mean - 2.0) < 1e-12
    assert abs(std - torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).std(unbiased=False).item()) < 1e-12

    planner = House3KCameraPlanner(
        House3KCameraConfig(
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
    )
    pose_a, _ = planner.build_camera_poses(
        idx=0,
        data_item={},
        model_name="house_1",
        num_views=3,
    )
    pose_b, _ = planner.build_camera_poses(
        idx=7,
        data_item={},
        model_name="house_1",
        num_views=3,
    )
    assert torch.allclose(pose_a, pose_b)


if __name__ == "__main__":
    run_smoke_checks()
    print("smoke checks passed")
