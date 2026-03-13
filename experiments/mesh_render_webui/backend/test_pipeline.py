from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest
import torch

from experiments.mesh_render_webui.backend import app as app_module
from experiments.mesh_render_webui.backend import pipeline


def test_list_meshes_filters_and_sorts(tmp_path: Path) -> None:
    repo_root = tmp_path
    mesh_root = repo_root / "models" / "House3K_obj"
    (mesh_root / "B").mkdir(parents=True)
    (mesh_root / "A").mkdir(parents=True)
    (mesh_root / "B" / "mesh2.obj").write_text("# obj", encoding="utf-8")
    (mesh_root / "A" / "mesh1.ply").write_text("ply", encoding="utf-8")
    (mesh_root / "A" / "ignore.txt").write_text("nope", encoding="utf-8")

    meshes = pipeline.list_meshes(repo_root, mesh_root=mesh_root)

    assert [item["relative_path"] for item in meshes] == [
        "models/House3K_obj/A/mesh1.ply",
        "models/House3K_obj/B/mesh2.obj",
    ]


def test_build_camera_path_preserves_first_view() -> None:
    camera = pipeline.ViewerCameraSpec(position=[2.0, 1.0, 0.5], target=[0.0, 0.0, 0.0])

    orbit = pipeline.build_camera_path(camera, trajectory_mode="orbit", frame_count=8)
    swing = pipeline.build_camera_path(camera, trajectory_mode="swing", frame_count=8)

    assert orbit[0].position == pytest.approx(camera.position)
    assert orbit[0].target == pytest.approx(camera.target)
    assert swing[0].position == pytest.approx(camera.position)
    assert all(frame.target == pytest.approx(camera.target) for frame in orbit)


def test_api_meshes_endpoint_returns_relative_paths(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    mesh_root = repo_root / "models" / "House3K_obj" / "Set_A"
    mesh_root.mkdir(parents=True)
    (mesh_root / "demo.obj").write_text("# demo", encoding="utf-8")

    monkeypatch.setattr(app_module, "REPO_ROOT", repo_root)

    client = TestClient(app_module.app)
    response = client.get("/api/meshes")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "models/House3K_obj/Set_A/demo.obj",
            "relative_path": "models/House3K_obj/Set_A/demo.obj",
            "name": "demo.obj",
        }
    ]


def test_build_camera_pose_tensor_shape() -> None:
    camera = pipeline.ViewerCameraSpec(position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])
    pose = pipeline._camera_pose_tensor(camera, device=torch.device("cpu"))
    assert tuple(pose.shape) == (1, 7)
