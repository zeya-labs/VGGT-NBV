from __future__ import annotations

import math

import torch

from nbv_framework.reconstruction import (
    build_recon_from_depth_z,
    build_recon_from_point_maps,
)


def _points_from_depth_and_translation(
    depth_z: torch.Tensor,
    translation: torch.Tensor,
    *,
    fov_degrees: float,
) -> torch.Tensor:
    height, width = depth_z.shape
    fov_radians = math.radians(float(fov_degrees))
    fy = 0.5 * float(height) / math.tan(fov_radians / 2.0)
    fx = 0.5 * float(width) / math.tan(fov_radians / 2.0)
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0

    u = torch.arange(width, dtype=depth_z.dtype)
    v = torch.arange(height, dtype=depth_z.dtype)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")

    x = (u_grid - cx) / fx * depth_z
    y = (v_grid - cy) / fy * depth_z
    cam_points = torch.stack((x, y, depth_z), dim=-1)
    return cam_points + translation.view(1, 1, 3)


def test_build_recon_from_depth_z_matches_point_map_path() -> None:
    fov_degrees = 60.0
    depth_z = torch.tensor(
        [
            [
                [[[1.0], [1.5], [2.0]], [[1.2], [0.8], [1.1]]],
                [[[0.7], [1.7], [0.9]], [[1.4], [1.1], [1.8]]],
            ]
        ],
        dtype=torch.float32,
    )
    valid_masks = torch.tensor(
        [[[[True, True, False], [True, False, True]], [[True, False, True], [True, True, True]]]],
        dtype=torch.bool,
    )
    camera_poses = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, -0.5, 0.25, 0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )

    first_view = _points_from_depth_and_translation(
        depth_z[0, 0, ..., 0],
        camera_poses[0, 0, :3],
        fov_degrees=fov_degrees,
    )
    second_view = _points_from_depth_and_translation(
        depth_z[0, 1, ..., 0],
        camera_poses[0, 1, :3],
        fov_degrees=fov_degrees,
    )
    point_maps = torch.stack((first_view, second_view), dim=0).unsqueeze(0)
    point_maps = point_maps.masked_fill(~valid_masks.unsqueeze(-1), 0.0)

    expected = build_recon_from_point_maps(
        point_maps=point_maps,
        valid_masks=valid_masks,
    )
    actual = build_recon_from_depth_z(
        camera_poses=camera_poses,
        depth_z=depth_z,
        valid_masks=valid_masks,
        fov_degrees=fov_degrees,
        xy_signs=(1, 1),
    )

    assert torch.allclose(actual.recon_world_points, expected.recon_world_points)
    assert torch.equal(actual.recon_mask, expected.recon_mask)
    assert torch.equal(actual.recon_conf, expected.recon_conf)


def test_build_recon_from_depth_z_infers_mask_from_depth() -> None:
    camera_poses = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    depth_z = torch.tensor(
        [
            [
                [[1.0], [0.0]],
                [[float("nan")], [2.0]],
            ]
        ],
        dtype=torch.float32,
    )

    recon = build_recon_from_depth_z(
        camera_poses=camera_poses,
        depth_z=depth_z,
    )

    expected_mask = torch.tensor([[[True, False], [False, True]]], dtype=torch.bool)
    assert torch.equal(recon.recon_mask, expected_mask)
    assert torch.equal(recon.recon_conf, expected_mask.to(dtype=torch.float32))
    invalid_points = recon.recon_world_points.masked_select(~expected_mask.unsqueeze(-1))
    assert torch.allclose(invalid_points, torch.zeros_like(invalid_points))


def test_build_recon_from_depth_z_matches_renderer_output_with_default_signs() -> None:
    try:
        from pytorch3d.utils import ico_sphere
    except ImportError:
        return

    from nbv_framework.geometry.camera_pose import position_to_pose_tensor
    from nbv_framework.adapters.renderer.pytorch3d_renderer_adapter import (
        PyTorch3DRendererAdapter,
    )
    from nbv_framework.infrastructure.rendering.differentiable_renderer import (
        DifferentiableRenderer,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 48
    fov_degrees = 60.0

    positions = torch.tensor(
        [
            [2.3, 0.8, 1.9],
            [-2.1, 1.1, 1.6],
            [0.6, 2.4, 1.7],
        ],
        dtype=torch.float32,
        device=device,
    )
    camera_poses = position_to_pose_tensor(positions, up_axis="Y").unsqueeze(0)

    mesh = ico_sphere(1, device=device)
    renderer = PyTorch3DRendererAdapter(
        DifferentiableRenderer(image_size=image_size, fov=fov_degrees).to(device)
    )
    render_out = renderer.render_views(
        mesh_batch=mesh,
        camera_poses=camera_poses,
        out_rgb=False,
        out_points=True,
        out_mask=True,
        out_depth=True,
    )

    recon_from_points = build_recon_from_point_maps(
        point_maps=render_out["points"],
        valid_masks=render_out["mask"],
    )
    recon_from_depth = build_recon_from_depth_z(
        camera_poses=camera_poses,
        depth_z=render_out["depth"],
        valid_masks=render_out["mask"],
        fov_degrees=fov_degrees,
    )

    mask = render_out["mask"].unsqueeze(-1)
    diff = (recon_from_depth.recon_world_points - recon_from_points.recon_world_points).abs()
    valid_diff = diff.masked_select(mask)

    assert torch.equal(recon_from_depth.recon_mask, recon_from_points.recon_mask)
    assert torch.equal(recon_from_depth.recon_conf, recon_from_points.recon_conf)
    assert float(valid_diff.mean()) < 1e-3
    assert float(valid_diff.max()) < 5e-3
