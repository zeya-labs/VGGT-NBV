from __future__ import annotations

import torch

from nbv_framework.application.services.batch_preparation_service import BatchPreparationService


class _NoopRenderer:
    def render_views(self, **kwargs):
        raise AssertionError("render_views should not be called when cache is complete")


def test_prepare_batch_keeps_tensor_cache_and_selects_views() -> None:
    batch_size = 2
    view_count = 3
    height, width = 4, 4

    initial_images = torch.rand(batch_size, view_count, 3, height, width)
    camera_poses = torch.rand(batch_size, view_count, 7)
    gt_point_maps = torch.rand(batch_size, view_count, height, width, 3)
    gt_valid_masks = torch.ones(batch_size, view_count, height, width, dtype=torch.bool)
    depth_z = torch.rand(batch_size, view_count, height, width)
    depth_z_viz = torch.rand(batch_size, view_count, height, width)

    batch = {
        "inputs": {
            "images": initial_images,
            "camera_poses": camera_poses,
        },
        "targets": {
            "gt_mesh_data": {
                "gt_point_maps": gt_point_maps,
                "gt_valid_masks": gt_valid_masks,
                "depth_z": depth_z,
                "depth_z_viz": depth_z_viz,
            }
        },
        "mesh": {
            "normalized": torch.zeros(batch_size, 1),
        },
        "meta": [
            {"mesh_path": "a.obj", "normalize_method": "unit_sphere"},
            {"mesh_path": "b.obj", "normalize_method": "unit_sphere"},
        ],
    }

    service = BatchPreparationService(
        renderer=_NoopRenderer(),
        dtype=torch.float32,
        mesh_load_workers=1,
        min_initial_views=2,
        max_initial_views=2,
        randomize_initial_views=False,
    )

    prepared = service.prepare(batch, randomize=False)
    assert prepared.initial_images.shape[1] == 2
    assert prepared.camera_poses.shape[1] == 2
    assert prepared.active_view_count == 2
    assert prepared.selection.shape[0] == 2
