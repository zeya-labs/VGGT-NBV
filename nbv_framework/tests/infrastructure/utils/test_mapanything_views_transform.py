from __future__ import annotations

import math

import torch

from nbv_framework.infrastructure.utils.mapanything_views import (
    transform_points_ref0_to_global,
    transform_prediction_pts3d_ref0_to_global,
)


def test_ref0_identity_no_change() -> None:
    points_ref = torch.tensor(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]],
        dtype=torch.float32,
    )  # [B=1, H=1, W=2, 3]
    ref_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    ref_trans = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

    points_global = transform_points_ref0_to_global(points_ref, ref_quat, ref_trans)

    assert torch.allclose(points_global, points_ref, atol=1e-6)


def test_ref0_rigid_transform_correctness() -> None:
    points_ref = torch.tensor(
        [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]],
        dtype=torch.float32,
    )
    theta = math.pi / 2.0
    ref_quat = torch.tensor(
        [[0.0, 0.0, math.sin(theta / 2.0), math.cos(theta / 2.0)]],
        dtype=torch.float32,
    )  # +90 deg around z axis (xyzw)
    ref_trans = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

    points_global = transform_points_ref0_to_global(points_ref, ref_quat, ref_trans)
    expected = torch.tensor(
        [[[[1.0, 3.0, 3.0], [0.0, 2.0, 3.0]]]],
        dtype=torch.float32,
    )

    assert torch.allclose(points_global, expected, atol=1e-5)


def test_batchwise_transform() -> None:
    points_ref = torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]],
            [[[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]],
        ],
        dtype=torch.float32,
    )  # [B=2, H=1, W=2, 3]
    ref_quat = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    ref_trans = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )

    points_global = transform_points_ref0_to_global(points_ref, ref_quat, ref_trans)
    expected = torch.tensor(
        [
            [[[1.0, 0.0, 0.0], [2.0, 1.0, 1.0]]],
            [[[2.0, 4.0, 2.0], [3.0, 5.0, 3.0]]],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(points_global, expected, atol=1e-6)


def test_gradient_flow_through_transform() -> None:
    points_ref = torch.randn(2, 3, 4, 3, dtype=torch.float32, requires_grad=True)
    ref_quat = torch.randn(2, 4, dtype=torch.float32, requires_grad=True)
    ref_trans = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)

    points_global = transform_points_ref0_to_global(points_ref, ref_quat, ref_trans)
    loss = points_global.pow(2).mean()
    loss.backward()

    assert points_ref.grad is not None
    assert ref_quat.grad is not None
    assert ref_trans.grad is not None
    assert torch.isfinite(points_ref.grad).all()
    assert torch.isfinite(ref_quat.grad).all()
    assert torch.isfinite(ref_trans.grad).all()


def test_prediction_list_transform_integrity() -> None:
    pred0_pts = torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float32)
    pred1_pts = torch.tensor([[[[0.0, 1.0, 0.0]]]], dtype=torch.float32)
    pred0_conf = torch.tensor([[[0.7]]], dtype=torch.float32)
    pred1_conf = torch.tensor([[[0.9]]], dtype=torch.float32)
    pred0_mask = torch.tensor([[[True]]], dtype=torch.bool)
    pred1_mask = torch.tensor([[[False]]], dtype=torch.bool)

    predictions = [
        {"pts3d": pred0_pts.clone(), "conf": pred0_conf, "non_ambiguous_mask": pred0_mask},
        {"pts3d": pred1_pts.clone(), "conf": pred1_conf, "non_ambiguous_mask": pred1_mask},
    ]
    views = [
        {
            "camera_pose_quats": torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
            "camera_pose_trans": torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float32),
        }
    ]

    transformed = transform_prediction_pts3d_ref0_to_global(predictions, views)

    assert transformed[0] is not predictions[0]
    assert transformed[1] is not predictions[1]
    assert torch.allclose(
        transformed[0]["pts3d"],
        torch.tensor([[[[6.0, 0.0, 0.0]]]], dtype=torch.float32),
        atol=1e-6,
    )
    assert torch.allclose(
        transformed[1]["pts3d"],
        torch.tensor([[[[5.0, 1.0, 0.0]]]], dtype=torch.float32),
        atol=1e-6,
    )
    assert torch.equal(transformed[0]["conf"], pred0_conf)
    assert torch.equal(transformed[1]["conf"], pred1_conf)
    assert torch.equal(transformed[0]["non_ambiguous_mask"], pred0_mask)
    assert torch.equal(transformed[1]["non_ambiguous_mask"], pred1_mask)

    assert torch.equal(predictions[0]["pts3d"], pred0_pts)
    assert torch.equal(predictions[1]["pts3d"], pred1_pts)
