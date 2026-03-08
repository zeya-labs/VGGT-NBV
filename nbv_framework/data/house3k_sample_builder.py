"""Sample-construction helpers for House3K dataset."""

from __future__ import annotations

from typing import Any, Dict

import torch


def build_house3k_sample(
    *,
    data_item: Dict[str, Any],
    mesh_path: str,
    gt_mesh_data: Dict[str, Any],
    camera_poses_tensor: torch.Tensor,
) -> Dict[str, Any]:
    gt_supervision = dict(gt_mesh_data)
    gt_supervision.pop("original_mesh", None)
    gt_supervision.pop("normalized_mesh", None)
    gt_supervision.pop("mesh_path", None)

    metadata = {
        "data_item": data_item,
        "mesh_path": mesh_path,
        "batch_name": data_item["batch_name"],
        "set_name": data_item["set_name"],
        "model_name": data_item["model_name"],
        "normalize_method": gt_supervision.get("normalize_method"),
        "num_samples": gt_supervision.get("num_samples"),
    }

    return {
        "inputs": {
            "camera_poses": camera_poses_tensor,
        },
        "targets": {
            "gt_mesh_data": gt_supervision,
        },
        "mesh": {
            "normalized": gt_mesh_data.get("normalized_mesh"),
        },
        "meta": metadata,
    }
