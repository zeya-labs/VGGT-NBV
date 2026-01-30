from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


def parse_mesh_metadata(
    meta: Any,
) -> Tuple[Optional[List[Optional[str]]], Optional[List[Optional[str]]]]:
    if not isinstance(meta, list):
        return None, None
    mesh_paths: List[Optional[str]] = []
    normalize_methods: List[Optional[str]] = []
    for entry in meta:
        if isinstance(entry, dict):
            mesh_paths.append(entry.get("mesh_path"))
            normalize_methods.append(entry.get("normalize_method"))
        else:
            mesh_paths.append(None)
            normalize_methods.append(None)
    return mesh_paths, normalize_methods


def trim_gt_mesh_data(
    gt_mesh_data: Dict[str, torch.Tensor],
    selection: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    trimmed = dict(gt_mesh_data)
    if selection is None:
        return trimmed

    selection_device = selection
    for key in ("gt_point_maps", "gt_valid_masks", "depth_z", "depth_z_viz"):
        value = trimmed.get(key)
        if value is None:
            continue
        selection_device = selection.to(value.device)
        trimmed[key] = value.index_select(1, selection_device).contiguous()
    return trimmed
