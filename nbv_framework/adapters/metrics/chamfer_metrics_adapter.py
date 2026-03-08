"""Chamfer-style metrics adapter."""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch

from nbv_framework.training.losses import ChamferDistance


class ChamferMetricsAdapter:
    def __init__(
        self,
        metrics: Iterable[str],
        *,
        max_points_per_cloud: int = 32768,
        use_log_warp: bool = False,
        point_cloud_dir_name: str = "point_clouds",
    ) -> None:
        self._metric_fns: Dict[str, ChamferDistance] = {}
        for metric in metrics:
            self._metric_fns[str(metric)] = ChamferDistance(
                max_points_per_cloud=max_points_per_cloud,
                save_point_clouds=False,
                point_cloud_dir_name=point_cloud_dir_name,
                use_log_warp=use_log_warp,
                distance_type=str(metric),
            )

    def compute(self, pred_points_list: List[torch.Tensor], gt_points: torch.Tensor) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for name, metric_fn in self._metric_fns.items():
            results[name] = float(metric_fn(pred_points_list, gt_points))
        return results
