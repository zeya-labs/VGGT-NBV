"""Test-time evaluation helpers for NBVTrainer."""

from __future__ import annotations

from loguru import logger
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from ..models.direct_reconstruction import build_recon_from_point_maps
from ..pipeline.step_ops import evaluate_candidate_pose, sample_random_positions
from ..training.loss import ChamferDistance
from ..utils.camera_utils import position_to_pose_tensor



class NBVTrainerTestMixin:
    """Test-time evaluation utilities for NBVTrainer."""

    def on_test_epoch_start(self) -> None:
        self._test_metric_fns = self._build_test_metric_fns()
        self._test_metric_values: Dict[str, Dict[str, List[float]]] = {
            name: {"model": [], "random": []} for name in self._test_metric_fns
        }

    def _build_test_metric_fns(self) -> Dict[str, ChamferDistance]:
        chamfer = self.loss_fn.chamfer_regularizer.chamfer
        max_points = getattr(chamfer, "max_points_per_cloud", 32768)
        use_log_warp = getattr(chamfer, "use_log_warp", False)
        point_cloud_dir_name = getattr(chamfer, "point_cloud_dir_name", "point_clouds")

        metric_fns: Dict[str, ChamferDistance] = {}
        for metric in self.test_chamfer_metrics:
            metric_fns[metric] = ChamferDistance(
                max_points_per_cloud=max_points,
                save_point_clouds=False,
                point_cloud_dir_name=point_cloud_dir_name,
                use_log_warp=use_log_warp,
                distance_type=metric,
            )
        return metric_fns

    def _compute_test_metrics(
        self,
        *,
        gt_mesh_data: Dict[str, torch.Tensor],
        combined_images_batch: torch.Tensor,
        combined_camera_poses: torch.Tensor,
        depth_z: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        point_maps = gt_mesh_data.get("gt_point_maps")
        valid_masks = gt_mesh_data.get("gt_valid_masks")
        if point_maps is None or valid_masks is None:
            raise RuntimeError("gt_point_maps or gt_valid_masks missing for test metrics.")

        recon_data = build_recon_from_point_maps(
            point_maps=point_maps,
            camera_poses=combined_camera_poses,
            valid_masks=valid_masks,
            depth_z=depth_z,
        )

        chamfer_reg = self.loss_fn.chamfer_regularizer
        pred_points_list, _ = chamfer_reg.extractor(
            recon_data=recon_data,
            combined_images_batch=combined_images_batch,
            confidence_threshold=chamfer_reg.confidence_threshold,
            source=chamfer_reg.point_source,
            gt_valid_masks=gt_mesh_data.get("gt_valid_masks"),
        )

        gt_points = gt_mesh_data.get("gt_points")
        if gt_points is None:
            raise RuntimeError("gt_points missing for test metrics.")

        results: Dict[str, float] = {}
        for name, metric_fn in self._test_metric_fns.items():
            metric_value = metric_fn(pred_points_list, gt_points)
            results[name] = float(metric_value)
        return results

    def _gather_values_across_ranks(self, values: List[float]) -> List[float]:
        if self.trainer.world_size <= 1:
            return values
        if not dist.is_available() or not dist.is_initialized():
            return values

        gathered: List[Optional[List[float]]] = [None for _ in range(self.trainer.world_size)]
        dist.all_gather_object(gathered, list(values))

        merged: List[float] = []
        for part in gathered:
            if part:
                merged.extend(part)
        return merged

    def _save_test_metrics_table(
        self,
        *,
        summary: Dict[str, Dict[str, Tuple[float, float, int]]],
        save_dir: Path,
    ) -> Optional[Path]:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.warning("Matplotlib not available; skip metrics table image: {}", exc)
            return None

        display_map = {
            "cd": "CD",
            "dcd": "DCD",
            "emd": "EMD",
            "geomloss": "Geomloss (Trainloss)",
        }
        metric_names = list(summary.keys())
        columns = ["Policy"] + [display_map.get(name, name) for name in metric_names]

        def _fmt(mean: float, std: float) -> str:
            if not (math.isfinite(mean) and math.isfinite(std)):
                return "nan"
            return f"{mean:.6f}+/-{std:.6f}"

        ours_row = ["Ours_xyz"]
        rand_row = ["Random_xyz"]
        for name in metric_names:
            model_mean, model_std, _ = summary[name]["model"]
            rand_mean, rand_std, _ = summary[name]["random"]
            ours_row.append(_fmt(model_mean, model_std))
            rand_row.append(_fmt(rand_mean, rand_std))

        table_data = [ours_row, rand_row]

        fig_width = max(6, 1.6 * len(columns))
        fig, ax = plt.subplots(figsize=(fig_width, 2.2))
        ax.axis("off")
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 1.6)

        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "test_metrics_summary.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def test_step(self, batch: Dict, batch_idx: int):
        prepared = self._prepare_batch(batch)

        with torch.no_grad():
            policy_inference = self._infer_next_pose(
                initial_images=prepared.initial_images,
                camera_poses_batch=prepared.camera_poses,
                depth_z_batch=prepared.depth_z,
            )

            policy_eval = evaluate_candidate_pose(
                renderer=self.renderer,
                loss_fn=self.loss_fn,
                pose=policy_inference.next_camera_pose,
                initial_images=prepared.initial_images,
                camera_poses_batch=prepared.camera_poses,
                gt_mesh_data=prepared.trimmed_gt_mesh_data,
                mesh_batch=prepared.mesh_batch,
                point_cloud_dir=None,
            )

            combined_images = torch.cat(
                [prepared.initial_images, policy_eval.new_images.unsqueeze(1)], dim=1
            )
            combined_camera_poses = torch.cat(
                [prepared.camera_poses, policy_inference.next_camera_pose.unsqueeze(1)], dim=1
            )
            model_metrics = self._compute_test_metrics(
                gt_mesh_data=policy_eval.gt_mesh_data,
                combined_images_batch=combined_images,
                combined_camera_poses=combined_camera_poses,
                depth_z=policy_eval.depth_z,
            )

            random_positions = sample_random_positions(
                batch_size=prepared.initial_images.shape[0],
                device=prepared.initial_images.device,
                loss_fn=self.loss_fn,
            )
            random_pose = position_to_pose_tensor(random_positions)
            random_eval = evaluate_candidate_pose(
                renderer=self.renderer,
                loss_fn=self.loss_fn,
                pose=random_pose,
                initial_images=prepared.initial_images,
                camera_poses_batch=prepared.camera_poses,
                gt_mesh_data=prepared.trimmed_gt_mesh_data,
                mesh_batch=prepared.mesh_batch,
                point_cloud_dir=None,
            )
            random_combined_images = torch.cat(
                [prepared.initial_images, random_eval.new_images.unsqueeze(1)], dim=1
            )
            random_combined_camera_poses = torch.cat(
                [prepared.camera_poses, random_pose.unsqueeze(1)], dim=1
            )
            random_metrics = self._compute_test_metrics(
                gt_mesh_data=random_eval.gt_mesh_data,
                combined_images_batch=random_combined_images,
                combined_camera_poses=random_combined_camera_poses,
                depth_z=random_eval.depth_z,
            )

            for name, value in model_metrics.items():
                self._test_metric_values[name]["model"].append(value)
            for name, value in random_metrics.items():
                self._test_metric_values[name]["random"].append(value)

        return {
            **{f"{name}_model": val for name, val in model_metrics.items()},
            **{f"{name}_random": val for name, val in random_metrics.items()},
        }

    def on_test_epoch_end(self) -> None:
        def _summarize(values: List[float]) -> Tuple[float, float, int]:
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                return float("nan"), float("nan"), 0
            valid = np.isfinite(arr)
            if not np.any(valid):
                return float("nan"), float("nan"), 0
            return float(arr[valid].mean()), float(arr[valid].std()), int(valid.sum())

        summary: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
        for name, values in self._test_metric_values.items():
            model_values = self._gather_values_across_ranks(values["model"])
            rand_values = self._gather_values_across_ranks(values["random"])
            model_mean, model_std, model_n = _summarize(model_values)
            rand_mean, rand_std, rand_n = _summarize(rand_values)
            summary[name] = {
                "model": (model_mean, model_std, model_n),
                "random": (rand_mean, rand_std, rand_n),
            }

        if getattr(self.trainer, "is_global_zero", True):
            for name, stats in summary.items():
                model_mean, model_std, model_n = stats["model"]
                rand_mean, rand_std, rand_n = stats["random"]

                self.log(f"test/{name}_model_mean", model_mean, prog_bar=True)
                self.log(f"test/{name}_model_std", model_std, prog_bar=False)
                self.log(f"test/{name}_random_mean", rand_mean, prog_bar=True)
                self.log(f"test/{name}_random_std", rand_std, prog_bar=False)

                logger.info(
                    "Test {} (mean ± std) | Model: {:.6f} ± {:.6f} (N={}) | "
                    "Random: {:.6f} ± {:.6f} (N={})",
                    name,
                    model_mean,
                    model_std,
                    model_n,
                    rand_mean,
                    rand_std,
                    rand_n,
                )

            table_path = self._save_test_metrics_table(
                summary=summary,
                save_dir=Path(self.log_dir) / "test_metrics",
            )
            if table_path is not None:
                logger.info("Saved test metrics table to {}", table_path)
