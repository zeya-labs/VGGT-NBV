"""Helpers for distributed test-metric aggregation and reporting."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch.distributed as dist
from loguru import logger

MetricStats = Tuple[float, float, int]
MetricSummary = Dict[str, Dict[str, MetricStats]]
MetricValues = Dict[str, Dict[str, List[float]]]


def init_test_metric_values(metric_names: Sequence[str]) -> MetricValues:
    return {metric: {"model": [], "random": []} for metric in metric_names}


def append_test_metric_values(
    metric_values: MetricValues,
    *,
    model_metrics: Dict[str, float],
    random_metrics: Dict[str, float],
) -> None:
    for name, value in model_metrics.items():
        metric_values.setdefault(name, {"model": [], "random": []})
        metric_values[name]["model"].append(float(value))
    for name, value in random_metrics.items():
        metric_values.setdefault(name, {"model": [], "random": []})
        metric_values[name]["random"].append(float(value))


def gather_values_across_ranks(values: List[float], *, world_size: int) -> List[float]:
    if world_size <= 1:
        return values
    if not dist.is_available() or not dist.is_initialized():
        return values

    gathered: List[List[float] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, list(values))
    merged: List[float] = []
    for part in gathered:
        if part:
            merged.extend(part)
    return merged


def summarize_values(values: List[float]) -> MetricStats:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    valid = np.isfinite(arr)
    if not np.any(valid):
        return float("nan"), float("nan"), 0
    return float(arr[valid].mean()), float(arr[valid].std()), int(valid.sum())


def build_test_metric_summary(
    metric_values: MetricValues,
    *,
    world_size: int,
) -> MetricSummary:
    summary: MetricSummary = {}
    for name, values in metric_values.items():
        model_values = gather_values_across_ranks(values["model"], world_size=world_size)
        random_values = gather_values_across_ranks(values["random"], world_size=world_size)
        summary[name] = {
            "model": summarize_values(model_values),
            "random": summarize_values(random_values),
        }
    return summary


def emit_test_metric_logs(module, summary: MetricSummary) -> None:
    for name, stats in summary.items():
        model_mean, model_std, model_n = stats["model"]
        random_mean, random_std, random_n = stats["random"]

        module.log(f"test/{name}_model_mean", model_mean, prog_bar=True)
        module.log(f"test/{name}_model_std", model_std, prog_bar=False)
        module.log(f"test/{name}_random_mean", random_mean, prog_bar=True)
        module.log(f"test/{name}_random_std", random_std, prog_bar=False)

        logger.info(
            "Test {} | Model: {:.6f} +/- {:.6f} (N={}) | Random: {:.6f} +/- {:.6f} (N={})",
            name,
            model_mean,
            model_std,
            model_n,
            random_mean,
            random_std,
            random_n,
        )


def save_test_metrics_table(log_dir: str, summary: MetricSummary) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("Matplotlib not available; skip metrics table image: {}", exc)
        return

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
    table = ax.table(cellText=table_data, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)

    save_dir = Path(log_dir) / "test_metrics"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "test_metrics_summary.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved test metrics table to {}", save_path)
