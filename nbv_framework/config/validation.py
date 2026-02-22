"""Cross-field validation helpers for NBV configs."""

from __future__ import annotations

from typing import Any


def _get(cfg: Any, *keys: str, default: Any = None) -> Any:
    cur = cfg
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def validate_config(cfg: Any) -> None:
    mode = str(_get(cfg, "experiment", "mode", default="train")).lower().strip()
    if mode not in {"train", "test", "train_test"}:
        raise ValueError(f"Unsupported experiment.mode={mode!r}. Expected train|test|train_test")

    train_ratio = float(_get(cfg, "data", "train_ratio", default=0.8))
    val_ratio = float(_get(cfg, "data", "val_ratio", default=0.1))
    if train_ratio < 0 or val_ratio < 0:
        raise ValueError("data.train_ratio and data.val_ratio must be non-negative")
    if train_ratio + val_ratio > 1.0 + 1e-6:
        raise ValueError(
            f"Invalid split ratios: train_ratio={train_ratio}, val_ratio={val_ratio}, sum>1"
        )

    min_views = int(_get(cfg, "data", "min_initial_views", default=1))
    max_views = int(_get(cfg, "data", "max_initial_views", default=1))
    if min_views < 1 or max_views < 1:
        raise ValueError("data.min_initial_views and data.max_initial_views must be >= 1")
    if min_views > max_views:
        raise ValueError(
            f"data.min_initial_views ({min_views}) must be <= data.max_initial_views ({max_views})"
        )

    wandb_mode = str(
        _get(cfg, "observability", "wandb", "mode", default="online")
    ).lower()
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError(
            f"Unsupported observability.wandb.mode={wandb_mode!r}. Expected online|offline|disabled"
        )
