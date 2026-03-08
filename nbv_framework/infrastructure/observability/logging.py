"""Logging and metrics helpers for Lightning modules."""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch

from nbv_framework.dto import PolicyInferenceResult, PoseEvaluationResult, PreparedBatch
from .artifacts import save_pre_images_grid


def resolve_step_output_dir(module) -> Optional[str]:
    trainer = module.trainer
    if trainer is None or not trainer.is_global_zero:
        return None

    if trainer.training:
        if (module.global_step + 1) % trainer.log_every_n_steps != 0:
            return None
        return os.path.join(
            module.log_dir,
            "images_train",
            f"step_{module.global_step:06d}",
            f"rank_{module.global_rank:02d}",
        )

    if getattr(module, "_val_images_saved", False):
        return None
    module._val_images_saved = True
    return os.path.join(
        module.log_dir,
        "images_val",
        f"step_{module.global_step:06d}",
        f"rank_{module.global_rank:02d}",
    )


def _prefix(stage: str, key: str) -> str:
    if key.startswith(f"{stage}/"):
        return key
    return f"{stage}/{key}"


def log_loss_metrics(module, *, loss_dict: Dict[str, float], stage: str) -> None:
    if not loss_dict:
        return

    metrics = {k: float(v) for k, v in loss_dict.items()}
    total_loss = metrics.pop("total_loss", None)
    if total_loss is not None:
        module.log(
            _prefix(stage, "total_loss"),
            total_loss,
            prog_bar=True,
            sync_dist=module.world_size > 1,
            batch_size=getattr(module, "_last_batch_size", None),
        )

    if metrics:
        module.log_dict(
            {_prefix(stage, k): v for k, v in metrics.items()},
            prog_bar=False,
            sync_dist=module.world_size > 1,
            batch_size=getattr(module, "_last_batch_size", None),
        )


def log_camera_pose_stats(
    module,
    *,
    next_camera_pose: torch.Tensor,
    predicted_relative_position: torch.Tensor,
    stage: str,
) -> None:
    positions = next_camera_pose[:, :3]
    quaternions = next_camera_pose[:, 3:]
    position_norms = torch.norm(positions, dim=1)

    module.log(
        _prefix(stage, "camera_pose/position_norm_mean"),
        position_norms.mean(),
        prog_bar=False,
        sync_dist=module.world_size > 1,
        batch_size=getattr(module, "_last_batch_size", None),
    )

    if position_norms.numel() > 1:
        module.log(
            _prefix(stage, "camera_pose/position_norm_std"),
            position_norms.std(),
            prog_bar=False,
            sync_dist=module.world_size > 1,
            batch_size=getattr(module, "_last_batch_size", None),
        )

    if quaternions.numel() > 0:
        module.log(
            _prefix(stage, "camera_pose/quaternion_w_abs_mean"),
            quaternions[:, 3].abs().mean(),
            prog_bar=False,
            sync_dist=module.world_size > 1,
            batch_size=getattr(module, "_last_batch_size", None),
        )

    relative_position_norms = torch.norm(predicted_relative_position, dim=1)
    module.log(
        _prefix(stage, "camera_pose/relative_position_norm_mean"),
        relative_position_norms.mean(),
        prog_bar=False,
        sync_dist=module.world_size > 1,
        batch_size=getattr(module, "_last_batch_size", None),
    )


def log_step_outputs(
    module,
    *,
    prepared: PreparedBatch,
    policy_inference: PolicyInferenceResult,
    policy_eval: PoseEvaluationResult,
    loss_dict: Dict[str, float],
    step_output_dir: Optional[str],
    stage: str,
) -> None:
    log_camera_pose_stats(
        module,
        next_camera_pose=policy_inference.next_camera_pose,
        predicted_relative_position=policy_inference.predicted_relative_position,
        stage=stage,
    )
    log_loss_metrics(module, loss_dict=loss_dict, stage=stage)

    if step_output_dir is not None:
        save_pre_images_grid(
            initial_images=prepared.initial_images,
            new_images=policy_eval.new_images,
            step_output_dir=step_output_dir,
            is_global_zero=module.trainer.is_global_zero,
        )
