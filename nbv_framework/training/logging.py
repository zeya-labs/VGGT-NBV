from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch
import torchvision
from lightning.pytorch.loggers import WandbLogger

from .step_types import (
    PolicyInferenceOutput,
    PoseEvaluationResult,
    PreparedBatch,
    RandomBaselineOutput,
)

logger = logging.getLogger(__name__)


def resolve_step_output_dir(trainer) -> Optional[str]:
    if not trainer.trainer.training:
        return os.path.join(
        trainer.log_dir,
        "images_val",
        f"step_{trainer.global_step:06d}",
        f"rank_{trainer.global_rank:02d}",
        )
    return os.path.join(
        trainer.log_dir,
        "images",
        f"step_{trainer.global_step:06d}",
        f"rank_{trainer.global_rank:02d}",
    )


def log_step_outputs(
    trainer,
    *,
    prepared: PreparedBatch,
    policy_inference: PolicyInferenceOutput,
    policy_eval: PoseEvaluationResult,
    random_baseline: Optional[RandomBaselineOutput],
    loss_dict: Dict[str, float],
    step_output_dir: Optional[str],
) -> None:
    step_index = trainer.global_step if trainer.trainer.training else None
    log_camera_pose_stats(
        trainer,
        policy_inference.next_camera_pose,
        policy_inference.predicted_relative_position,
        step_index,
    )
    log_view_diagnostics(
        trainer,
        new_images=policy_eval.new_images,
        new_depth_z=policy_eval.depth_z,
        initial_images=prepared.initial_images,
        initial_depth_z=prepared.depth_z,
    )
    if step_output_dir is not None:
        save_pre_images_grid(
            trainer,
            initial_images=prepared.initial_images,
            new_images=policy_eval.new_images,
            step_output_dir=step_output_dir,
        )
    log_training_metrics(trainer, loss_dict, prepared.active_view_count)
    log_random_baseline(trainer, random_baseline, step_output_dir)


def log_camera_pose_stats(
    trainer,
    next_camera_pose: torch.Tensor,
    predicted_relative_position: torch.Tensor,
    step_index: Optional[int],
) -> None:
    """Record camera pose stats for debugging."""
    if step_index is None:
        return

    positions = next_camera_pose[:, :3]
    quaternions = next_camera_pose[:, 3:]

    position_norms = torch.norm(positions, dim=1)
    trainer.log(
        "Camera_pose/position_norm_mean",
        position_norms.mean(),
        on_step=True,
        on_epoch=False,
        prog_bar=False,
        sync_dist=trainer.world_size > 1,
    )
    if position_norms.numel() > 1:
        trainer.log(
            "Camera_pose/position_norm_std",
            position_norms.std(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=trainer.world_size > 1,
        )

    if quaternions.numel() > 0:
        qw_abs = quaternions[:, 3].abs()
        trainer.log(
            "Camera_pose/quaternion_w_abs_mean",
            qw_abs.mean(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=trainer.world_size > 1,
        )

    relative_position_norms = torch.norm(predicted_relative_position, dim=1)
    trainer.log(
        "Camera_pose/relative_position_norm_mean",
        relative_position_norms.mean(),
        on_step=True,
        on_epoch=False,
        prog_bar=False,
        sync_dist=trainer.world_size > 1,
    )
    if relative_position_norms.numel() > 1:
        trainer.log(
            "Camera_pose/relative_position_norm_std",
            relative_position_norms.std(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=trainer.world_size > 1,
        )


def log_training_metrics(trainer, loss_dict: Dict[str, float], active_view_count: int) -> None:
    """Record scalar metrics for training."""
    metrics: Dict[str, float] = {
        "train/num_initial_views": float(active_view_count),
    }
    if "chamfer_loss" in loss_dict:
        metrics["train/chamfer_loss"] = loss_dict["chamfer_loss"]
    if "pose_penalty_loss" in loss_dict:
        metrics["train/pose_penalty_loss"] = loss_dict["pose_penalty_loss"]
    # for key in (
    #     "chamfer_pred_points_mean",
    #     "chamfer_pred_points_min",
    #     "chamfer_pred_points_zero_frac",
    #     "chamfer_pred_points_last_view_mean",
    #     "chamfer_pred_points_last_view_min",
    #     "chamfer_pred_points_last_view_zero_frac",
    # ):
    #     if key in loss_dict:
    #         metrics[f"train/{key}"] = loss_dict[key]
    trainer.log_dict(
        metrics,
        on_step=True,
        on_epoch=False,
        prog_bar=False,
        sync_dist=trainer.world_size > 1,
    )


def log_random_baseline(
    trainer,
    random_baseline: Optional[RandomBaselineOutput],
    step_output_dir: Optional[str],
) -> None:
    """Log random baseline diagnostics."""
    if random_baseline is None:
        return

    trainer.log(
        "train/random_baseline_chamfer_loss",
        float(random_baseline.chamfer_loss),
        on_step=True,
        on_epoch=False,
        prog_bar=False,
        sync_dist=trainer.world_size > 1,
    )
    trainer.log(
        "train/random_baseline_position_norm_mean",
        float(random_baseline.position_norm_mean),
        on_step=True,
        on_epoch=False,
        prog_bar=False,
        sync_dist=trainer.world_size > 1,
    )
    if not trainer.trainer.is_global_zero:
        return
    if random_baseline.images is None or step_output_dir is None:
        return

    random_image_dir = os.path.join(step_output_dir, "random_baseline")
    os.makedirs(random_image_dir, exist_ok=True)
    save_path = os.path.join(random_image_dir, "random_view.png")
    random_images_cpu = random_baseline.images.detach().cpu()
    torchvision.utils.save_image(random_images_cpu, save_path)


def save_pre_images_grid(
    trainer,
    *,
    initial_images: torch.Tensor,
    new_images: torch.Tensor,
    step_output_dir: Optional[str],
) -> None:
    """Save a stitched grid of initial views + NBV view into step_output_dir/pre_images/pre_images.png."""
    if not trainer.trainer.is_global_zero:
        return
    if step_output_dir is None:
        return
    if initial_images.ndim != 5:
        logger.warning(
            "Skip pre_images grid: initial_images expected [B, N, C, H, W], got %s",
            tuple(initial_images.shape),
        )
        return
    if new_images.ndim != 4:
        logger.warning(
            "Skip pre_images grid: new_images expected [B, C, H, W], got %s",
            tuple(new_images.shape),
        )
        return

    batch_size, num_views, channels, height, width = initial_images.shape
    if new_images.shape[0] != batch_size:
        logger.warning(
            "Skip pre_images grid: batch size mismatch initial_images=%d vs new_images=%d",
            batch_size,
            new_images.shape[0],
        )
        return
    if tuple(new_images.shape[1:]) != (channels, height, width):
        logger.warning(
            "Skip pre_images grid: new_images shape %s does not match expected %s",
            tuple(new_images.shape),
            (batch_size, channels, height, width),
        )
        return

    initial_cpu = initial_images.detach().float().cpu().clamp(0.0, 1.0)
    new_cpu = new_images.detach().float().cpu().clamp(0.0, 1.0)
    images_for_grid = []
    for sample_idx in range(batch_size):
        for view_idx in range(num_views):
            images_for_grid.append(initial_cpu[sample_idx, view_idx])
        images_for_grid.append(new_cpu[sample_idx])

    grid_tensor = torch.stack(images_for_grid, dim=0)
    grid = torchvision.utils.make_grid(
        grid_tensor,
        nrow=num_views + 1,
        padding=2,
    )

    pre_images_dir = os.path.join(step_output_dir, "pre_images")
    os.makedirs(pre_images_dir, exist_ok=True)
    save_path = os.path.join(pre_images_dir, "pre_images.png")
    torchvision.utils.save_image(grid, save_path)


def log_view_diagnostics(
    trainer,
    *,
    new_images: torch.Tensor,
    new_depth_z: Optional[torch.Tensor],
    initial_images: Optional[torch.Tensor] = None,
    initial_depth_z: Optional[torch.Tensor] = None,
) -> None:
    """Record rendered view diagnostics to catch black/empty views."""
    if not trainer.trainer.training:
        return
    if new_images.numel() == 0 and (initial_images is None or initial_images.numel() == 0):
        return

    with torch.no_grad():
        metrics: Dict[str, torch.Tensor] = {}

        if new_images.numel() > 0:
            mean_intensity = new_images.mean()
            min_val = new_images.min()
            max_val = new_images.max()
            gray = new_images.mean(dim=1) if new_images.dim() == 4 else None
            black_frac = None
            if gray is not None:
                black_frac = (gray < 0.05).float().mean()

            metrics.update(
                {
                    "render/new_view_intensity_mean": mean_intensity,
                    "render/new_view_intensity_min": min_val,
                    "render/new_view_intensity_max": max_val,
                }
            )
            if black_frac is not None:
                metrics["render/new_view_black_frac"] = black_frac

            if new_depth_z is not None and torch.is_tensor(new_depth_z) and new_depth_z.numel() > 0:
                depth_nonzero = (new_depth_z.abs() > 1e-6).float()
                metrics["render/new_view_valid_frac"] = depth_nonzero.mean()

        if initial_images is not None and initial_images.numel() > 0:
            init_mean = initial_images.mean()
            init_min = initial_images.min()
            init_max = initial_images.max()
            init_gray = initial_images.mean(dim=2) if initial_images.dim() == 5 else None
            init_black_frac = None
            if init_gray is not None:
                init_black_frac = (init_gray < 0.05).float().mean()

            metrics.update(
                {
                    "render/initial_view_intensity_mean": init_mean,
                    "render/initial_view_intensity_min": init_min,
                    "render/initial_view_intensity_max": init_max,
                }
            )
            if init_black_frac is not None:
                metrics["render/initial_view_black_frac"] = init_black_frac

            if (
                initial_depth_z is not None
                and torch.is_tensor(initial_depth_z)
                and initial_depth_z.numel() > 0
            ):
                init_depth_nonzero = (initial_depth_z.abs() > 1e-6).float()
                metrics["render/initial_view_valid_frac"] = init_depth_nonzero.mean()

        if metrics:
            trainer.log_dict(
                metrics,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=trainer.world_size > 1,
            )


def log_image(trainer, tag: str, img_tensor: torch.Tensor, step: int) -> None:
    if not trainer.trainer.is_global_zero:
        return
    if not isinstance(trainer.logger, WandbLogger):
        return
    try:
        import wandb  # type: ignore
    except ModuleNotFoundError:
        return

    image_cpu = img_tensor.detach().float().cpu()
    if image_cpu.ndim == 3 and image_cpu.shape[0] in (1, 3):
        image_cpu = image_cpu.permute(1, 2, 0).contiguous()
    elif image_cpu.ndim != 2 and image_cpu.ndim != 3:
        return

    run = trainer.logger.experiment
    current_step = getattr(run, "step", None)
    if current_step is None:
        run.log({tag: wandb.Image(image_cpu.numpy())})
        return

    safe_step = max(int(step), int(current_step))
    run.log({tag: wandb.Image(image_cpu.numpy())}, step=safe_step)
