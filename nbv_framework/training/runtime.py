"""PyTorch Lightning orchestration helpers for NBV training."""

from __future__ import annotations

import logging
import os
from typing import Tuple

import torch

from omegaconf import OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.profilers import AdvancedProfiler

from nbv_framework.models.mapanything_wrapper import MapAnythingWrapper
from nbv_framework.models.nbv_policy_networks import AttentionNBVPolicy
from nbv_framework.rendering.differentiable_renderer import DifferentiableRenderer
from nbv_framework.training.config import NBVExperimentConfig
from nbv_framework.training.data_module import NBVDataModule
from nbv_framework.training.loss import ReconstructionLoss
from nbv_framework.utils.device_utils import (
    coerce_device,
    dtype_to_string,
    resolve_device,
    resolve_dtype,
)
from nbv_framework.training.runtime_utils import set_random_seed
from nbv_framework.training.trainer import NBVTrainer
from nbv_framework.utils.data_utils import create_synthetic_training_data
from nbv_framework.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def build_lightning_model(cfg: NBVExperimentConfig) -> NBVTrainer:
    """Instantiate the LightningModule-backed trainer with all dependencies."""
    runtime_device = coerce_device(cfg.device)
    runtime_dtype = resolve_dtype(cfg.tensor_dtype)
    mapanything, policy, renderer, loss_fn = _build_components(cfg, runtime_device, runtime_dtype)
    return NBVTrainer(
        vggt_wrapper=mapanything,
        policy_network=policy,
        renderer=renderer,
        loss_fn=loss_fn,
        min_initial_views=cfg.min_initial_views,
        max_initial_views=cfg.max_initial_views,
        randomize_initial_views=cfg.randomize_initial_views,
        max_epochs=cfg.max_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        log_dir=cfg.log_dir,
        device=str(runtime_device),
        tensor_dtype=runtime_dtype,
        use_epoch_seed=cfg.use_epoch_seed,
        enable_random_baseline=True,
        distributed=False,
        world_size=cfg.world_size,
        rank=cfg.rank,
    )


def build_trainer(cfg: NBVExperimentConfig, profiler: AdvancedProfiler) -> Trainer:
    """Configure the PyTorch Lightning Trainer from Hydra config."""
    trainer_conf = OmegaConf.to_container(cfg.trainer, resolve=True)  # type: ignore[arg-type]
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.save_dir,
            filename="nbv-{epoch:04d}-{val/total_loss:.4f}",
            save_top_k=1,
            monitor="val/total_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = TensorBoardLogger(save_dir=cfg.log_dir, name="events")
    trainer_kwargs = {
        **trainer_conf,
        "profiler": profiler,
        "default_root_dir": cfg.output_dir,
        "logger": logger,
        "callbacks": callbacks,
    }
    return Trainer(**trainer_kwargs)

def build_datamodule(cfg: NBVExperimentConfig) -> NBVDataModule:
    return NBVDataModule(cfg)

@rank_zero_only
def maybe_create_synthetic_data(cfg: NBVExperimentConfig) -> None:
    """Optionally create synthetic data upfront."""
    if not (cfg.create_data or not os.path.exists(cfg.synthetic_data_root)):
        return

    LOGGER.info("Creating synthetic training data at %s", cfg.synthetic_data_root)
    create_synthetic_training_data(
        output_dir=cfg.synthetic_data_root,
        num_objects=20,
        num_views_per_object=15,
        image_size=cfg.image_size,
    )
    LOGGER.info("Synthetic data creation finished")


def configure_run(cfg: NBVExperimentConfig) -> None:
    """Populate runtime attributes and log summary."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    LOGGER.info(f"RANK: {rank}, LOCAL_RANK: {local_rank}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    resolved_device = resolve_device(getattr(cfg, "device", None), local_rank)
    resolved_dtype = resolve_dtype(getattr(cfg, "tensor_dtype", None))
    cfg.device = str(resolved_device)
    cfg.tensor_dtype = dtype_to_string(resolved_dtype)

    cfg.rank = local_rank
    cfg.world_size = world_size
    cfg.distributed = world_size > 1
    cfg.is_main_process = rank == 0

    LOGGER.info(
        "NBV Lightning run: mode=%s, device=%s, dtype=%s, rank=%d/%d",
        cfg.mode,
        resolved_device,
        resolved_dtype,
        rank,
        world_size,
    )


def _build_components(
    cfg: NBVExperimentConfig,
    runtime_device: torch.device,
    runtime_dtype: torch.dtype,
) -> Tuple[MapAnythingWrapper, AttentionNBVPolicy, DifferentiableRenderer, ReconstructionLoss]:
    LOGGER.info("Setting up models on device: %s", runtime_device)
    mapanything = MapAnythingWrapper(
        model_name="facebook/map-anything",
        device=str(runtime_device),
    )
    policy = AttentionNBVPolicy(
        scene_feature_dim=cfg.scene_feature_dim,
        hidden_dim=cfg.policy_hidden_dim,
        num_heads=cfg.policy_num_heads,
        num_layers=cfg.policy_num_layers,
        output_mode=cfg.policy_output_mode,
    )
    renderer = DifferentiableRenderer(
        image_size=cfg.image_size,
        device=str(runtime_device),
        quality="high",
        downsample_factor=2,
    )
    loss_fn = ReconstructionLoss(
        renderer=renderer,
        pose_up_axis=cfg.up_axis,
        default_device=runtime_device,
        tensor_dtype=runtime_dtype,
    )
    return mapanything, policy, renderer, loss_fn
