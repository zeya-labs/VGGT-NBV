"""PyTorch Lightning orchestration helpers for NBV training."""

from __future__ import annotations

import logging
import os
from typing import Tuple

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from nbv_framework.models.mapanything_wrapper import MapAnythingWrapper
from nbv_framework.models.nbv_policy_networks import AttentionNBVPolicy
from nbv_framework.rendering.differentiable_renderer import DifferentiableRenderer
from nbv_framework.training.config import NBVExperimentConfig
from nbv_framework.training.data_module import NBVDataModule
from nbv_framework.training.loss import ReconstructionLoss
from nbv_framework.training.runtime_utils import set_random_seed
from nbv_framework.training.trainer import NBVTrainer
from nbv_framework.utils.data_utils import create_synthetic_training_data
from nbv_framework.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def build_lightning_model(cfg: NBVExperimentConfig) -> NBVTrainer:
    """Instantiate the LightningModule-backed trainer with all dependencies."""
    mapanything, policy, renderer, loss_fn = _build_components(cfg)
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
        device=str(cfg.device),
        use_epoch_seed=cfg.use_epoch_seed,
        enable_random_baseline=True,
        distributed=False,
        world_size=cfg.world_size,
        rank=cfg.rank,
    )


def build_trainer(cfg: NBVExperimentConfig) -> Trainer:
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
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        cfg.device = f"cuda:{local_rank}"
    else:
        cfg.device = "cpu"

    cfg.rank = rank
    cfg.world_size = world_size
    cfg.distributed = world_size > 1
    cfg.is_main_process = rank == 0

    LOGGER.info(
        "NBV Lightning run: mode=%s, device=%s, rank=%d/%d",
        cfg.mode,
        cfg.device,
        rank,
        world_size,
    )


def _build_components(
    cfg: NBVExperimentConfig,
) -> Tuple[MapAnythingWrapper, AttentionNBVPolicy, DifferentiableRenderer, ReconstructionLoss]:
    LOGGER.info("Setting up models on device: %s", cfg.device)
    mapanything = MapAnythingWrapper(
        model_name="facebook/map-anything",
        device=str(cfg.device),
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
        device=str(cfg.device),
        quality="high",
        downsample_factor=2,
    )
    loss_fn = ReconstructionLoss(renderer=renderer, pose_up_axis=cfg.up_axis)
    return mapanything, policy, renderer, loss_fn
