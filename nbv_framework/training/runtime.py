"""PyTorch Lightning orchestration helpers for NBV training."""

from __future__ import annotations

from loguru import logger
import os
from typing import Tuple

from omegaconf import OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers.profiler import Profiler

from nbv_framework.models.mapanything_wrapper import MapAnythingWrapper
from nbv_framework.models.nbv_policy_networks import AttentionNBVPolicy
from nbv_framework.rendering.differentiable_renderer import DifferentiableRenderer
from nbv_framework.training.config import NBVExperimentConfig
from nbv_framework.training.data_module import NBVDataModule
from nbv_framework.training.loss import ReconstructionLoss
from nbv_framework.training.trainer import NBVTrainer




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
        use_epoch_seed=cfg.use_epoch_seed,
        test_chamfer_metrics=getattr(cfg, "test_chamfer_metrics", None),
        mesh_load_workers=cfg.mesh_load_workers,
    )


def build_trainer(cfg: NBVExperimentConfig, profiler: Profiler = None) -> Trainer:
    """Configure the PyTorch Lightning Trainer from Hydra config."""
    trainer_conf = OmegaConf.to_container(cfg.trainer, resolve=True)  # type: ignore[arg-type]
    limit_val_batches = trainer_conf.get("limit_val_batches", 1.0)
    val_enabled = float(limit_val_batches) != 0.0

    if val_enabled:
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.save_dir,
            filename="nbv-{epoch:04d}-{val/total_loss:.4f}",
            save_top_k=1,
            monitor="val/total_loss",
            mode="min",
            save_last=True,
        )
    else:
        logger.info(
            "Validation disabled (trainer.limit_val_batches={}); "
            "checkpoint callback will only save last.",
            limit_val_batches,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.save_dir,
            filename="nbv-{epoch:04d}",
            save_top_k=0,
            save_last=True,
        )

    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer_logger = None
    if cfg.wandb.enabled and str(cfg.wandb.mode).lower() != "disabled":
        wandb_mode = str(cfg.wandb.mode).lower()
        if wandb_mode not in {"online", "offline"}:
            raise ValueError(f"Unsupported wandb.mode={cfg.wandb.mode!r}")
        os.environ["WANDB_MODE"] = wandb_mode
        os.environ.setdefault("WANDB_DIR", os.path.abspath(cfg.log_dir))
        trainer_logger = WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            save_dir=cfg.log_dir,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes,
            log_model=cfg.wandb.log_model,
        )
    trainer_kwargs = {
        **trainer_conf,
        "profiler": profiler,
        "default_root_dir": cfg.output_dir,
        "logger": trainer_logger,
        "callbacks": callbacks,
    }
    return Trainer(**trainer_kwargs)

def build_datamodule(cfg: NBVExperimentConfig) -> NBVDataModule:
    return NBVDataModule(cfg)

def _build_components(
    cfg: NBVExperimentConfig,
) -> Tuple[MapAnythingWrapper, AttentionNBVPolicy, DifferentiableRenderer, ReconstructionLoss]:
    mapanything = MapAnythingWrapper(
        model_name=cfg.mapanything_model_name,
        revision=cfg.mapanything_revision,
        local_files_only=cfg.mapanything_local_files_only,
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
    )
    loss_fn = ReconstructionLoss(
        renderer=renderer,
        pose_up_axis=cfg.up_axis,
        pose_outer_radius=cfg.pose_outer_radius,
        pose_inner_radius=cfg.pose_inner_radius,
        pose_floor_margin=cfg.pose_floor_margin,
    )
    return mapanything, policy, renderer, loss_fn
