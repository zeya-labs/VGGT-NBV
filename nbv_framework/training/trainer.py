"""Trainer bootstrap helpers."""

from __future__ import annotations

import os
from typing import Optional

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers.profiler import Profiler
from loguru import logger
from omegaconf import OmegaConf

from nbv_framework.config import NBVConfig


def build_trainer(cfg: NBVConfig, profiler: Optional[Profiler] = None) -> Trainer:
    trainer_conf = OmegaConf.to_container(cfg.runtime.trainer, resolve=True)
    limit_val_batches = trainer_conf.get("limit_val_batches", 1.0)
    val_enabled = float(limit_val_batches) != 0.0

    if val_enabled:
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(cfg.observability.save_dir),
            filename="nbv-{epoch:04d}-{val/total_loss:.4f}",
            save_top_k=1,
            monitor="val/total_loss",
            mode="min",
            save_last=True,
        )
    else:
        logger.info(
            "Validation disabled (runtime.trainer.limit_val_batches={}); checkpoint saves last only.",
            limit_val_batches,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(cfg.observability.save_dir),
            filename="nbv-{epoch:04d}",
            save_top_k=0,
            save_last=True,
        )

    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer_logger = None
    if bool(cfg.observability.wandb.enabled) and str(cfg.observability.wandb.mode).lower() != "disabled":
        wandb_mode = str(cfg.observability.wandb.mode).lower()
        if wandb_mode not in {"online", "offline"}:
            raise ValueError(f"Unsupported wandb mode: {cfg.observability.wandb.mode!r}")

        os.environ["WANDB_MODE"] = wandb_mode
        os.environ.setdefault("WANDB_DIR", os.path.abspath(str(cfg.observability.log_dir)))
        trainer_logger = WandbLogger(
            project=str(cfg.observability.wandb.project),
            name=cfg.observability.wandb.name,
            save_dir=str(cfg.observability.log_dir),
            entity=cfg.observability.wandb.entity,
            group=cfg.observability.wandb.group,
            tags=cfg.observability.wandb.tags,
            notes=cfg.observability.wandb.notes,
            log_model=bool(cfg.observability.wandb.log_model),
        )

    trainer_kwargs = {
        **trainer_conf,
        "profiler": profiler,
        "default_root_dir": str(cfg.experiment.output_dir),
        "logger": trainer_logger,
        "callbacks": callbacks,
    }
    return Trainer(**trainer_kwargs)
