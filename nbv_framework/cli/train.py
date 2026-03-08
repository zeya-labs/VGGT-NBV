"""Hydra-driven train/test entrypoint for NBV."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from nbv_framework.config import NBVConfig, validate_config
from nbv_framework.infrastructure.utils.logging_utils import setup_logging

cs = ConfigStore.instance()
cs.store(name="nbv_schema", node=NBVConfig)
_CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs/nbv")


@hydra.main(config_path=_CONFIG_PATH, config_name="train", version_base="1.3")
def main(cfg: NBVConfig) -> None:
    from nbv_framework.training import (
        build_datamodule,
        build_lightning_module,
        build_trainer,
    )

    rank = int(os.environ.get("RANK", -1))
    setup_logging(rank=rank)

    validate_config(cfg)
    seed_everything(int(cfg.experiment.seed), workers=True)

    module = build_lightning_module(cfg)
    datamodule = build_datamodule(cfg)
    trainer = build_trainer(cfg)

    checkpoint_weights_only = bool(cfg.experiment.checkpoint_weights_only)
    resume_ckpt = (
        to_absolute_path(cfg.experiment.resume_checkpoint)
        if cfg.experiment.resume_checkpoint
        else None
    )

    mode = str(cfg.experiment.mode).lower().strip()

    if rank in {-1, 0}:
        print("Resolved config:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    if mode == "train":
        trainer.fit(
            model=module,
            datamodule=datamodule,
            ckpt_path=resume_ckpt or None,
            weights_only=checkpoint_weights_only,
        )
        return

    if mode == "test":
        if not resume_ckpt:
            raise ValueError("experiment.mode=test requires experiment.resume_checkpoint")
        trainer.test(
            model=module,
            datamodule=datamodule,
            ckpt_path=resume_ckpt,
            weights_only=checkpoint_weights_only,
        )
        return

    if mode == "train_test":
        trainer.fit(
            model=module,
            datamodule=datamodule,
            ckpt_path=resume_ckpt or None,
            weights_only=checkpoint_weights_only,
        )
        trainer.test(
            model=module,
            datamodule=datamodule,
            ckpt_path=resume_ckpt or "last",
            weights_only=checkpoint_weights_only,
        )
        return

    raise ValueError(
        f"Unknown experiment.mode={cfg.experiment.mode!r}. Supported: train, test, train_test"
    )


if __name__ == "__main__":
    main()
