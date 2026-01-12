"""NBV Lightning entrypoint driven by Hydra."""

from __future__ import annotations

import hydra
from hydra.core.config_store import ConfigStore
from nbv_framework.utils.logging_utils import get_logger
from lightning.pytorch import seed_everything
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler

from nbv_framework.training.config import NBVExperimentConfig
from nbv_framework.training.runtime import (
    build_datamodule,
    build_lightning_model,
    build_trainer,
    configure_run,
    maybe_create_synthetic_data,
)

cs = ConfigStore.instance()
cs.store(name="nbv_schema", node=NBVExperimentConfig)
LOGGER = get_logger(__name__)


@hydra.main(config_path="configs/nbv", config_name="train", version_base="1.3")
def main(cfg: NBVExperimentConfig) -> None:
    seed_everything(cfg.seed, workers=True)
    configure_run(cfg)
    # maybe_create_synthetic_data(cfg)
    profiler = AdvancedProfiler(dirpath=".", filename="profile_report")
    model = build_lightning_model(cfg)
    datamodule = build_datamodule(cfg)
    trainer = build_trainer(cfg, profiler=profiler)

    if cfg.mode not in {"train", "all"}:
        LOGGER.info("Mode %s requested; skipping trainer.fit()", cfg.mode)
        return

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.resume_checkpoint or None)


if __name__ == "__main__":
    main()
