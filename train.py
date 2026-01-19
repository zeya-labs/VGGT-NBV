"""NBV Lightning entrypoint driven by Hydra."""

from __future__ import annotations

import hydra
from hydra.core.config_store import ConfigStore
from nbv_framework.utils.logging_utils import setup_logging
import torch
from rich.console import Console
import sys, os

from lightning.pytorch import seed_everything
from lightning.pytorch.profilers import PyTorchProfiler

from nbv_framework.training.config import NBVExperimentConfig
from nbv_framework.training.runtime import (
    build_datamodule,
    build_lightning_model,
    build_trainer,
)

cs = ConfigStore.instance()
cs.store(name="nbv_schema", node=NBVExperimentConfig)

@hydra.main(config_path="configs/nbv", config_name="train", version_base="1.3")
def main(cfg: NBVExperimentConfig) -> None:
    rank = int(os.environ.get("RANK", -1))
    setup_logging(rank=rank)
    seed_everything(cfg.seed, workers=True)
    model = build_lightning_model(cfg)
    datamodule = build_datamodule(cfg)

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=0)
    profiler = PyTorchProfiler(
        dirpath=".",           # 保存路径
        filename="perf_logs",  # 文件名前缀
        export_to_chrome=True,
        schedule=schedule,
        profile_memory=True,   # 看显存是不是瓶颈
        record_shapes=True,    # 看 Tensor 形状
        with_stack=True        # 能定位到具体代码行
    )
    trainer = build_trainer(cfg, profiler=profiler)
    
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.resume_checkpoint or None)

if __name__ == "__main__":
    main()
