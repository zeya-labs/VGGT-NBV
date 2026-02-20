"""NBV Lightning entrypoint driven by Hydra."""

from __future__ import annotations

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from nbv_framework.utils.logging_utils import setup_logging
import os

import torch
# torch.autograd.set_detect_anomaly(True) # 当检查每一步计算的梯度是否出现异常（NaN 或 Inf）

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
# os.environ["TMPDIR"] = "/mnt/sdb/chenmohan/tmp" # 可以缓解共享内存不足
# os.makedirs(os.environ["TMPDIR"], exist_ok=True)

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
    checkpoint_weights_only = bool(getattr(cfg, "checkpoint_weights_only", False))

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=0)
    profiler = PyTorchProfiler(
        dirpath="perf_logs",   # 保存路径
        filename="training_trace",  # 文件名前缀
        export_to_chrome=True,
        schedule=schedule,
        profile_memory=True,   # 看显存是不是瓶颈
        record_shapes=True,    # 看 Tensor 形状
        with_stack=True        # 能定位到具体代码行
    )
    trainer = build_trainer(cfg)
    # trainer = build_trainer(cfg, profiler=profiler)

    resume_ckpt = to_absolute_path(cfg.resume_checkpoint) if cfg.resume_checkpoint else None

    mode = str(cfg.mode).lower().strip()
    if mode in {"train"}:
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=resume_ckpt or None,
            weights_only=checkpoint_weights_only,
        )
        return

    if mode in {"test"}:
        if not resume_ckpt:
            raise ValueError("mode=test requires resume_checkpoint pointing to a .ckpt file.")
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=resume_ckpt,
            weights_only=checkpoint_weights_only,
        )
        return

    if mode in {"train_test"}:
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=resume_ckpt or None,
            weights_only=checkpoint_weights_only,
        )
        ckpt_path = resume_ckpt or "last"
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            weights_only=checkpoint_weights_only,
        )
        return

    raise ValueError(
        "Unknown mode="
        f"{cfg.mode!r}. Supported: train, test, train_test"
    )

if __name__ == "__main__":
    main()
