"""
NBV框架入口脚本

使用 Hydra 驱动的现代化配置，并委托给 nbv_framework.training.runtime.NBVExperiment
完成模型构建、训练以及评估。
"""

from __future__ import annotations

import torch
import torch.multiprocessing as mp
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from nbv_framework.training.config import NBVExperimentConfig
from nbv_framework.training.distributed import init_distributed_mode, cleanup_distributed
from nbv_framework.training.runtime import NBVExperiment

cs = ConfigStore.instance()
cs.store(name="nbv_schema", node=NBVExperimentConfig)

@hydra.main(config_path="configs/nbv", config_name="train", version_base="1.3")
def main(cfg: NBVExperimentConfig) -> None:
    """Hydra entrypoint."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    init_distributed_mode(cfg)

    cfg.distributed = cfg.world_size > 1
    cfg.is_main_process = cfg.rank == 0
    cfg.device = (
        torch.device(f"cuda:{cfg.local_rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    runner = NBVExperiment(cfg)
    try:
        runner.launch()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
