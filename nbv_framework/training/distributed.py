"""Helper utilities for distributed/torchrun launches."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_distributed_mode(cfg) -> None:
    """Initialize torch.distributed if launched via torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    cfg.world_size = world_size
    cfg.rank = rank
    cfg.local_rank = local_rank
    cfg.distributed = world_size > 1

    if not cfg.distributed:
        cfg.world_size = 1
        cfg.rank = 0
        cfg.local_rank = 0
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA devices.")

    torch.cuda.set_device(cfg.local_rank)
    dist.init_process_group(backend=cfg.dist_backend, init_method="env://")
    dist.barrier()


def cleanup_distributed() -> None:
    """Tear down torch.distributed state if initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
