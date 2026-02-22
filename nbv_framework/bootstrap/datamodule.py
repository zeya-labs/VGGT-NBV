"""Data module bootstrap helpers."""

from __future__ import annotations

from typing import Any

from nbv_framework.infrastructure.training.data_module import NBVDataModule


def build_datamodule(cfg: Any) -> NBVDataModule:
    return NBVDataModule(cfg)
