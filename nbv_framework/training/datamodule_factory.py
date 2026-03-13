"""Data module bootstrap helpers."""

from __future__ import annotations

from nbv_framework.config import NBVConfig
from nbv_framework.training.data_module import NBVDataModule


def build_datamodule(cfg: NBVConfig) -> NBVDataModule:
    return NBVDataModule(cfg)
