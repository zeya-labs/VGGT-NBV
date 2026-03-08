"""Lightning DataModule for NBV config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import lightning.pytorch as pl
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from nbv_framework.data.data_loaders import (
    create_test_loader,
    create_train_loader,
    create_val_loader,
)
from nbv_framework.data.mixed_dataset import MixedDataset
from nbv_framework.data.repeated_dataset import RepeatedDataset


class NBVDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    @property
    def data_cfg(self):
        return self.cfg.data

    @property
    def model_cfg(self):
        return self.cfg.model

    @property
    def runtime_cfg(self):
        return self.cfg.runtime

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            logger.info("Loading train/val datasets...")
            self.train_dataset = self._build_train_dataset()
            self.val_dataset = self._build_val_dataset()
        if stage == "test" or stage is None:
            logger.info("Loading test dataset...")
            self.test_dataset = self._build_test_dataset()

    def train_dataloader(self) -> DataLoader:
        return create_train_loader(
            self.train_dataset,
            batch_size=int(self.data_cfg.batch_size),
            num_workers=int(self.data_cfg.num_workers),
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return create_val_loader(
            self.val_dataset,
            batch_size=int(self.data_cfg.batch_size),
            num_workers=int(self.data_cfg.num_workers),
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return create_test_loader(
            self.test_dataset,
            batch_size=int(self.data_cfg.test_batch_size),
            num_workers=int(self.data_cfg.num_workers),
        )

    def _build_train_dataset(self) -> Dataset:
        dataset = MixedDataset(
            dataset_configs=[self._house3k_config(split="train")],
            seed=int(self.cfg.experiment.seed),
        )
        repeat_factor = max(1, int(getattr(self.data_cfg, "train_repeat_factor", 1)))
        if repeat_factor > 1:
            dataset = RepeatedDataset(dataset, repeat_factor)
        return dataset

    def _build_val_dataset(self) -> Optional[Dataset]:
        if float(self._trainer_cfg_get("limit_val_batches", 1.0)) == 0.0:
            logger.info("Validation disabled via runtime.trainer.limit_val_batches=0")
            return None

        dataset = MixedDataset(
            dataset_configs=[self._house3k_config(split="val")],
            seed=int(self.cfg.experiment.seed),
        )

        repeat_factor = max(1, int(getattr(self.data_cfg, "val_repeat_factor", 1)))
        if repeat_factor > 1:
            dataset = RepeatedDataset(dataset, repeat_factor)
        return dataset

    def _build_test_dataset(self) -> Optional[Dataset]:
        dataset = MixedDataset(
            dataset_configs=[self._house3k_config(split="test")],
            seed=int(self.cfg.experiment.seed),
        )

        repeat_factor = max(1, int(getattr(self.data_cfg, "test_repeat_factor", 1)))
        if repeat_factor > 1:
            dataset = RepeatedDataset(dataset, repeat_factor)
        return dataset

    def _house3k_config(self, split: str) -> dict:
        raw_root = Path(str(self.data_cfg.data_root)).expanduser()
        if raw_root.is_absolute():
            data_root = raw_root
        else:
            repo_root = Path(__file__).resolve().parents[2]
            data_root = repo_root / raw_root

        return {
            "name": "House3KDataset",
            "type": "house3k",
            "data_root": str(data_root),
            "num_initial_views": int(self.data_cfg.max_initial_views),
            "image_size": int(self.model_cfg.image_size),
            "normalize_method": str(self.data_cfg.normalize_method),
            "num_samples": int(self.data_cfg.num_samples),
            "split": split,
            "max_meshes": int(self.data_cfg.max_meshes),
            "up_axis": str(self.model_cfg.up_axis),
            "train_ratio": float(self.data_cfg.train_ratio),
            "val_ratio": float(self.data_cfg.val_ratio),
            "camera_radius": float(self.data_cfg.camera_radius),
            "camera_radius_variation": float(self.data_cfg.camera_radius_variation),
            "camera_radius_mode": str(self.data_cfg.camera_radius_mode),
            "manual_camera_position": self.data_cfg.manual_camera_position,
            "manual_camera_look_at": self.data_cfg.manual_camera_look_at,
            "use_manual_camera": bool(self.data_cfg.use_manual_camera),
            "view_sampling_mode": str(self.data_cfg.view_sampling_mode),
        }

    def _trainer_cfg_get(self, key: str, default: Any) -> Any:
        trainer_cfg = self.runtime_cfg.trainer
        if isinstance(trainer_cfg, dict):
            return trainer_cfg.get(key, default)
        return getattr(trainer_cfg, key, default)
