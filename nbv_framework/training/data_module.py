"""Lightning DataModule counterpart for NBV training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from nbv_framework.datasets.data_loaders import create_train_loader, create_val_loader, create_test_loader
from nbv_framework.datasets.mixed_dataset import MixedDataset
from nbv_framework.datasets.repeated_dataset import RepeatedDataset
from nbv_framework.training.config import NBVExperimentConfig
from loguru import logger


class NBVDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for NBV training.

    This module handles dataset setup and loading for both training and validation stages.
    It supports mixed datasets and repeated datasets for training.

    Args:
        cfg (NBVExperimentConfig): Experiment configuration containing dataset parameters.
    """

    def __init__(self, cfg: NBVExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            logger.info("正在加载训练和验证数据...")
            self.train_dataset = self._build_train_dataset()
            self.val_dataset = self._build_val_dataset()
        if stage == "test" or stage is None:
            logger.info("正在加载测试数据...")
            self.test_dataset = self._build_test_dataset()

    def train_dataloader(self) -> DataLoader:
        # sampler = DistributedSampler(
        #     self.train_dataset,
        #     shuffle=False
        # )
        return create_train_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            # sampler=sampler,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return create_val_loader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        test_batch_size = int(getattr(self.cfg, "test_batch_size", 1))
        return create_test_loader(
            self.test_dataset,
            batch_size=test_batch_size,
            num_workers=self.cfg.num_workers,
        )

    def _build_train_dataset(self) -> Dataset:
        dataset = MixedDataset(
            dataset_configs=[self._house3k_config(split="train")],
            seed=self.cfg.seed,
        )
        repeat_factor = max(1, int(self.cfg.train_repeat_factor))
        if repeat_factor > 1:
            dataset = RepeatedDataset(dataset, repeat_factor)
        return dataset

    def _build_val_dataset(self) -> Optional[Dataset]:
        if self.cfg.trainer.get("limit_val_batches", 1.0) == 0.0:
            logger.info("Validation disabled via trainer.limit_val_batches=0; skipping val dataset.")
            return None

        dataset = MixedDataset(
            dataset_configs=[self._house3k_config(split="val")],
            seed=self.cfg.seed,
        )

        repeat_factor = max(1, int(getattr(self.cfg, "val_repeat_factor", 1)))
        if repeat_factor > 1:
            dataset = RepeatedDataset(dataset, repeat_factor)
        return dataset

    def _build_test_dataset(self) -> Optional[Dataset]:
        dataset = MixedDataset(
            dataset_configs=[self._house3k_config(split="test")],
            seed=self.cfg.seed,
        )

        repeat_factor = max(1, int(getattr(self.cfg, "test_repeat_factor", 1)))
        if repeat_factor > 1:
            dataset = RepeatedDataset(dataset, repeat_factor)
        return dataset

    def _house3k_config(self, split: str) -> dict:
        raw_root = Path(str(self.cfg.data_root)).expanduser()
        if raw_root.is_absolute():
            data_root = raw_root
        else:
            repo_root = Path(__file__).resolve().parents[2]
            data_root = repo_root / raw_root

        return {
            "name": "House3KDataset",
            "type": "house3k",
            "data_root": str(data_root),
            "num_initial_views": self.cfg.max_initial_views,
            "image_size": self.cfg.image_size,
            "normalize_method": self.cfg.normalize_method,
            "num_samples": self.cfg.num_samples,
            "split": split,
            "max_meshes": self.cfg.max_meshes,
            "up_axis": self.cfg.up_axis,
            "train_ratio": getattr(self.cfg, "train_ratio", 0.8),
            "val_ratio": getattr(self.cfg, "val_ratio", 0.2),
            "camera_radius": getattr(self.cfg, "camera_radius", 2.6),
            "camera_radius_variation": getattr(self.cfg, "camera_radius_variation", 0.0),
            "camera_radius_mode": getattr(self.cfg, "camera_radius_mode", "random"),
            "manual_camera_position": self.cfg.manual_camera_position,
            "manual_camera_look_at": self.cfg.manual_camera_look_at,
            "use_manual_camera": self.cfg.use_manual_camera,
            "view_sampling_mode": getattr(self.cfg, "view_sampling_mode", "deterministic_per_call"),
        }
