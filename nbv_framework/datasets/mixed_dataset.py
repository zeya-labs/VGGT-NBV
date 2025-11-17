"""Utilities for combining multiple datasets into a single deterministic view."""

from bisect import bisect_right
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .base_dataset import BaseDataset
from .dataset_factory import DatasetFactory
from nbv_framework.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


class MixedDataset(Dataset):
    """Deterministically concatenates several datasets into a single dataset.

    The dataset returned by this class exposes every sample from all configured
    child datasets. Samples are ordered by dataset order followed by the
    dataset-local index. No shuffling, weighting, or probabilistic sampling is
    performed—``__getitem__`` derives the target dataset and index directly from
    the requested global index.
    """

    def __init__(self, dataset_configs: List[Dict[str, Any]], seed: Optional[int] = None) -> None:
        if not dataset_configs:
            raise ValueError("dataset_configs must contain at least one dataset configuration")

        # ``seed`` is propagated to child datasets to keep their sampling deterministic.
        self.dataset_configs = dataset_configs
        self.seed = seed

        self.datasets: List[BaseDataset] = []
        self.dataset_names: List[str] = []
        self.dataset_lengths: List[int] = []
        self.cumulative_lengths: List[int] = []

        total = 0
        for index, config in enumerate(dataset_configs):
            config_with_seed = config.copy()
            if "seed" not in config_with_seed and self.seed is not None:
                config_with_seed["seed"] = self.seed
            dataset = DatasetFactory.create_from_config(config_with_seed)

            dataset_name = config.get("name", f"dataset_{index}")
            dataset_length = len(dataset)

            self.datasets.append(dataset)
            self.dataset_names.append(dataset_name)
            self.dataset_lengths.append(dataset_length)

            total += dataset_length
            self.cumulative_lengths.append(total)

        if total == 0:
            raise ValueError("All configured datasets are empty; MixedDataset has no samples to expose")

        self.total_length = total
        self._epoch: int = 0

        LOGGER.info("混合数据集创建成功 (deterministic mode):")
        for name, length in zip(self.dataset_names, self.dataset_lengths):
            LOGGER.info("  - %s: %d 样本", name, length)
        LOGGER.info("总计: %d 样本", self.total_length)

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_idx, sample_idx = self._resolve_indices(idx)
        sample = self.datasets[dataset_idx][sample_idx]

        # Annotate provenance under the meta namespace，保持批次结构一致
        if not isinstance(sample, dict):
            raise TypeError("MixedDataset expects child datasets to return dict samples")

        meta = sample.setdefault("meta", {})
        meta["source_dataset"] = self.dataset_names[dataset_idx]
        meta["source_dataset_idx"] = dataset_idx
        meta["source_dataset_sample_idx"] = sample_idx
        return sample

    def _resolve_indices(self, idx: int) -> Tuple[int, int]:
        """Translate a global index into ``(dataset_idx, local_sample_idx)``."""

        if self.total_length == 0:
            raise IndexError("MixedDataset is empty; no indices are valid")

        adjusted_idx = idx + self.total_length if idx < 0 else idx
        if adjusted_idx < 0 or adjusted_idx >= self.total_length:
            raise IndexError(f"Index {idx} is out of range for MixedDataset of length {self.total_length}")

        dataset_idx = bisect_right(self.cumulative_lengths, adjusted_idx)
        dataset_start = 0 if dataset_idx == 0 else self.cumulative_lengths[dataset_idx - 1]
        sample_idx = adjusted_idx - dataset_start
        return dataset_idx, sample_idx

    def get_dataset_info(self) -> Dict[str, Any]:
        """Return descriptive metadata about the mixed dataset."""

        return {
            "mixed_dataset": True,
            "total_samples": self.total_length,
            "num_datasets": len(self.datasets),
            "datasets": [
                {
                    "name": name,
                    "type": dataset.__class__.__name__,
                    "samples": length,
                    "info": dataset.dataset_info if hasattr(dataset, "dataset_info") else {},
                }
                for name, dataset, length in zip(self.dataset_names, self.datasets, self.dataset_lengths)
            ],
        }

    def get_dataset_by_name(self, name: str) -> BaseDataset:
        for dataset_name, dataset in zip(self.dataset_names, self.datasets):
            if dataset_name == name:
                return dataset
        raise ValueError(f"未找到名称为 '{name}' 的数据集")

    def set_epoch(self, epoch: int) -> None:
        """将当前epoch传递给所有子数据集以获得一致的采样。"""
        self._epoch = int(epoch)
        for dataset in self.datasets:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
