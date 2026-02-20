"""Base dataset abstractions for NBV datasets."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Common foundation for NBV datasets.

    This base class only keeps shared metadata/state and mesh-loading helpers.
    Sample-construction logic is delegated to subclasses via ``__getitem__``.
    """

    def __init__(
        self,
        data_root: str,
        num_initial_views: int = 4,
        image_size: int = 518,
        split: str = "train",
        normalize_method: str = "quantile",
        num_samples: int = 10000,
        up_axis: str = "Y",
        seed: Optional[int] = 42,
        tensor_dtype: torch.dtype = torch.float32,
        **_: Any,
    ) -> None:
        self.data_root = data_root
        self.num_initial_views = int(num_initial_views)
        self.image_size = int(image_size)
        self.split = split
        self.normalize_method = normalize_method
        self.num_samples = int(num_samples)
        self.up_axis = up_axis.upper()
        self.seed = seed
        self.tensor_dtype = tensor_dtype
        self._epoch: int = 0

        if not os.path.exists(data_root):
            raise ValueError(f"Data root does not exist: {data_root}")

        self.data_list = self._load_data_list()

        logger.info(
            "[{}] loaded {} samples for split='{}'",
            self.__class__.__name__,
            len(self.data_list),
            self.split,
        )
        logger.info(
            "[{}] normalize_method={}, num_samples={}",
            self.__class__.__name__,
            self.normalize_method,
            self.num_samples,
        )

    @abstractmethod
    def _load_data_list(self) -> List[Dict[str, Any]]:
        """Load dataset index entries."""

    @abstractmethod
    def _get_mesh_path(self, data_item: Dict[str, Any]) -> str:
        """Resolve mesh path for one data item."""

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one sample in namespaced format (inputs/targets/mesh/meta)."""

    def _load_mesh_data(
        self,
        mesh_path: str,
        normalize_method: str = "quantile",
        num_samples: int = 10000,
    ) -> Dict[str, Any]:
        """Load and normalize mesh, then sample GT points."""
        from ..utils.mesh_utils import load_and_normalize_mesh

        try:
            return load_and_normalize_mesh(
                mesh_path=mesh_path,
                normalize_method=normalize_method,
                num_samples=num_samples,
            )
        except Exception as exc:  # pragma: no cover - converted to domain error
            raise RuntimeError(
                f"Failed to load mesh: {mesh_path}. Check mesh file and normalization settings."
            ) from exc

    def set_epoch(self, epoch: int) -> None:
        """Expose current epoch for deterministic per-epoch sampling in subclasses."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.data_list)

    @property
    def dataset_info(self) -> Dict[str, Any]:
        return {
            "dataset_type": self.__class__.__name__,
            "data_root": self.data_root,
            "split": self.split,
            "num_samples": len(self.data_list),
            "num_initial_views": self.num_initial_views,
            "image_size": self.image_size,
            "normalize_method": self.normalize_method,
            "num_mesh_samples": self.num_samples,
        }
