"""
Dataset wrapper that virtually repeats a base dataset to control apparent length.

Useful when you want to run larger batch sizes on a tiny dataset without
duplicating data on disk. Samples are repeated deterministically by reusing
the base dataset's indices modulo its length.
"""

from __future__ import annotations

from typing import Any, Dict

from torch.utils.data import Dataset


class RepeatedDataset(Dataset):
    """
    Wraps an existing dataset and repeats it ``repeat_factor`` times.

    This is a lightweight way to expose a longer dataset length to a DataLoader
    while still drawing samples from the same underlying meshes/scenes.
    """

    def __init__(self, base_dataset: Dataset, repeat_factor: int = 1) -> None:
        if repeat_factor < 1:
            raise ValueError("repeat_factor must be >= 1")

        base_len = len(base_dataset)  # type: ignore[arg-type]
        if base_len == 0:
            raise ValueError("Cannot repeat an empty dataset")

        self.base_dataset = base_dataset
        self.repeat_factor = repeat_factor
        self._base_length = base_len

    def __len__(self) -> int:
        return self._base_length * self.repeat_factor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_idx = idx % self._base_length
        return self.base_dataset[base_idx]

    def set_epoch(self, epoch: int) -> None:
        """
        Forward epoch updates to the underlying dataset so any deterministic
        sampling logic (e.g., camera pose selection) continues to work.
        """

        if hasattr(self.base_dataset, "set_epoch"):
            self.base_dataset.set_epoch(epoch)  # type: ignore[attr-defined]

    @property
    def dataset(self) -> Dataset:
        """Expose the wrapped dataset for debugging/introspection."""

        return self.base_dataset
