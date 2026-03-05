"""In-memory cache for prepared run payloads."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

import torch


@dataclass
class PreparedRun:
    run_id: str
    mesh_path: str
    model_name: str
    created_at: str
    output_dir: Path
    image_size: int
    fov: float
    device_used: str
    initial_images: torch.Tensor
    camera_poses: torch.Tensor
    depth_z: Optional[torch.Tensor]
    depth_z_viz: Optional[torch.Tensor]
    gt_point_maps: torch.Tensor
    gt_valid_masks: torch.Tensor


class PreparedRunCache:
    def __init__(self, max_entries: int = 8) -> None:
        self.max_entries = int(max_entries)
        self._data: "OrderedDict[str, PreparedRun]" = OrderedDict()
        self._lock = Lock()

    def put(self, item: PreparedRun) -> None:
        with self._lock:
            self._data[item.run_id] = item
            self._data.move_to_end(item.run_id)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def get(self, run_id: str) -> Optional[PreparedRun]:
        with self._lock:
            found = self._data.get(run_id)
            if found is None:
                return None
            self._data.move_to_end(run_id)
            return found

    def remove(self, run_id: str) -> None:
        with self._lock:
            self._data.pop(run_id, None)

    def clear(self) -> int:
        with self._lock:
            count = len(self._data)
            self._data.clear()
            return count

    def size(self) -> int:
        with self._lock:
            return len(self._data)

    def snapshot(self) -> Dict[str, PreparedRun]:
        with self._lock:
            return dict(self._data)
