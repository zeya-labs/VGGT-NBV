"""Mesh data access port interfaces."""

from __future__ import annotations

from typing import List, Optional, Protocol

import torch


class MeshRepositoryPort(Protocol):
    def load_meshes_as_batch(
        self,
        *,
        mesh_paths: Optional[List[Optional[str]]],
        normalize_methods: Optional[List[Optional[str]]],
        device: torch.device,
        num_workers: int,
    ):
        """Load mesh assets and return framework-specific mesh batch."""
