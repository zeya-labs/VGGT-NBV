"""Infrastructure adapter for mesh loading utilities."""

from __future__ import annotations

from typing import List, Optional

import torch

from nbv_framework.infrastructure.utils.mesh_utils import load_meshes_as_batch


class PyTorch3DMeshRepositoryAdapter:
    def load_meshes_as_batch(
        self,
        *,
        mesh_paths: Optional[List[Optional[str]]],
        normalize_methods: Optional[List[Optional[str]]],
        device: torch.device,
        num_workers: int,
    ):
        return load_meshes_as_batch(
            mesh_paths=mesh_paths,
            normalize_methods=normalize_methods,
            device=device,
            num_workers=num_workers,
        )
