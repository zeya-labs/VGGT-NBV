"""Renderer port interface."""

from __future__ import annotations

from typing import Dict, Protocol

import torch


class RendererPort(Protocol):
    def render_views(
        self,
        *,
        mesh_batch,
        camera_poses: torch.Tensor,
        out_rgb: bool,
        out_points: bool,
        out_mask: bool,
        out_depth: bool,
    ) -> Dict[str, torch.Tensor]:
        """Render multiple known views."""

    def render_candidate(
        self,
        *,
        mesh_batch,
        pose: torch.Tensor,
        out_rgb: bool,
        out_points: bool,
        out_mask: bool,
        out_depth: bool,
    ) -> Dict[str, torch.Tensor]:
        """Render one candidate pose per sample."""
