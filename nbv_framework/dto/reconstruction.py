"""Reconstruction contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from nbv_framework.reconstruction import ReconstructionData


@dataclass
class ReconstructionResult:
    recon_data: ReconstructionData
    combined_images: torch.Tensor
    combined_camera_poses: torch.Tensor
    depth_z: Optional[torch.Tensor]
