"""Reconstruction contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class ReconstructionResult:
    recon_data: Dict[str, torch.Tensor]
    combined_images: torch.Tensor
    combined_camera_poses: torch.Tensor
    depth_z: Optional[torch.Tensor]
