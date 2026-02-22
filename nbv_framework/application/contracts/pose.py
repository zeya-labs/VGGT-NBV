"""Pose and policy inference contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class SceneFeatureBatch:
    features: torch.Tensor
    views: List[Dict[str, Any]]


@dataclass
class PolicyInferenceResult:
    next_camera_pose: torch.Tensor
    predicted_relative_position: torch.Tensor
