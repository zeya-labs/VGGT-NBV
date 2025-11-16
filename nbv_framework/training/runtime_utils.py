"""Utility helpers shared across runtime orchestration code."""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import torch

from nbv_framework.training.config import NBVExperimentConfig
from nbv_framework.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def set_random_seed(seed: int) -> None:
    """Set every RNG we rely on for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    LOGGER.info("Random seed set to %d for reproducibility", seed)