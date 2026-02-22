from __future__ import annotations

import pytest

from nbv_framework.config import NBVConfig, validate_config


def test_validate_config_accepts_default() -> None:
    cfg = NBVConfig()
    validate_config(cfg)


def test_validate_config_rejects_invalid_mode() -> None:
    cfg = NBVConfig()
    cfg.experiment.mode = "invalid"
    with pytest.raises(ValueError, match="Unsupported experiment.mode"):
        validate_config(cfg)


def test_validate_config_rejects_invalid_split_sum() -> None:
    cfg = NBVConfig()
    cfg.data.train_ratio = 0.9
    cfg.data.val_ratio = 0.2
    with pytest.raises(ValueError, match="Invalid split ratios"):
        validate_config(cfg)
