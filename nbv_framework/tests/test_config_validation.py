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


def test_validate_config_rejects_invalid_candidate_reconstruction_mode() -> None:
    cfg = NBVConfig()
    cfg.model.candidate_reconstruction_mode = "invalid"
    with pytest.raises(ValueError, match="Unsupported model.candidate_reconstruction_mode"):
        validate_config(cfg)


def test_validate_config_accepts_depthanything3_scene_encoder() -> None:
    cfg = NBVConfig()
    cfg.model.scene_encoder_type = "depthanything3"
    validate_config(cfg)


def test_validate_config_rejects_invalid_scene_encoder_type() -> None:
    cfg = NBVConfig()
    cfg.model.scene_encoder_type = "invalid"
    with pytest.raises(ValueError, match="Unsupported model.scene_encoder_type"):
        validate_config(cfg)
