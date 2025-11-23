"""Structured configuration objects for NBV experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


@dataclass
class NBVExperimentConfig:
    """Typed representation of the Hydra configuration."""

    # High-level job controls
    mode: str = "train"
    create_data: bool = False
    resume_checkpoint: Optional[str] = None
    auto_resume: bool = True
    seed: int = 42
    dist_backend: str = "nccl"

    # Distributed runtime metadata
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    is_main_process: bool = True

    # Model / policy hyperparameters
    scene_feature_dim: int = 768
    policy_hidden_dim: int = 512
    policy_num_heads: int = 4
    policy_num_layers: int = 3
    policy_output_mode: str = "position_only"

    # Optimizer & trainer controls
    learning_rate: float = 1e-5
    batch_size: int = 1
    max_epochs: int = 1000
    normalize_method: str = "quantile"
    num_samples: int = 100000
    weight_decay: float = 1e-5
    use_epoch_seed: bool = False

    # Dataset / camera knobs
    synthetic_data_root: str = "./models/synthetic_data"
    min_initial_views: int = 2
    max_initial_views: int = 2
    randomize_initial_views: bool = True
    image_size: int = 518
    up_axis: str = "Y"
    max_meshes: int = 16
    train_repeat_factor: int = 1
    val_repeat_factor: int = 1
    # View sampling modes:
    # - "fixed": 每次都采样同一组视角（可通过 view_sampling_seed 控制全局随机性）
    # - "deterministic_per_call": 单次调用随机但可重现；不同进程/worker/batch 会打散，跨 epoch 保持一致
    # - "fully_random": 每次调用完全随机，不可重现
    view_sampling_mode: str = "deterministic_per_call"
    view_sampling_seed: Optional[int] = None
    manual_camera_position: List[List[float]] = field(
        default_factory=lambda: [[-1.093546, 1.648833, -1.686863]]
    )
    manual_camera_look_at: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    use_manual_camera: bool = False

    # Storage locations / bookkeeping
    experiment_name: str = "dataset-house3k"
    timestamp: str = ""

    output_dir: str = "./outputs"
    save_dir: str = "${output_dir}/checkpoints"
    log_dir: str = "${output_dir}/logs"

    # Device / dtype get resolved at runtime
    device: Optional[str] = None
    tensor_dtype: str = "float32"

    # Lightning Trainer overrides (Hydra mapping)
    trainer: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NBVExperimentConfig":
        """Create an instance from a (possibly broader) dictionary."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
