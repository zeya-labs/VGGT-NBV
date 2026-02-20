"""Structured configuration objects for NBV experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""

    enabled: bool = True
    project: str = "nbv-framework"
    entity: Optional[str] = None
    name: Optional[str] = None
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    mode: str = "online"  # online | offline | disabled
    log_model: bool = False


@dataclass
class NBVExperimentConfig:
    """Typed representation of the Hydra configuration."""

    # High-level job controls
    mode: str = "train"
    resume_checkpoint: Optional[str] = None
    # PyTorch 2.6 changed torch.load default to weights_only=True, which can
    # break loading older Lightning checkpoints that store OmegaConf objects.
    checkpoint_weights_only: bool = False
    auto_resume: bool = True
    seed: int = 42

    # Model / policy hyperparameters
    scene_feature_dim: int = 768
    policy_hidden_dim: int = 512
    policy_num_heads: int = 4
    policy_num_layers: int = 3
    policy_output_mode: str = "position_only"

    # Optimizer & trainer controls
    learning_rate: float = 1e-5
    batch_size: int = 1
    num_workers: int = 4
    mesh_load_workers: int = 4
    max_epochs: int = 1000
    normalize_method: str = "unit_sphere"
    num_samples: int = 32768
    weight_decay: float = 0
    use_epoch_seed: bool = False

    # Dataset / camera knobs
    min_initial_views: int = 2
    max_initial_views: int = 2
    randomize_initial_views: bool = True
    image_size: int = 518
    up_axis: str = "Y"
    max_meshes: int = 20
    camera_radius: float = 1.6
    camera_radius_variation: float = 0.0
    camera_radius_mode: str = "random"
    pose_outer_radius: float = 2.0
    pose_inner_radius: float = 1.3
    pose_floor_margin: float = 1.0
    train_repeat_factor: int = 1
    val_repeat_factor: int = 1
    test_repeat_factor: int = 1
    test_batch_size: int = 16
    test_chamfer_metrics: List[str] = field(default_factory=lambda: ["geomloss", "cd", "dcd", "emd"])
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    # View sampling modes:
    # - "fixed": 每次都采样同一组视角
    # - "deterministic_per_call": 单次调用随机但可重现；不同进程/worker/batch 会打散，跨 epoch 保持一致
    # - "fully_random": 每次调用完全随机，不可重现
    view_sampling_mode: str = "deterministic_per_call"
    manual_camera_position: List[List[float]] = field(
        default_factory=lambda: [[-1.093546, 1.648833, -1.686863]]
    )
    manual_camera_look_at: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    use_manual_camera: bool = False

    # Storage locations / bookkeeping
    experiment_name: str = "dataset-house3k"
    timestamp: str = ""
    data_root: str = "models/House3K_obj"

    output_dir: str = "./outputs"
    save_dir: str = "${output_dir}/checkpoints"
    log_dir: str = "${output_dir}/logs"
    wandb: WandbConfig = field(default_factory=WandbConfig)

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
