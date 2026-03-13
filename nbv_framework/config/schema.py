"""Structured configuration schema for NBV framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WandbConfig:
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
class ExperimentConfig:
    name: str = "dataset-house3k"
    mode: str = "train"  # train | test | train_test
    seed: int = 42
    resume_checkpoint: Optional[str] = None
    checkpoint_weights_only: bool = False
    output_dir: str = "${hydra:runtime.cwd}/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"


@dataclass
class ModelConfig:
    scene_encoder_type: str = "mapanything"
    scene_feature_dim: int = 768
    policy_hidden_dim: int = 512
    policy_num_heads: int = 4
    policy_num_layers: int = 3
    policy_output_mode: str = "position_only"
    candidate_reconstruction_mode: str = "scene_encoder"
    candidate_depth_z_detach: bool = False
    mapanything_model_name: str = "facebook/map-anything"
    mapanything_revision: Optional[str] = "6f3a25bfbb8fcc799176bb01e9d07dfb49d5416a"
    mapanything_local_files_only: bool = True
    depthanything3_model_name_or_path: str = "depth-anything/DA3-BASE"
    depthanything3_revision: Optional[str] = None
    depthanything3_local_files_only: bool = False
    depthanything3_feature_layer: Optional[int] = None
    depthanything3_use_ray_pose: bool = False
    depthanything3_ref_view_strategy: str = "saddle_balanced"
    image_size: int = 518
    up_axis: str = "Y"
    pose_outer_radius: float = 2.0
    pose_inner_radius: float = 1.3
    pose_floor_margin: float = 1.0


@dataclass
class DataConfig:
    data_root: str = "models/House3K_obj"
    batch_size: int = 16
    num_workers: int = 4
    max_meshes: int = 3000
    normalize_method: str = "unit_sphere"
    num_samples: int = 32768
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    train_repeat_factor: int = 1
    val_repeat_factor: int = 1
    test_repeat_factor: int = 1
    test_batch_size: int = 16
    test_chamfer_metrics: List[str] = field(default_factory=lambda: ["geomloss", "cd", "dcd", "emd"])
    min_initial_views: int = 2
    max_initial_views: int = 2
    randomize_initial_views: bool = False
    view_sampling_mode: str = "fully_random"
    camera_radius: float = 1.6
    camera_radius_variation: float = 0.0
    camera_radius_mode: str = "random"
    manual_camera_position: List[List[float]] = field(
        default_factory=lambda: [[-1.093546, 1.648833, -1.686863]]
    )
    manual_camera_look_at: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    use_manual_camera: bool = False


@dataclass
class OptimizationConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_epochs: int = 100


@dataclass
class RuntimeConfig:
    mesh_load_workers: int = 4
    use_epoch_seed: bool = False
    device: Optional[str] = None
    trainer: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservabilityConfig:
    save_dir: str = "${experiment.output_dir}/checkpoints"
    log_dir: str = "${experiment.output_dir}/logs"
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class NBVConfig:
    """Top-level configuration schema."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
