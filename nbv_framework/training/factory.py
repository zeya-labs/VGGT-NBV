"""Component composition for Lightning module assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nbv_framework.adapters import (
    ChamferMetricsAdapter,
    DepthVisualizationAdapter,
    MapAnythingSceneEncoderAdapter,
    PyTorch3DMeshRepositoryAdapter,
    PyTorch3DRendererAdapter,
    ReconstructionLossAdapter,
)
from nbv_framework.infrastructure.rendering.differentiable_renderer import DifferentiableRenderer
from nbv_framework.training.lightning_module import LightningNBVModule
from nbv_framework.training.losses import ReconstructionLoss
from nbv_framework.models.policy.attention_policy_network import AttentionNBVPolicy
from nbv_framework.models.scene_encoder.mapanything_encoder import MapAnythingWrapper
from nbv_framework.workflows import (
    BatchPreparationUseCase,
    CandidateEvaluationUseCase,
    PolicyInferenceUseCase,
    TestEvaluationUseCase,
    TrainingStepUseCase,
)


@dataclass(frozen=True)
class _CoreComponents:
    mapanything: MapAnythingWrapper
    policy_network: AttentionNBVPolicy
    renderer: DifferentiableRenderer
    loss_module: ReconstructionLoss


def build_lightning_module(cfg: Any) -> LightningNBVModule:
    components = _build_core_components(cfg)

    scene_encoder = MapAnythingSceneEncoderAdapter(components.mapanything)
    renderer_adapter = PyTorch3DRendererAdapter(components.renderer)
    loss_adapter = ReconstructionLossAdapter(components.loss_module)

    chamfer = components.loss_module.chamfer_regularizer.chamfer
    metrics_adapter = ChamferMetricsAdapter(
        metrics=list(cfg.data.test_chamfer_metrics),
        max_points_per_cloud=getattr(chamfer, "max_points_per_cloud", 32768),
        use_log_warp=getattr(chamfer, "use_log_warp", False),
        point_cloud_dir_name=getattr(chamfer, "point_cloud_dir_name", "point_clouds"),
    )

    batch_preparation = BatchPreparationUseCase(
        renderer=renderer_adapter,
        mesh_repository=PyTorch3DMeshRepositoryAdapter(),
        depth_visualizer=DepthVisualizationAdapter(),
        mesh_load_workers=cfg.runtime.mesh_load_workers,
        min_initial_views=cfg.data.min_initial_views,
        max_initial_views=cfg.data.max_initial_views,
        randomize_initial_views=cfg.data.randomize_initial_views,
    )
    policy_inference = PolicyInferenceUseCase(
        scene_encoder=scene_encoder,
        policy_network=components.policy_network,
    )
    candidate_evaluation = CandidateEvaluationUseCase(
        renderer=renderer_adapter,
        loss=loss_adapter,
        scene_encoder=scene_encoder,
        reconstruction_mode=cfg.model.candidate_reconstruction_mode,
        depth_z_detach=cfg.model.candidate_depth_z_detach,
    )
    test_evaluator = TestEvaluationUseCase(
        loss=loss_adapter,
        metrics=metrics_adapter,
    )

    training_step = TrainingStepUseCase(
        batch_preparation=batch_preparation,
        policy_inference=policy_inference,
        candidate_evaluation=candidate_evaluation,
    )

    return LightningNBVModule(
        mapanything_module=components.mapanything,
        policy_network=components.policy_network,
        orchestrator=training_step,
        test_evaluator=test_evaluator,
        learning_rate=cfg.optimization.learning_rate,
        weight_decay=cfg.optimization.weight_decay,
        max_epochs=cfg.optimization.max_epochs,
        log_dir=str(cfg.observability.log_dir),
        test_chamfer_metrics=cfg.data.test_chamfer_metrics,
    )


def _build_core_components(cfg: Any) -> _CoreComponents:
    mapanything = MapAnythingWrapper(
        model_name=cfg.model.mapanything_model_name,
        revision=cfg.model.mapanything_revision,
        local_files_only=cfg.model.mapanything_local_files_only,
    )
    policy_network = AttentionNBVPolicy(
        scene_feature_dim=cfg.model.scene_feature_dim,
        hidden_dim=cfg.model.policy_hidden_dim,
        num_heads=cfg.model.policy_num_heads,
        num_layers=cfg.model.policy_num_layers,
        output_mode=cfg.model.policy_output_mode,
    )
    renderer = DifferentiableRenderer(image_size=cfg.model.image_size)
    loss_module = ReconstructionLoss(
        renderer=renderer,
        pose_up_axis=cfg.model.up_axis,
        pose_outer_radius=cfg.model.pose_outer_radius,
        pose_inner_radius=cfg.model.pose_inner_radius,
        pose_floor_margin=cfg.model.pose_floor_margin,
    )
    return _CoreComponents(
        mapanything=mapanything,
        policy_network=policy_network,
        renderer=renderer,
        loss_module=loss_module,
    )
