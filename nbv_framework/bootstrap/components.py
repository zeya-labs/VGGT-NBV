"""Component composition for Lightning module assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nbv_framework.infrastructure.adapters import (
    ChamferMetricsAdapter,
    MapAnythingSceneEncoderAdapter,
    PyTorch3DRendererAdapter,
    ReconstructionLossAdapter,
)
from nbv_framework.infrastructure.rendering.differentiable_renderer import DifferentiableRenderer
from nbv_framework.infrastructure.training.lightning_module import LightningNBVModule
from nbv_framework.infrastructure.training.loss import ReconstructionLoss
from nbv_framework.domain.models.mapanything_wrapper import MapAnythingWrapper
from nbv_framework.domain.models.nbv_policy_networks import AttentionNBVPolicy
from nbv_framework.application.services import (
    BatchPreparationService,
    CandidateEvaluationService,
    PolicyInferenceService,
    TestEvaluationService,
    TrainingOrchestrator,
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

    batch_preparation = BatchPreparationService(
        renderer=renderer_adapter,
        mesh_load_workers=cfg.runtime.mesh_load_workers,
        min_initial_views=cfg.data.min_initial_views,
        max_initial_views=cfg.data.max_initial_views,
        randomize_initial_views=cfg.data.randomize_initial_views,
    )
    policy_inference = PolicyInferenceService(
        scene_encoder=scene_encoder,
        policy_network=components.policy_network,
    )
    candidate_evaluation = CandidateEvaluationService(
        renderer=renderer_adapter,
        loss=loss_adapter,
    )
    test_evaluator = TestEvaluationService(
        loss=loss_adapter,
        metrics=metrics_adapter,
    )

    orchestrator = TrainingOrchestrator(
        batch_preparation=batch_preparation,
        policy_inference=policy_inference,
        candidate_evaluation=candidate_evaluation,
    )

    return LightningNBVModule(
        mapanything_module=components.mapanything,
        policy_network=components.policy_network,
        orchestrator=orchestrator,
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
