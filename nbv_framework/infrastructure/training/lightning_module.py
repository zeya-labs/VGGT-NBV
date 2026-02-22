"""Thin Lightning module delegating core logic to services."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
import torch.optim as optim
from lightning.pytorch import LightningModule

from nbv_framework.domain.geometry.pose_sampling import sample_random_positions
from nbv_framework.infrastructure.observability import log_step_outputs, resolve_step_output_dir
from nbv_framework.infrastructure.training.test_metrics import (
    append_test_metric_values,
    build_test_metric_summary,
    emit_test_metric_logs,
    init_test_metric_values,
    save_test_metrics_table,
)
from nbv_framework.infrastructure.utils.camera_utils import position_to_pose_tensor


class LightningNBVModule(LightningModule):
    """Lightning wrapper over the new service-oriented training pipeline."""

    def __init__(
        self,
        *,
        mapanything_module,
        policy_network,
        orchestrator,
        test_evaluator,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        log_dir: str,
        test_chamfer_metrics: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        # Register frozen scene encoder as a submodule so Lightning strategy/precision
        # plugins can move/cast it consistently with the training graph.
        self.mapanything_module = mapanything_module
        self.policy_network = policy_network
        self.orchestrator = orchestrator
        self.test_evaluator = test_evaluator

        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.log_dir = log_dir
        self.test_chamfer_metrics = list(test_chamfer_metrics) if test_chamfer_metrics else ["geomloss"]

        self.world_size = 1
        self._last_batch_size: Optional[int] = None
        self._val_images_saved = False
        self._test_metric_values: Dict[str, Dict[str, List[float]]] = {}

        self.save_hyperparameters(
            ignore=[
                "mapanything_module",
                "policy_network",
                "orchestrator",
                "test_evaluator",
            ]
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            self.world_size = self.trainer.world_size

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.policy_network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=1e-7,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_validation_epoch_start(self) -> None:
        self._val_images_saved = False

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        loss, _, _, _, _ = self._run_step(batch, stage="train")
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        loss, _, _, _, _ = self._run_step(batch, stage="val")
        return loss

    def _run_step(self, batch: Dict, *, stage: str):
        is_training = stage == "train"
        step_output_dir = resolve_step_output_dir(self)
        (
            total_loss,
            loss_dict,
            prepared,
            policy_inference,
            policy_eval,
        ) = self.orchestrator.run_step(
            batch,
            training=is_training,
            point_cloud_dir=step_output_dir,
            on_new_point_maps=self._track_new_point_maps_gradients,
        )
        self._last_batch_size = int(prepared.camera_poses.shape[0])

        self._track_policy_gradients(
            policy_inference.predicted_relative_position,
            policy_inference.next_camera_pose,
        )

        log_step_outputs(
            self,
            prepared=prepared,
            policy_inference=policy_inference,
            policy_eval=policy_eval,
            loss_dict=loss_dict,
            step_output_dir=step_output_dir,
            stage=stage,
        )
        return total_loss, loss_dict, prepared, policy_inference, policy_eval

    def _attach_gradient_metric_hooks(self, tensor, metrics) -> None:
        trainer = self.trainer
        if trainer is None or not trainer.training:
            return
        if tensor is None or not tensor.requires_grad:
            return

        def _capture(grad: torch.Tensor) -> torch.Tensor:
            if grad is None or grad.numel() == 0:
                return grad
            detached_grad = grad.detach()
            for log_key, selector in metrics:
                selected_grad = selector(detached_grad)
                if selected_grad is None or selected_grad.numel() == 0:
                    continue
                self.log(
                    log_key,
                    selected_grad.float().norm(dim=-1).mean(),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    batch_size=self._last_batch_size,
                )
            return grad

        try:
            tensor.register_hook(_capture)
        except RuntimeError:
            return

    def _track_policy_gradients(
        self,
        predicted_relative_position: torch.Tensor,
        next_camera_pose: torch.Tensor,
    ) -> None:
        self._attach_gradient_metric_hooks(
            predicted_relative_position,
            (("gradients/predicted_relative_position_grad_norm", lambda grad: grad),),
        )
        self._attach_gradient_metric_hooks(
            next_camera_pose,
            (
                ("gradients/next_pose_position_grad_norm", lambda grad: grad[:, :3]),
                ("gradients/next_pose_quaternion_grad_norm", lambda grad: grad[:, 3:]),
            ),
        )

    def _track_new_point_maps_gradients(self, value: Optional[torch.Tensor]) -> None:
        self._attach_gradient_metric_hooks(
            value,
            (("gradients/new_point_maps_render_grad_norm", lambda grad: grad),),
        )

    def on_before_optimizer_step(self, optimizer) -> None:
        grad_params = [p for p in self.policy_network.parameters() if p.grad is not None]
        if grad_params:
            grad_norm_sq = torch.zeros((), device=grad_params[0].grad.device, dtype=torch.float32)
            num_params = torch.zeros((), device=grad_params[0].grad.device, dtype=torch.float32)
            for p in grad_params:
                grad = p.grad.detach()
                num_params = num_params + float(grad.numel())
                grad_norm_sq = grad_norm_sq + grad.float().pow(2).sum()

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(grad_norm_sq, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_params, op=dist.ReduceOp.SUM)
                world_size = float(dist.get_world_size())
                grad_norm_sq = grad_norm_sq / world_size
                num_params = num_params / world_size

            grad_norm = torch.sqrt(grad_norm_sq)
            grad_rms = grad_norm / torch.sqrt(num_params.clamp_min(1.0))
        else:
            grad_norm = torch.zeros((), device=self.device, dtype=torch.float32)
            grad_rms = grad_norm

        self.log(
            "gradients/global_norm",
            grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=self._last_batch_size,
        )
        self.log(
            "gradients/grad_rms",
            grad_rms,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=self._last_batch_size,
        )

    def on_test_epoch_start(self) -> None:
        self._test_metric_values = init_test_metric_values(self.test_chamfer_metrics)

    def test_step(self, batch: Dict, batch_idx: int):
        with torch.no_grad():
            (
                _,
                _,
                prepared,
                policy_inference,
                policy_eval,
            ) = self.orchestrator.run_step(
                batch,
                training=False,
                point_cloud_dir=None,
                on_new_point_maps=None,
            )

            combined_images = torch.cat([prepared.initial_images, policy_eval.new_images.unsqueeze(1)], dim=1)
            combined_camera_poses = torch.cat(
                [prepared.camera_poses, policy_inference.next_camera_pose.unsqueeze(1)], dim=1
            )
            model_metrics = self.test_evaluator.compute_metrics(
                gt_mesh_data=policy_eval.gt_mesh_data,
                combined_images_batch=combined_images,
                combined_camera_poses=combined_camera_poses,
                depth_z=policy_eval.depth_z,
            )

            random_positions = sample_random_positions(
                batch_size=prepared.initial_images.shape[0],
                device=prepared.initial_images.device,
                loss_fn=self.orchestrator.candidate_evaluation.loss.loss_module,
            )
            random_pose = position_to_pose_tensor(random_positions)
            random_eval = self.orchestrator.candidate_evaluation.evaluate_candidate_pose(
                pose=random_pose,
                initial_images=prepared.initial_images,
                camera_poses_batch=prepared.camera_poses,
                gt_mesh_data=prepared.trimmed_gt_mesh_data,
                mesh_batch=prepared.mesh_batch,
                point_cloud_dir=None,
            )
            random_combined_images = torch.cat(
                [prepared.initial_images, random_eval.new_images.unsqueeze(1)],
                dim=1,
            )
            random_combined_camera_poses = torch.cat(
                [prepared.camera_poses, random_pose.unsqueeze(1)],
                dim=1,
            )
            random_metrics = self.test_evaluator.compute_metrics(
                gt_mesh_data=random_eval.gt_mesh_data,
                combined_images_batch=random_combined_images,
                combined_camera_poses=random_combined_camera_poses,
                depth_z=random_eval.depth_z,
            )

            append_test_metric_values(
                self._test_metric_values,
                model_metrics=model_metrics,
                random_metrics=random_metrics,
            )

        return {
            **{f"{name}_model": val for name, val in model_metrics.items()},
            **{f"{name}_random": val for name, val in random_metrics.items()},
        }

    def on_test_epoch_end(self) -> None:
        world_size = self.trainer.world_size if self.trainer is not None else 1
        summary = build_test_metric_summary(
            self._test_metric_values,
            world_size=world_size,
        )

        if getattr(self.trainer, "is_global_zero", True):
            emit_test_metric_logs(self, summary)
            save_test_metrics_table(self.log_dir, summary)
