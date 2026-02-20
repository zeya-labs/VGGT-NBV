"""Evaluation utilities for NBV policy inference."""

from __future__ import annotations

from loguru import logger
import time
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from ..rendering import DifferentiableRenderer
from ..training.loss import ChamferDistance
from ..utils.camera_utils import position_to_pose_tensor
from ..utils.render_utils import render_mesh_views

if TYPE_CHECKING:
    from ..models import AttentionNBVPolicy, MapAnythingWrapper




def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    return value


def _ensure_batched_images(images: torch.Tensor) -> torch.Tensor:
    if images.dim() == 4:
        return images.unsqueeze(0)
    if images.dim() == 5:
        return images
    raise ValueError(f"Expected images with shape [S, 3, H, W] or [B, S, 3, H, W], got {tuple(images.shape)}")


def _ensure_batched_camera_poses(camera_poses: torch.Tensor) -> torch.Tensor:
    if camera_poses.dim() == 2:
        if camera_poses.shape[-1] != 7:
            raise ValueError(f"Expected camera pose width 7, got {camera_poses.shape[-1]}")
        return camera_poses.unsqueeze(0)
    if camera_poses.dim() == 3 and camera_poses.shape[-1] == 7:
        return camera_poses
    raise ValueError(
        f"Expected camera poses with shape [S, 7] or [B, S, 7], got {tuple(camera_poses.shape)}"
    )


def _extract_scene_features(
    vggt_wrapper: "MapAnythingWrapper",
    images: torch.Tensor,
    camera_poses: torch.Tensor,
) -> torch.Tensor:
    result = vggt_wrapper.extract_scene_features(images, camera_poses)
    if isinstance(result, tuple):
        return result[0]
    return result


def _predict_next_pose(
    policy_network: "AttentionNBVPolicy",
    scene_features: torch.Tensor,
    camera_poses: torch.Tensor,
) -> torch.Tensor:
    prediction = policy_network(scene_features, camera_poses)

    if isinstance(prediction, list):
        if not prediction:
            raise ValueError("Policy returned an empty prediction list")
        prediction = prediction[-1]

    if prediction.dim() == 3:
        prediction = prediction[:, -1, :]

    if prediction.dim() != 2:
        raise ValueError(f"Expected policy output shape [B, D], got {tuple(prediction.shape)}")

    if prediction.shape[-1] >= 7:
        return prediction[:, :7]
    if prediction.shape[-1] == 3:
        return position_to_pose_tensor(prediction)

    raise ValueError(f"Unsupported policy output width {prediction.shape[-1]} (expected 3 or >=7)")


def _sample_random_pose(batch_size: int, device: torch.device) -> torch.Tensor:
    direction = torch.randn(batch_size, 3, device=device)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    radius = torch.empty(batch_size, 1, device=device).uniform_(1.3, 2.0)
    position = direction * radius
    return position_to_pose_tensor(position)


def _prepare_sample(
    test_sample: Dict,
    renderer: DifferentiableRenderer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Any]:
    if "inputs" not in test_sample:
        raise KeyError("Evaluation expects sample with inputs/targets/mesh namespaces")

    inputs = test_sample["inputs"]
    targets = test_sample.get("targets", {})
    mesh = test_sample.get("mesh", {})

    camera_poses = _ensure_batched_camera_poses(inputs["camera_poses"])
    camera_poses = camera_poses.to(device)

    initial_images = inputs.get("images")
    if initial_images is not None:
        initial_images = _ensure_batched_images(initial_images).to(device)

    mesh_batch = mesh.get("normalized")
    if mesh_batch is not None:
        mesh_batch = mesh_batch.to(device)

    if mesh_batch is None:
        raise RuntimeError("Evaluation sample does not contain mesh['normalized']; cannot render candidate views.")

    if initial_images is None:
        render_out = render_mesh_views(
            renderer=renderer,
            mesh_batch=mesh_batch,
            camera_poses=camera_poses,
            out_rgb=True,
        )
        initial_images = render_out["rgb"]

    gt_mesh_data = _move_to_device(targets.get("gt_mesh_data", {}), device)
    return initial_images, camera_poses, gt_mesh_data, mesh_batch


def _rollout_single_sample(
    *,
    vggt_wrapper: "MapAnythingWrapper",
    renderer: DifferentiableRenderer,
    initial_images: torch.Tensor,
    initial_camera_poses: torch.Tensor,
    mesh_batch,
    gt_mesh_data: Dict[str, torch.Tensor],
    max_views: int,
    chamfer_loss: ChamferDistance,
    policy_network: "AttentionNBVPolicy" | None,
) -> Dict[str, float]:
    current_images = initial_images
    current_camera_poses = initial_camera_poses

    initial_recon = vggt_wrapper.reconstruct_and_evaluate(current_images, current_camera_poses)
    initial_quality = _compute_reconstruction_quality(initial_recon)
    quality_progression = [initial_quality]

    start_time = time.time()
    for _ in range(max_views):
        if policy_network is None:
            next_pose = _sample_random_pose(current_camera_poses.shape[0], current_camera_poses.device)
        else:
            scene_features = _extract_scene_features(vggt_wrapper, current_images, current_camera_poses)
            next_pose = _predict_next_pose(policy_network, scene_features, current_camera_poses)

        render_out = render_mesh_views(
            renderer=renderer,
            mesh_batch=mesh_batch,
            camera_poses=next_pose,
            out_rgb=True,
        )
        new_images = render_out["rgb"]
        if new_images.dim() != 5 or new_images.shape[1] != 1:
            raise ValueError(f"Expected rendered rgb shape [B, 1, 3, H, W], got {tuple(new_images.shape)}")

        current_images = torch.cat([current_images, new_images], dim=1)
        current_camera_poses = torch.cat([current_camera_poses, next_pose.unsqueeze(1)], dim=1)

        recon = vggt_wrapper.reconstruct_and_evaluate(current_images, current_camera_poses)
        new_quality = _compute_reconstruction_quality(recon)
        quality_progression.append(new_quality)

        if len(quality_progression) > 1:
            improvement = quality_progression[-1] - quality_progression[-2]
            if improvement < 0.001:
                break

    elapsed = time.time() - start_time
    views_used = len(quality_progression) - 1
    final_quality = quality_progression[-1]
    total_improvement = final_quality - initial_quality
    efficiency = total_improvement / views_used if views_used > 0 else 0.0
    inference_time = elapsed / views_used if views_used > 0 else 0.0

    final_recon = vggt_wrapper.reconstruct_and_evaluate(current_images, current_camera_poses)
    chamfer_dist = _compute_chamfer_distance(final_recon, gt_mesh_data, chamfer_loss)

    return {
        "reconstruction_quality": float(final_quality),
        "coverage_improvement": float(total_improvement),
        "view_efficiency": float(efficiency),
        "inference_time": float(inference_time),
        "chamfer_distances": float(chamfer_dist),
    }


def evaluate_nbv_policy(
    policy_network: "AttentionNBVPolicy",
    vggt_wrapper: "MapAnythingWrapper",
    renderer: DifferentiableRenderer,
    test_data: List[Dict],
    max_views: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate the learned NBV policy on a list of test samples."""
    target_device = torch.device(device)
    policy_network.eval()

    results = {
        "reconstruction_quality": [],
        "coverage_improvement": [],
        "view_efficiency": [],
        "inference_time": [],
        "chamfer_distances": [],
    }

    chamfer_loss = ChamferDistance()

    with torch.no_grad():
        for test_sample in tqdm(test_data, desc="Evaluating NBV Policy"):
            initial_images, initial_camera_poses, gt_mesh_data, mesh_batch = _prepare_sample(
                test_sample,
                renderer,
                target_device,
            )
            sample_results = _rollout_single_sample(
                vggt_wrapper=vggt_wrapper,
                renderer=renderer,
                initial_images=initial_images,
                initial_camera_poses=initial_camera_poses,
                mesh_batch=mesh_batch,
                gt_mesh_data=gt_mesh_data,
                max_views=max_views,
                chamfer_loss=chamfer_loss,
                policy_network=policy_network,
            )
            for key, value in sample_results.items():
                if key in results:
                    results[key].append(value)

    return _summarize_metric_dict(results)


def compare_with_baselines(
    policy_network: "AttentionNBVPolicy",
    vggt_wrapper: "MapAnythingWrapper",
    renderer: DifferentiableRenderer,
    test_data: List[Dict],
    device: str = "cuda",
    max_views: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Compare learned policy with a random-view baseline."""
    comparison = {
        "learned_policy": evaluate_nbv_policy(
            policy_network,
            vggt_wrapper,
            renderer,
            test_data,
            max_views=max_views,
            device=device,
        )
    }

    random_results = _evaluate_random_baseline(
        vggt_wrapper=vggt_wrapper,
        renderer=renderer,
        test_data=test_data,
        max_views=max_views,
        device=device,
    )
    comparison["random_sampling"] = random_results
    return comparison


def _evaluate_random_baseline(
    *,
    vggt_wrapper: "MapAnythingWrapper",
    renderer: DifferentiableRenderer,
    test_data: List[Dict],
    max_views: int,
    device: str,
) -> Dict[str, float]:
    target_device = torch.device(device)
    results = {
        "reconstruction_quality": [],
        "coverage_improvement": [],
        "view_efficiency": [],
        "inference_time": [],
        "chamfer_distances": [],
    }
    chamfer_loss = ChamferDistance()

    with torch.no_grad():
        for test_sample in tqdm(test_data, desc="Evaluating random baseline"):
            initial_images, initial_camera_poses, gt_mesh_data, mesh_batch = _prepare_sample(
                test_sample,
                renderer,
                target_device,
            )
            sample_results = _rollout_single_sample(
                vggt_wrapper=vggt_wrapper,
                renderer=renderer,
                initial_images=initial_images,
                initial_camera_poses=initial_camera_poses,
                mesh_batch=mesh_batch,
                gt_mesh_data=gt_mesh_data,
                max_views=max_views,
                chamfer_loss=chamfer_loss,
                policy_network=None,
            )
            for key, value in sample_results.items():
                if key in results:
                    results[key].append(value)

    return _summarize_metric_dict(results)


def _compute_reconstruction_quality(recon_data: Dict[str, torch.Tensor]) -> float:
    world_points_conf = recon_data.get("world_points_conf")
    if world_points_conf is None or world_points_conf.numel() == 0:
        return 0.0
    return float(world_points_conf.float().mean().item())


def _compute_chamfer_distance(
    recon_data: Dict[str, torch.Tensor],
    gt_data: Dict[str, torch.Tensor],
    chamfer_loss: ChamferDistance,
) -> float:
    world_points = recon_data.get("world_points")
    world_points_conf = recon_data.get("world_points_conf")
    gt_points = gt_data.get("gt_points")

    if world_points is None or world_points_conf is None or gt_points is None:
        return float("inf")

    if world_points.dim() != 5 or world_points_conf.dim() != 4:
        return float("inf")

    batch_size = world_points.shape[0]
    pred_points_list: List[torch.Tensor] = []
    gt_points_list: List[torch.Tensor] = []

    if torch.is_tensor(gt_points):
        if gt_points.dim() == 2:
            gt_points_by_batch = [gt_points for _ in range(batch_size)]
        elif gt_points.dim() == 3:
            gt_points_by_batch = [gt_points[i] for i in range(min(batch_size, gt_points.shape[0]))]
        else:
            return float("inf")
    elif isinstance(gt_points, list):
        gt_points_by_batch = [p for p in gt_points if torch.is_tensor(p)]
    else:
        return float("inf")

    valid_batch = min(batch_size, len(gt_points_by_batch))
    for batch_idx in range(valid_batch):
        conf = world_points_conf[batch_idx]
        points = world_points[batch_idx]
        mask = conf > 0.5
        pred_points = points[mask]
        if pred_points.numel() == 0:
            continue

        pred_points_list.append(pred_points.view(-1, 3))
        gt_points_list.append(gt_points_by_batch[batch_idx].view(-1, 3))

    if not pred_points_list:
        return float("inf")

    try:
        distance = chamfer_loss(pred_points_list, gt_points_list)
        return float(distance.detach().item())
    except RuntimeError as exc:
        logger.warning("Failed to compute Chamfer distance during evaluation: {}", exc)
        return float("inf")


def _summarize_metric_dict(results: Dict[str, List[float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for key, values in results.items():
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(array)
        if not np.any(valid):
            continue
        summary[f"avg_{key}"] = float(array[valid].mean())
        summary[f"std_{key}"] = float(array[valid].std())
    return summary


def generate_evaluation_report(comparison_results: Dict[str, Dict[str, float]], save_path: str) -> None:
    """Write a text summary report for comparison results."""
    with open(save_path, "w", encoding="utf-8") as report_file:
        report_file.write("NBV Policy Evaluation Report\n")
        report_file.write("=" * 50 + "\n\n")

        report_file.write("Method Comparison:\n")
        report_file.write("-" * 30 + "\n")
        report_file.write(f"{'Method':<20} {'Quality':<12} {'Efficiency':<12} {'Chamfer':<12}\n")
        report_file.write("-" * 56 + "\n")

        for method_name, results in comparison_results.items():
            quality = results.get("avg_reconstruction_quality", float("nan"))
            efficiency = results.get("avg_view_efficiency", float("nan"))
            chamfer = results.get("avg_chamfer_distances", float("nan"))
            report_file.write(
                f"{method_name:<20} {quality:<12.4f} {efficiency:<12.4f} {chamfer:<12.4f}\n"
            )

        report_file.write("\nDetailed Results:\n")
        report_file.write("-" * 20 + "\n")
        for method_name, results in comparison_results.items():
            report_file.write(f"\n{method_name.upper()}:\n")
            for metric, value in results.items():
                report_file.write(f"  {metric}: {value:.6f}\n")

    logger.info("Evaluation report saved to {}", save_path)


__all__ = [
    "evaluate_nbv_policy",
    "compare_with_baselines",
    "generate_evaluation_report",
]
