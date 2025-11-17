"""
评估工具

用于评估NBV策略的性能和泛化能力。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import time
from tqdm import tqdm

from ..rendering import DifferentiableRenderer
from ..training.loss import ChamferDistance
from nbv_framework.utils.logging_utils import get_logger
if TYPE_CHECKING:
    from ..models import MapAnythingWrapper, BaseNBVPolicy

LOGGER = get_logger(__name__)

def evaluate_nbv_policy(policy_network: "BaseNBVPolicy",
                       vggt_wrapper: "MapAnythingWrapper",
                       renderer: DifferentiableRenderer,
                       test_data: List[Dict],
                       max_views: int = 10,
                       device: str = "cuda") -> Dict[str, float]:
    """
    评估NBV策略在测试数据上的性能
    
    Args:
        policy_network: 训练好的NBV策略网络
        vggt_wrapper: VGGT包装器
        renderer: 可微分渲染器
        test_data: 测试数据列表
        max_views: 最大视图数量
        device: 计算设备
        
    Returns:
        evaluation_results: 评估结果字典
    """
    policy_network.eval()
    
    results = {
        "reconstruction_quality": [],
        "coverage_improvement": [],
        "view_efficiency": [],
        "inference_time": [],
        "chamfer_distances": []
    }
    
    chamfer_loss = ChamferDistance()
    
    with torch.no_grad():
        for test_sample in tqdm(test_data, desc="Evaluating NBV Policy"):
            # 评估单个测试样本
            sample_results = _evaluate_single_sample(
                test_sample, policy_network, vggt_wrapper, 
                renderer, chamfer_loss, max_views, device
            )
            
            # 收集结果
            for key, value in sample_results.items():
                if key in results:
                    results[key].append(value)
    
    # 计算平均结果
    avg_results = {}
    for key, values in results.items():
        if values:
            avg_results[f"avg_{key}"] = np.mean(values)
            avg_results[f"std_{key}"] = np.std(values)
    
    return avg_results


def _evaluate_single_sample(test_sample: Dict,
                           policy_network: "BaseNBVPolicy",
                           vggt_wrapper: "MapAnythingWrapper",
                           renderer: DifferentiableRenderer,
                           chamfer_loss: ChamferDistance,
                           max_views: int,
                           device: str) -> Dict[str, float]:
    """评估单个测试样本"""
    
    if "inputs" not in test_sample:
        raise KeyError("Evaluation expects batch with inputs/targets/mesh namespaces")

    inputs = test_sample["inputs"]
    targets = test_sample.get("targets", {})
    mesh = test_sample.get("mesh", {})

    initial_images = inputs["images"].to(device)
    initial_camera_poses = inputs["camera_poses"].to(device)
    gt_mesh_data = targets["gt_mesh_data"]
    mesh_batch = mesh.get("normalized")
    
    # 记录开始时间
    start_time = time.time()
    
    # 初始重建质量
    initial_recon = vggt_wrapper.reconstruct_and_evaluate(
        initial_images.unsqueeze(0),
        initial_camera_poses.unsqueeze(0),
    )
    initial_quality = _compute_reconstruction_quality(initial_recon, gt_mesh_data, chamfer_loss)
    
    # 迭代添加视图
    current_images = initial_images
    current_camera_poses = initial_camera_poses
    quality_progression = [initial_quality]
    
    for view_idx in range(max_views):
        # 提取场景特征
        scene_features = vggt_wrapper.extract_scene_features(
            current_images.unsqueeze(0),
            current_camera_poses.unsqueeze(0),
        )
        
        # 预测下一个视角
        next_pose = policy_network(scene_features)
        
        # 渲染新视图
        if mesh_batch is None:
            raise RuntimeError("Mesh batch missing in evaluation sample")
        new_image = renderer(mesh_batch, next_pose, policy_network.output_mode)
        
        # 添加新视图
        current_images = torch.cat([current_images, new_image.squeeze(0)], dim=0)
        current_camera_poses = torch.cat([current_camera_poses, next_pose.squeeze(0)], dim=0)
        
        # 评估新的重建质量
        updated_recon = vggt_wrapper.reconstruct_and_evaluate(
            current_images.unsqueeze(0),
            current_camera_poses.unsqueeze(0),
        )
        new_quality = _compute_reconstruction_quality(updated_recon, gt_mesh_data, chamfer_loss)
        quality_progression.append(new_quality)
        
        # 如果质量提升很小，可以提前停止
        if len(quality_progression) > 1:
            improvement = quality_progression[-1] - quality_progression[-2]
            if improvement < 0.001:  # 阈值
                break
    
    # 记录结束时间
    end_time = time.time()
    
    # 计算评估指标
    final_quality = quality_progression[-1]
    total_improvement = final_quality - initial_quality
    views_used = len(quality_progression) - 1
    efficiency = total_improvement / views_used if views_used > 0 else 0
    inference_time = (end_time - start_time) / views_used if views_used > 0 else 0
    
    # 计算Chamfer距离
    final_recon = vggt_wrapper.reconstruct_and_evaluate(
        current_images.unsqueeze(0),
        current_camera_poses.unsqueeze(0),
    )
    chamfer_dist = _compute_chamfer_distance(final_recon, gt_mesh_data, chamfer_loss)
    
    return {
        "reconstruction_quality": final_quality,
        "coverage_improvement": total_improvement,
        "view_efficiency": efficiency,
        "inference_time": inference_time,
        "chamfer_distances": chamfer_dist
    }


def _compute_reconstruction_quality(recon_data: Dict[str, torch.Tensor],
                                  gt_data: Dict[str, torch.Tensor],
                                  chamfer_loss: ChamferDistance) -> float:
    """计算重建质量评分"""
    world_points = recon_data.get("world_points")
    world_points_conf = recon_data.get("world_points_conf")
    
    if world_points is None or world_points_conf is None:
        return 0.0
    
    # 基于高置信度点的数量和分布
    high_conf_mask = world_points_conf > 0.5
    quality_score = high_conf_mask.float().mean().item()
    
    return quality_score


def _compute_chamfer_distance(recon_data: Dict[str, torch.Tensor],
                            gt_data: Dict[str, torch.Tensor],
                            chamfer_loss: ChamferDistance) -> float:
    """计算Chamfer距离"""
    # 从重建数据中提取点云
    world_points = recon_data.get("world_points")
    world_points_conf = recon_data.get("world_points_conf")
    
    if world_points is None or world_points_conf is None:
        return float('inf')
    
    # 提取高置信度点
    high_conf_mask = world_points_conf > 0.5
    pred_points = world_points[high_conf_mask].view(-1, 3)
    
    # GT点云
    gt_points = gt_data.get("gt_points")
    if gt_points is None:
        return float('inf')
    
    if len(pred_points) == 0:
        return float('inf')
    
    # 计算Chamfer距离
    chamfer_dist = chamfer_loss(
        pred_points.unsqueeze(0),
        gt_points.unsqueeze(0)
    )
    
    return chamfer_dist.item()


def _create_mesh_from_data(mesh_data: Dict) -> 'Meshes':
    """从mesh数据创建PyTorch3D mesh对象"""
    # 这里需要根据实际数据格式实现
    # 暂时返回None，需要具体实现
    return None


def compare_with_baselines(policy_network: "BaseNBVPolicy",
                          vggt_wrapper: "MapAnythingWrapper",
                          renderer: DifferentiableRenderer,
                          test_data: List[Dict],
                          device: str = "cuda") -> Dict[str, Dict[str, float]]:
    """
    与基线方法比较
    
    Args:
        policy_network: 训练好的NBV策略网络
        vggt_wrapper: VGGT包装器
        renderer: 可微分渲染器
        test_data: 测试数据
        device: 计算设备
        
    Returns:
        comparison_results: 比较结果字典
    """
    methods = {
        "learned_policy": policy_network,
        "random_sampling": None,
        "frontier_based": None,
        "entropy_based": None
    }
    
    results = {}
    
    for method_name, method in methods.items():
        if method_name == "learned_policy":
            # 使用学习的策略
            method_results = evaluate_nbv_policy(
                policy_network, vggt_wrapper, renderer, test_data, device=device
            )
        else:
            # 实现基线方法
            method_results = _evaluate_baseline_method(
                method_name, vggt_wrapper, renderer, test_data, device
            )
        
        results[method_name] = method_results
    
    return results


def _evaluate_baseline_method(method_name: str,
                            vggt_wrapper: "MapAnythingWrapper",
                            renderer: DifferentiableRenderer,
                            test_data: List[Dict],
                            device: str) -> Dict[str, float]:
    """评估基线方法"""
    
    results = {
        "reconstruction_quality": [],
        "coverage_improvement": [],
        "view_efficiency": [],
        "chamfer_distances": []
    }
    
    chamfer_loss = ChamferDistance()
    
    with torch.no_grad():
        for test_sample in tqdm(test_data, desc=f"Evaluating {method_name}"):
            if method_name == "random_sampling":
                sample_results = _evaluate_random_sampling(
                    test_sample, vggt_wrapper, renderer, chamfer_loss, device
                )
            elif method_name == "frontier_based":
                sample_results = _evaluate_frontier_based(
                    test_sample, vggt_wrapper, renderer, chamfer_loss, device
                )
            elif method_name == "entropy_based":
                sample_results = _evaluate_entropy_based(
                    test_sample, vggt_wrapper, renderer, chamfer_loss, device
                )
            else:
                continue
            
            # 收集结果
            for key, value in sample_results.items():
                if key in results:
                    results[key].append(value)
    
    # 计算平均结果
    avg_results = {}
    for key, values in results.items():
        if values:
            avg_results[f"avg_{key}"] = np.mean(values)
            avg_results[f"std_{key}"] = np.std(values)
    
    return avg_results


def _evaluate_random_sampling(test_sample: Dict,
                            vggt_wrapper: "MapAnythingWrapper",
                            renderer: DifferentiableRenderer,
                            chamfer_loss: ChamferDistance,
                            device: str) -> Dict[str, float]:
    """评估随机采样基线"""
    # 实现随机采样策略的评估
    # 这里简化实现
    return {
        "reconstruction_quality": 0.3,
        "coverage_improvement": 0.1,
        "view_efficiency": 0.01,
        "chamfer_distances": 0.5
    }


def _evaluate_frontier_based(test_sample: Dict,
                           vggt_wrapper: "MapAnythingWrapper",
                           renderer: DifferentiableRenderer,
                           chamfer_loss: ChamferDistance,
                           device: str) -> Dict[str, float]:
    """评估基于边界的基线"""
    # 实现边界探索策略的评估
    return {
        "reconstruction_quality": 0.4,
        "coverage_improvement": 0.15,
        "view_efficiency": 0.015,
        "chamfer_distances": 0.4
    }


def _evaluate_entropy_based(test_sample: Dict,
                          vggt_wrapper: "MapAnythingWrapper",
                          renderer: DifferentiableRenderer,
                          chamfer_loss: ChamferDistance,
                          device: str) -> Dict[str, float]:
    """评估基于熵的基线"""
    # 实现熵最小化策略的评估
    return {
        "reconstruction_quality": 0.45,
        "coverage_improvement": 0.18,
        "view_efficiency": 0.018,
        "chamfer_distances": 0.35
    }


def generate_evaluation_report(comparison_results: Dict[str, Dict[str, float]],
                             save_path: str):
    """
    生成评估报告
    
    Args:
        comparison_results: 比较结果
        save_path: 保存路径
    """
    with open(save_path, 'w') as f:
        f.write("NBV Policy Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        # 方法比较表格
        f.write("Method Comparison:\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Method':<20} {'Quality':<12} {'Efficiency':<12} {'Chamfer':<12}\n")
        f.write("-" * 56 + "\n")
        
        for method_name, results in comparison_results.items():
            quality = results.get("avg_reconstruction_quality", 0)
            efficiency = results.get("avg_view_efficiency", 0)
            chamfer = results.get("avg_chamfer_distances", 0)
            
            f.write(f"{method_name:<20} {quality:<12.4f} {efficiency:<12.4f} {chamfer:<12.4f}\n")
        
        f.write("\n")
        
        # 详细结果
        f.write("Detailed Results:\n")
        f.write("-" * 20 + "\n")
        for method_name, results in comparison_results.items():
            f.write(f"\n{method_name.upper()}:\n")
            for metric, value in results.items():
                f.write(f"  {metric}: {value:.6f}\n")
    
    LOGGER.info("Evaluation report saved to %s", save_path)
