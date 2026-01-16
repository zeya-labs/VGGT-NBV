"""
数据集批处理函数
包含各种用于数据加载器的collate函数
"""

from typing import List, Dict, Any
import torch
from pytorch3d.structures import join_meshes_as_batch
from torch.utils.data.dataloader import default_collate


def _collate_meshes(mesh_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """专门处理 mesh 相关字段，避免污染其余字典结构。"""
    original_meshes = [m for m in (d.get("original") for d in mesh_dicts) if m is not None]
    normalized_meshes = [m for m in (d.get("normalized") for d in mesh_dicts) if m is not None]

    batched: Dict[str, Any] = {}
    if original_meshes:
        batched["original"] = join_meshes_as_batch(original_meshes)
    if normalized_meshes:
        batched["normalized"] = join_meshes_as_batch(normalized_meshes)
    return batched


def custom_nbv_collate_fn(batch: List[Dict]) -> Dict:
    """结构化 collate：分离 inputs/targets/mesh/meta，Meshes 用专门逻辑拼批。"""
    if not isinstance(batch[0], dict):
        return default_collate(batch)

    final_batch: Dict[str, Any] = {}

    # 直接按 namespace 递归 collate
    for namespace in ("inputs", "targets"):
        if namespace in batch[0]:
            final_batch[namespace] = default_collate([sample.get(namespace, {}) for sample in batch])

    if "meta" in batch[0]:
        final_batch["meta"] = [sample.get("meta", {}) for sample in batch]

    # 单独处理 mesh 嵌套（Meshes 无法直接 default_collate）
    if "mesh" in batch[0]:
        mesh_dicts = [sample.get("mesh", {}) for sample in batch]
        final_batch["mesh"] = _collate_meshes(mesh_dicts)

    if not final_batch:
        raise TypeError("Batch samples must contain inputs/targets/meta/mesh namespaces")

    return final_batch

def get_collate_fn(dataset_type: str = "nbv"):
    """
    根据数据集类型获取合适的collate函数
    
    Args:
        dataset_type: 数据集类型
        
    Returns:
        对应的collate函数
    """
    if dataset_type in ["nbv"]:
        return custom_nbv_collate_fn
    else:
        return default_collate
