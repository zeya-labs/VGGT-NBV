"""
数据集批处理函数
包含各种用于数据加载器的collate函数
"""

from typing import List, Dict, Any
from pytorch3d.structures import join_meshes_as_batch
from torch.utils.data.dataloader import default_collate


def _collate_meshes(mesh_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """专门处理 mesh 相关字段，避免污染其余字典结构。"""
    batched: Dict[str, Any] = {}
    for key in ("original", "normalized"):
        values = [d.get(key) for d in mesh_dicts]
        if all(v is None for v in values):
            continue
        if any(v is None for v in values):
            batched[key] = values
        else:
            batched[key] = join_meshes_as_batch(values)
    return batched


def _collate_optional(values: List[Any]) -> Any:
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        return list(values)
    first = values[0]
    if isinstance(first, dict):
        keys = set()
        for item in values:
            if isinstance(item, dict):
                keys.update(item.keys())
        result: Dict[str, Any] = {}
        for key in keys:
            sub_values = [item.get(key) if isinstance(item, dict) else None for item in values]
            collated = _collate_optional(sub_values)
            if collated is not None:
                result[key] = collated
        return result
    return default_collate(values)


def custom_nbv_collate_fn(batch: List[Dict]) -> Dict:
    """结构化 collate：分离 inputs/targets/mesh/meta，Meshes 用专门逻辑拼批。"""
    if not isinstance(batch[0], dict):
        return default_collate(batch)

    final_batch: Dict[str, Any] = {}

    # 直接按 namespace 递归 collate（支持缺失/None）
    for namespace in ("inputs", "targets"):
        if namespace in batch[0]:
            final_batch[namespace] = _collate_optional([sample.get(namespace, {}) for sample in batch])

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
