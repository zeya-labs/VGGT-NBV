"""
数据集批处理函数
包含各种用于数据加载器的collate函数
"""

from typing import List, Dict, Any
import torch
from pytorch3d.structures import join_meshes_as_batch
from torch.utils.data.dataloader import default_collate


def custom_nbv_collate_fn(batch: List[Dict]) -> Dict:
    """
    为 NBV 数据集定制的 Collate Function
    
    该函数能够正确处理包含嵌套 PyTorch3D Meshes 对象的批次数据。
    它会:
    1. 从每个样本的 `gt_mesh_data` 字典中分离出 `Meshes` 对象
    2. 使用 `join_meshes_as_batch` 将 `Meshes` 对象列表打包成一个批次
    3. 使用 `default_collate` 处理批次中所有其他标准数据类型 (Tensors, lists, etc.)
    4. 将打包好的 Meshes 和其他数据重新组合成一个批次字典
    
    Args:
        batch: 数据样本列表
        
    Returns:
        批处理后的数据字典
    """
    if not isinstance(batch[0], dict):
        return default_collate(batch)

    # 初始化最终的批次字典
    final_batch = {}

    # 1. 提取并批量处理顶层数据 (除了 gt_mesh_data)
    special_keys = {'gt_mesh_data'}
    
    # 遍历第一个样本的所有键
    for key in batch[0].keys():
        # 如果不是特殊键，就用默认方式处理
        if key not in special_keys:
            final_batch[key] = default_collate([d[key] for d in batch])

    # 2. 专门处理嵌套的 'gt_mesh_data'
    if 'gt_mesh_data' in batch[0]:
        gt_mesh_data_list = [d['gt_mesh_data'] for d in batch]
        
        # 分离出 Meshes 对象
        original_meshes_list = [
            gd['original_mesh'] for gd in gt_mesh_data_list 
            if 'original_mesh' in gd
        ]
        normalized_meshes_list = [
            gd['normalized_mesh'] for gd in gt_mesh_data_list 
            if 'normalized_mesh' in gd
        ]
        
        # 使用 PyTorch3D 的函数批量处理 Meshes
        # 即使列表中只有一个元素，join_meshes_as_batch 也能正常工作
        if original_meshes_list:
            batched_original_mesh = join_meshes_as_batch(original_meshes_list)
        else:
            batched_original_mesh = None
            
        if normalized_meshes_list:
            batched_normalized_mesh = join_meshes_as_batch(normalized_meshes_list)
        else:
            batched_normalized_mesh = None

        # 分离出其他数据，让 default_collate 处理
        other_gt_data = []
        for gd in gt_mesh_data_list:
            item_copy = gd.copy()
            item_copy.pop('original_mesh', None)
            item_copy.pop('normalized_mesh', None)
            other_gt_data.append(item_copy)
        
        batched_other_gt_data = default_collate(other_gt_data)

        # 重新组合成 'gt_mesh_data' 批次字典
        final_batch['gt_mesh_data'] = batched_other_gt_data
        if batched_original_mesh is not None:
            final_batch['gt_mesh_data']['original_mesh'] = batched_original_mesh
        if batched_normalized_mesh is not None:
            final_batch['gt_mesh_data']['normalized_mesh'] = batched_normalized_mesh
        
    return final_batch

def get_collate_fn(dataset_type: str = "nbv"):
    """
    根据数据集类型获取合适的collate函数
    
    Args:
        dataset_type: 数据集类型
        
    Returns:
        对应的collate函数
    """
    if dataset_type in ["nbv", "synthetic", "shapenet", "modelnet"]:
        return custom_nbv_collate_fn
    else:
        return default_collate
