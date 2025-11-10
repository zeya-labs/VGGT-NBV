"""
数据加载器创建工具
提供创建各种数据加载器的便捷函数
"""

from typing import Optional, Callable
from torch.utils.data import DataLoader
from .base_dataset import BaseDataset
from .collate_functions import get_collate_fn


def create_data_loader(
    dataset: BaseDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        pin_memory: 是否使用固定内存
        drop_last: 是否丢弃最后不完整的批次
        collate_fn: 自定义的collate函数，如果为None则自动选择
        
    Returns:
        数据加载器
    """
    # 如果没有指定collate函数，根据数据集类型自动选择
    if collate_fn is None:
        dataset_class_name = dataset.__class__.__name__.lower()
        # 映射数据集类名到类型名
        if "synthetic" in dataset_class_name:
            dataset_type = "synthetic"
        elif "shapenet" in dataset_class_name:
            dataset_type = "shapenet"
        elif "modelnet" in dataset_class_name:
            dataset_type = "modelnet"
        else:
            dataset_type = "nbv"  # 默认类型
        
        collate_fn = get_collate_fn(dataset_type)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )


def create_train_loader(
    dataset: BaseDataset,
    batch_size: int = 4,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    创建训练数据加载器
    
    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        num_workers: 工作进程数
        **kwargs: 其他参数
        
    Returns:
        训练数据加载器
    """
    return create_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        **kwargs
    )


def create_val_loader(
    dataset: BaseDataset,
    batch_size: int = 4,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    创建验证数据加载器
    
    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        num_workers: 工作进程数
        **kwargs: 其他参数
        
    Returns:
        验证数据加载器
    """
    return create_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        **kwargs
    )


def create_test_loader(
    dataset: BaseDataset,
    batch_size: int = 1,
    num_workers: int = 1,
    **kwargs
) -> DataLoader:
    """
    创建测试数据加载器
    
    Args:
        dataset: 数据集实例
        batch_size: 批次大小（测试时通常为1）
        num_workers: 工作进程数
        **kwargs: 其他参数
        
    Returns:
        测试数据加载器
    """
    return create_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        **kwargs
    )
