"""
数据加载器创建工具
提供创建各种数据加载器的便捷函数
"""

from typing import Optional, Callable
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Sampler
from .base_dataset import BaseDataset
from .collate_functions import get_collate_fn


def create_data_loader(
    dataset: BaseDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = True,
    collate_fn: Optional[Callable] = None,
    sampler: Optional[Sampler] = None,
    prefetch_factor: Optional[int] = 2,
) -> DataLoader:
    """
    创建数据加载器

    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        pin_memory: 是否使用固定内存
        persistent_workers: 是否保持工作进程持久化
        drop_last: 是否丢弃最后不完整的批次
        collate_fn: 自定义的collate函数，如果为None则自动选择
        sampler: 可选采样器（如 DistributedSampler）

    Returns:
        数据加载器
    """
    if collate_fn is None:
        dataset_type = "nbv"
        collate_fn = get_collate_fn(dataset_type)

    multiprocessing_context = None
    if num_workers > 0:
        multiprocessing_context = mp.get_context("spawn")
    else:
        persistent_workers = False

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
        sampler=sampler,
        multiprocessing_context=multiprocessing_context,
    )
    # prefetch_factor 仅在 num_workers > 0 时可用，否则 DataLoader 会报参数无效
    if prefetch_factor is not None and num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    
    from loguru import logger
    
    logger.info(f"创建数据加载器: {loader_kwargs}")
    return DataLoader(**loader_kwargs)


def create_train_loader(
    dataset: BaseDataset,
    batch_size: int = 4,
    num_workers: int = 4,
    sampler: Optional[Sampler] = None,
    **kwargs
) -> DataLoader:
    """
    创建训练数据加载器

    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        num_workers: 工作进程数
        sampler: 采样器（分布式训练时使用）
        **kwargs: 其他参数

    Returns:
        训练数据加载器
    """
    return create_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        sampler=sampler,
        **kwargs
    )


def create_val_loader(
    dataset: BaseDataset,
    batch_size: int = 4,
    num_workers: int = 4,
    sampler: Optional[Sampler] = None,
    **kwargs
) -> DataLoader:
    """
    创建验证数据加载器

    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        num_workers: 工作进程数
        sampler: 采样器（分布式验证时使用）
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
        persistent_workers=True,
        drop_last=False,
        sampler=sampler,
        **kwargs
    )


def create_test_loader(
    dataset: BaseDataset,
    batch_size: int = 1,
    num_workers: int = 1,
    sampler: Optional[Sampler] = None,
    **kwargs
) -> DataLoader:
    """
    创建测试数据加载器

    Args:
        dataset: 数据集实例
        batch_size: 批次大小（测试时通常为1）
        num_workers: 工作进程数
        sampler: 采样器
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
        sampler=sampler,
        **kwargs
    )
