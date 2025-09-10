"""
数据集与数据加载工具 (已弃用)

警告：此模块已被移动到 nbv_framework.datasets
请使用新的数据集模块：
    from nbv_framework.datasets import SyntheticDataset, create_data_loader

此文件保留用于向后兼容，但将在未来版本中移除。
"""

import warnings
warnings.warn(
    "nbv_framework.utils.datasets 已弃用，请使用 nbv_framework.datasets",
    DeprecationWarning,
    stacklevel=2
)

# 导入新模块以保持向后兼容
from ..datasets import SyntheticDataset as NBVDataset
from ..datasets import create_data_loader, custom_nbv_collate_fn

__all__ = [
    "NBVDataset",
    "create_data_loader",
]
