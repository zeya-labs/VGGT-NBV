"""
NBV框架数据集模块

该模块提供用于 Next Best View 任务的数据集基础设施，包括：
- 基础数据集抽象类
- 具体数据集实现
- 数据加载器和批处理函数
- 数据集工厂类
"""

# 基础数据集类
from .base_dataset import BaseDataset

# 具体数据集实现
from .mixed_dataset import MixedDataset
from .repeated_dataset import RepeatedDataset

# 数据集工厂
from .dataset_factory import DatasetFactory

# 数据加载器
from .data_loaders import (
    create_data_loader,
    create_train_loader,
    create_val_loader,
    create_test_loader,
)

# 批处理函数
from .collate_functions import (
    custom_nbv_collate_fn,
    get_collate_fn,
)

# 导出的公共接口
__all__ = [
    # 数据集类
    "BaseDataset",
    "MixedDataset",
    "RepeatedDataset",
    
    # 工厂类
    "DatasetFactory",
    
    # 数据加载器
    "create_data_loader",
    "create_train_loader", 
    "create_val_loader",
    "create_test_loader",
    
    # 批处理函数
    "custom_nbv_collate_fn",
    "get_collate_fn",
]
