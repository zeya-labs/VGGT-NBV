"""
数据集工厂类
提供统一的数据集创建接口
"""

from typing import Dict, Any, Type
from .base_dataset import BaseDataset
from .house3k_dataset import House3KDataset
from loguru import logger


class DatasetFactory:
    """
    数据集工厂类
    
    提供统一的接口来创建不同类型的数据集
    """
    
    # 注册的数据集类型
    _dataset_registry: Dict[str, Type[BaseDataset]] = {
        "house3k": House3KDataset,
    }
    
    @classmethod
    def register_dataset(cls, name: str, dataset_class: Type[BaseDataset]):
        """
        注册新的数据集类型
        
        Args:
            name: 数据集类型名称
            dataset_class: 数据集类
        """
        cls._dataset_registry[name] = dataset_class
        logger.info(f"成功注册数据集类型: {name}")
    
    @classmethod
    def create_dataset(
        cls, # 类方法，可以访问类属性，但不能访问实例属性
        dataset_type: str,
        data_root: str,
        **kwargs
    ) -> BaseDataset:
        """
        创建数据集实例
        
        Args:
            dataset_type: 数据集类型 ("house3k" 等)
            data_root: 数据根目录
            **kwargs: 传递给数据集构造函数的其他参数
            
        Returns:
            数据集实例
            
        Raises:
            ValueError: 如果数据集类型未注册
        """
        if dataset_type not in cls._dataset_registry:
            raise ValueError(
                f"Unknown dataset type: {dataset_type}. "
                f"Available types: {list(cls._dataset_registry.keys())}"
            )
        
        dataset_class = cls._dataset_registry[dataset_type]
        return dataset_class(data_root=data_root, **kwargs)
    
    @classmethod
    def get_available_types(cls) -> list:
        """获取所有可用的数据集类型"""
        return list(cls._dataset_registry.keys())

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> BaseDataset:
        """
        从配置字典创建数据集

        Args:
            config: 包含数据集配置的字典，必须包含 'type' 和 'data_root' 字段

        Returns:
            数据集实例
        """
        config = config.copy()

        if "type" not in config:
            raise ValueError("Config must contain 'type' field")
        if "data_root" not in config:
            raise ValueError("Config must contain 'data_root' field")

        dataset_type = config.pop('type')
        return cls.create_dataset(dataset_type, **config)
