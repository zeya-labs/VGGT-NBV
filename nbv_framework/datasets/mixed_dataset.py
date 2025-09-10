"""
混合数据集类
支持将多个不同类型的数据集合并成一个统一的数据集
"""

import random
from typing import List, Dict, Any, Union
import torch
from torch.utils.data import Dataset

from .base_dataset import BaseDataset
from .dataset_factory import DatasetFactory


class MixedDataset(Dataset):
    """
    混合数据集类
    
    可以将多个不同类型的数据集（如合成数据集、ShapeNet、ModelNet等）
    合并成一个统一的数据集进行训练
    """
    
    def __init__(
        self,
        dataset_configs: List[Dict[str, Any]],
        sampling_strategy: str = "uniform",
        weights: List[float] = None,
        seed: int = None
    ):
        """
        初始化混合数据集
        
        Args:
            dataset_configs: 数据集配置列表，每个配置包含数据集类型和参数
            sampling_strategy: 采样策略 ("uniform", "weighted", "sequential")
            weights: 各数据集的采样权重（仅在weighted策略下使用）
            seed: 随机种子
        """
        self.dataset_configs = dataset_configs
        self.sampling_strategy = sampling_strategy
        self.weights = weights
        self.seed = seed
        
        # 创建局部随机数生成器
        self.rng = random.Random(seed)
        
        # 创建各个子数据集
        self.datasets = []
        self.dataset_names = []
        self.cumulative_lengths = []
        
        total_length = 0
        for i, config in enumerate(dataset_configs):
            # 创建数据集
            try:
                dataset = DatasetFactory.create_from_config(config)
            except Exception as e:
                print(f"创建数据集失败: {e}")
                continue
            self.datasets.append(dataset)
            
            # 记录数据集名称
            dataset_name = config.get('name', f'dataset_{i}')
            self.dataset_names.append(dataset_name)
            
            # 记录累积长度（用于sequential策略）
            total_length += len(dataset)
            self.cumulative_lengths.append(total_length)
        
        self.total_length = total_length
        
        # 验证权重
        if sampling_strategy == "weighted":
            if weights is None:
                # 默认权重为各数据集长度的比例
                self.weights = [len(ds) / self.total_length for ds in self.datasets]
            else:
                assert len(weights) == len(self.datasets), \
                    f"权重数量({len(weights)})必须等于数据集数量({len(self.datasets)})"
                # 归一化权重
                weight_sum = sum(weights)
                self.weights = [w / weight_sum for w in weights]
        
        print(f"混合数据集创建成功:")
        for i, (name, dataset) in enumerate(zip(self.dataset_names, self.datasets)):
            weight_info = f", 权重: {self.weights[i]:.3f}" if self.weights else ""
            print(f"  - {name}: {len(dataset)} 样本{weight_info}")
        print(f"总计: {self.total_length} 样本")
    
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取样本
        
        根据采样策略从不同的子数据集中获取样本
        """
        if self.sampling_strategy == "sequential":
            return self._get_sequential_sample(idx)
        elif self.sampling_strategy == "uniform":
            return self._get_uniform_sample(idx)
        elif self.sampling_strategy == "weighted":
            return self._get_weighted_sample(idx)
        else:
            raise ValueError(f"不支持的采样策略: {self.sampling_strategy}")
    
    def _get_sequential_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """顺序采样：按数据集顺序依次获取样本"""
        for i, cumulative_length in enumerate(self.cumulative_lengths):
            if idx < cumulative_length:
                # 计算在当前数据集中的索引
                if i == 0:
                    dataset_idx = idx
                else:
                    dataset_idx = idx - self.cumulative_lengths[i-1]
                
                sample = self.datasets[i][dataset_idx]
                # 添加数据集来源信息
                sample["source_dataset"] = self.dataset_names[i]
                sample["source_dataset_idx"] = i
                return sample
        
        raise IndexError(f"索引 {idx} 超出范围")
    
    def _get_uniform_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """均匀采样：随机选择数据集，然后随机选择样本"""
        # 使用局部RNG随机选择一个数据集
        dataset_idx = self.rng.randint(0, len(self.datasets) - 1)
        dataset = self.datasets[dataset_idx]
        
        # 使用局部RNG随机选择该数据集中的一个样本
        sample_idx = self.rng.randint(0, len(dataset) - 1)
        sample = dataset[sample_idx]
        
        # 添加数据集来源信息
        sample["source_dataset"] = self.dataset_names[dataset_idx]
        sample["source_dataset_idx"] = dataset_idx
        return sample
    
    def _get_weighted_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """加权采样：根据权重随机选择数据集"""
        # 使用局部RNG根据权重随机选择数据集
        dataset_idx = self.rng.choices(
            range(len(self.datasets)), 
            weights=self.weights, 
            k=1
        )[0]
        
        dataset = self.datasets[dataset_idx]
        
        # 使用局部RNG随机选择该数据集中的一个样本
        sample_idx = self.rng.randint(0, len(dataset) - 1)
        sample = dataset[sample_idx]
        
        # 添加数据集来源信息
        sample["source_dataset"] = self.dataset_names[dataset_idx]
        sample["source_dataset_idx"] = dataset_idx
        return sample
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取混合数据集信息"""
        return {
            "mixed_dataset": True,
            "total_samples": self.total_length,
            "num_datasets": len(self.datasets),
            "sampling_strategy": self.sampling_strategy,
            "weights": self.weights,
            "datasets": [
                {
                    "name": name,
                    "type": dataset.__class__.__name__,
                    "samples": len(dataset),
                    "info": dataset.dataset_info if hasattr(dataset, 'dataset_info') else {}
                }
                for name, dataset in zip(self.dataset_names, self.datasets)
            ]
        }
    
    def get_dataset_by_name(self, name: str) -> BaseDataset:
        """根据名称获取子数据集"""
        for dataset_name, dataset in zip(self.dataset_names, self.datasets):
            if dataset_name == name:
                return dataset
        raise ValueError(f"未找到名称为 '{name}' 的数据集")
    
    def get_samples_by_dataset(self, dataset_name: str, num_samples: int = 5) -> List[Dict]:
        """获取指定数据集的样本用于检查"""
        dataset = self.get_dataset_by_name(dataset_name)
        samples = []
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            sample["source_dataset"] = dataset_name
            samples.append(sample)
        return samples

