"""
VGGT基础模型封装类

该类封装预训练的VGGT模型，提供两个核心功能：
1. 场景编码器：从多视图中提取高维场景特征
2. 质量评估器：生成三维重建并用于质量评估

关键改进：
- 正确利用aggregated_tokens_list的所有tokens信息
- 保持原始2048维度，不做无意义的降维  
- 移除不必要的复杂度评估和特征投影
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Sequence, Union
import sys
import os

from nbv_framework.utils.logging_utils import get_logger

# 添加vggt路径到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../vggt'))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry_torch import unproject_depth_map_to_point_map_torch

LOGGER = get_logger(__name__)


class VGGTWrapper(nn.Module):
    """
    VGGT基础模型封装类
    
    该类将预训练的VGGT模型封装为两个核心功能：
    1. 场景编码器：提取场景的高维特征表示
    2. 质量评估器：生成三维重建结果用于质量评估
    
    所有参数都被冻结，只作为特征提取器使用。
    """
    
    def __init__(self, model_name: str = "facebook/VGGT-1B", device: str = "cuda"):
        """
        初始化VGGT包装器
        
        Args:
            model_name: VGGT模型名称或路径
            device: 计算设备
        """
        super().__init__()
        
        self.device = device
        # 加载预训练的VGGT模型
        LOGGER.info("Loading VGGT model: %s", model_name)
        self.vggt_model = VGGT.from_pretrained(model_name).to(device)
        
        # 冻结所有参数
        for param in self.vggt_model.parameters():
            param.requires_grad = False
        
        self.vggt_model.eval()
        LOGGER.info("VGGT model loaded and frozen successfully")

        # 梯度捕获配置（用于TensorBoard分析）
        self._capture_gradients: bool = False
        self._grad_capture_keys: Tuple[str, ...] = ("depth")
        self._capture_input_grad: bool = False
        self._captured_tensors: Dict[str, torch.Tensor] = {}
        self._captured_input: Optional[torch.Tensor] = None

    def configure_gradient_capture(self,
                                   enable: bool = True,
                                   keys: Optional[Sequence[str]] = None,
                                   capture_input: bool = True) -> None:
        """配置VGGT输出的梯度捕获，用于调试和TensorBoard记录。

        Args:
            enable: 是否启用梯度捕获。
            keys: 需要观察梯度的`predictions`键集合。
            capture_input: 是否捕获输入图像张量的梯度。
        """
        self._capture_gradients = enable
        self._capture_input_grad = capture_input

        if keys is not None:
            self._grad_capture_keys = tuple(keys)

        if not enable:
            self._captured_tensors.clear()
            self._captured_input = None

    def _prepare_gradient_logging(self,
                                   images: torch.Tensor,
                                   predictions: Dict[str, torch.Tensor]) -> None:
        """在需要时对指定张量调用retain_grad以便后续记录梯度。"""
        if not self._capture_gradients:
            return

        # 捕获输出张量梯度
        self._captured_tensors = {}
        for key in self._grad_capture_keys:
            tensor = predictions.get(key)
            if tensor is None or not torch.is_tensor(tensor):
                continue
            if not tensor.requires_grad:
                continue
            tensor.retain_grad()
            self._captured_tensors[key] = tensor

        # 捕获输入图像梯度
        if self._capture_input_grad and images.requires_grad:
            images.retain_grad()
            self._captured_input = images
        else:
            self._captured_input = None

    def collect_gradient_stats(self) -> Dict[str, float]:
        """收集最近一次前向传播中保留的梯度统计信息。"""
        if not self._capture_gradients:
            return {}

        grad_stats: Dict[str, float] = {}

        for key, tensor in list(self._captured_tensors.items()):
            grad = tensor.grad
            # print(key, grad)
            if grad is None:
                continue
            grad_stats[f"{key}/grad_norm"] = grad.norm().detach().item()
            grad_stats[f"{key}/grad_mean_abs"] = grad.abs().mean().detach().item()

        if self._capture_input_grad and self._captured_input is not None:
            grad = self._captured_input.grad
            if grad is not None:
                grad_stats["input/grad_norm"] = grad.norm().detach().item()
                grad_stats["input/grad_mean_abs"] = grad.abs().mean().detach().item()

        self._captured_tensors.clear()
        self._captured_input = None

        return grad_stats
    
    def extract_scene_features(self, images: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        场景编码器功能：从多视图中提取场景特征
        
        根据用户指导正确实现：
        - aggregated_tokens_list长度为24，每个元素形状为[B, S, P, 2048] 
        - P = patch_num*patch_num + 1(camera token) + 4(register tokens)
        - 保持原始2048维度传给NBV头
        
        Args:
            images: 输入图像张量 [B, S, 3, H, W] 或 [S, 3, H, W]
            layer_idx: 使用哪一层的特征，-1表示最后一层
            
        Returns:
            scene_features: 场景特征张量 [B, S, P, 2048] （不做池化，保留所有tokens）
        """
        # 确保输入有batch维度
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        
        # 获取aggregated tokens列表
        # aggregated_tokens_list长度=24, 每个元素形状=[B, S, P, 2048]
        aggregated_tokens_list, patch_start_idx = self.vggt_model.aggregator(images)
        
        # 选择指定层的tokens
        if layer_idx == -1:
            layer_idx = len(aggregated_tokens_list) - 1
        
        tokens = aggregated_tokens_list[layer_idx]  # [B, S, P, 2048]
        return tokens
    
    def extract_token_features(self, images: torch.Tensor, layer_idx: int = -1, 
                              token_type: str = "all") -> torch.Tensor:
        """
        提取特定类型的token特征
        
        Args:
            images: 输入图像张量 [B, S, 3, H, W] 或 [S, 3, H, W]
            layer_idx: 使用哪一层的特征，-1表示最后一层
            token_type: token类型 "camera", "register", "patch", "all"
            
        Returns:
            token_features: 对应类型的token特征
        """
        # 确保输入有batch维度
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        
        # 获取aggregated tokens列表
        aggregated_tokens_list, patch_start_idx = self.vggt_model.aggregator(images)
        
        # 选择指定层的tokens
        if layer_idx == -1:
            layer_idx = len(aggregated_tokens_list) - 1
        
        tokens = aggregated_tokens_list[layer_idx]  # [B, S, P, 2048]
        
        if token_type == "camera":
            # 只返回camera token (索引0)
            return tokens[:, :, 0, :]  # [B, S, 2048]
        elif token_type == "register":
            # 返回register tokens (索引1-4)
            return tokens[:, :, 1:5, :]  # [B, S, 4, 2048]
        elif token_type == "patch":
            # 返回patch tokens (索引5以后)
            return tokens[:, :, patch_start_idx:, :]  # [B, S, patch_num, 2048]
        elif token_type == "all":
            # 返回所有tokens
            return tokens  # [B, S, P, 2048]
        else:
            raise ValueError(f"Unknown token_type: {token_type}")
    
    def reconstruct_and_evaluate(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        质量评估器功能：从多视图生成三维重建结果
        
        Args:
            images: 输入图像张量 [B, S, 3, H, W] 或 [S, 3, H, W]
            
        Returns:
            reconstruction_data: 包含重建结果的字典
                - world_points: 3D点云 [B, S, H, W, 3]
                - world_points_conf: 点云置信度 [B, S, H, W]
                - depth: 深度图 [B, S, H, W, 1]
                - depth_conf: 深度置信度 [B, S, H, W]
                - pose_enc: 相机位姿编码 [B, S, 9]
        """
        predictions = self.vggt_model(images)

        # # 输出predictions每个键的requires_grad
        # for key, tensor in predictions.items():
        #     if tensor is not None and torch.is_tensor(tensor):
        #         print(f"{key}: {tensor.requires_grad}")

        # print(f"Pose enc shape: {predictions['pose_enc'].shape}")
        # 将姿态编码转换为外参和内参矩阵

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # 从深度图生成世界坐标点
        depth_map = predictions["depth"]  # (B, S, H, W, 1)
        # print(f"Depth map shape: {depth_map.shape}")
        # print(f"Extrinsic shape: {predictions['extrinsic'].shape}")
        # print(f"Intrinsic shape: {predictions['intrinsic'].shape}")
        world_points = unproject_depth_map_to_point_map_torch(depth_map, predictions["extrinsic"], predictions["intrinsic"])
        predictions["world_points_from_depth"] = world_points

        # 如果启用了梯度捕获，则在返回前保留梯度信息
        self._prepare_gradient_logging(images, predictions)

        return predictions
    
    def forward(self, images: torch.Tensor, mode: str = "encode") -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            images: 输入图像
            mode: 模式 "encode" 或 "reconstruct"
            
        Returns:
            输出张量或字典
        """
        if mode == "encode":
            return self.extract_scene_features(images)
        elif mode == "reconstruct":
            return self.reconstruct_and_evaluate(images)
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported: encode, reconstruct")
