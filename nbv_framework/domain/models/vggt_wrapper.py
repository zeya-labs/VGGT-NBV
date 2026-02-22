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
from typing import Dict, Union

from loguru import logger

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry_torch import unproject_depth_map_to_point_map_torch
except ModuleNotFoundError:
    from vggt.vggt.models.vggt import VGGT  # type: ignore
    from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore
    from vggt.vggt.utils.geometry_torch import unproject_depth_map_to_point_map_torch  # type: ignore


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
        logger.info("Loading VGGT model: {}", model_name)
        self.vggt_model = VGGT.from_pretrained(model_name).to(device)
        
        # 冻结所有参数
        for param in self.vggt_model.parameters():
            param.requires_grad = False
        
        self.vggt_model.eval()
        logger.info("VGGT model loaded and frozen successfully")
    
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
