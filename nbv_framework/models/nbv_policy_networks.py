"""
NBV策略网络统一框架

提供多种NBV策略网络架构，支持[B, S, 2048]或[B, S, P, 2048]输入格式：
1. BasicNBVPolicy - 基础策略网络
2. AttentionNBVPolicy - 注意力机制策略网络
3. IterativeNBVPolicy - 迭代细化策略网络
4. MultiScaleNBVPolicy - 多尺度特征融合策略网络
5. HybridNBVPolicy - 混合架构策略网络
6. GeometryAwareNBVPolicy - 几何感知策略网络

所有网络统一接收[B, S, 2048]或[B, S, P, 2048]格式的场景特征，其中：
- B: batch size
- S: sequence length (多视角特征数量)
- P: token数量（相机/注册/patch等）
- 2048: VGGT特征维度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import math


class BaseNBVPolicy(nn.Module):
    """
    NBV策略网络基类
    
    提供通用的功能和接口定义
    """
    
    def __init__(
        self,
        output_mode: str = "cartesian",
        token_pooling_mode: str = "mean",
        position_bounds: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.token_pooling_mode = token_pooling_mode  # 处理[B, S, P, 2048]时的token池化方式
        if position_bounds is None:
            position_bounds = (-3.0,3.0)
        if position_bounds[0] >= position_bounds[1]:
            raise ValueError(
                f"Invalid position_bounds: {position_bounds}. Expected (min, max) with min < max"
            )
        self.position_bounds = position_bounds
        
        if output_mode == "spherical":
            self.target_dim = 7  # theta, phi, radius + qx, qy, qz, qw (球面位置+四元数旋转)
        elif output_mode == "cartesian":
            self.target_dim = 7  # x, y, z, qx, qy, qz, qw (笛卡尔位置+四元数旋转)
        elif output_mode == "euler":
            self.target_dim = 6  # x, y, z, roll, pitch, yaw (笛卡尔位置+欧拉角旋转)
        elif output_mode == "position_only":
            self.target_dim = 3  # x, y, z (仅笛卡尔位置，姿态自动确定)
        else:
            raise ValueError(f"Unknown output_mode: {output_mode}. Supported: spherical, cartesian, euler, position_only")

    def _pool_tokens_if_needed(self, scene_features: torch.Tensor) -> torch.Tensor:
        """如果输入为[B, S, P, D]，按token维度P进行池化，返回[B, S, D]。
        支持: mean/max/camera(取camera token索引0)。
        """
        if scene_features.dim() == 4:
            if self.token_pooling_mode == "mean":
                return scene_features.mean(dim=2)
            if self.token_pooling_mode == "max":
                return scene_features.max(dim=2)[0]
            if self.token_pooling_mode == "camera":
                return scene_features[:, :, 0, :]
            raise ValueError(f"Unknown token_pooling_mode: {self.token_pooling_mode}. Supported: mean, max, camera")
        return scene_features
    
    def _activate_nbv(self, nbv: torch.Tensor) -> torch.Tensor:
        """激活NBV预测，约束输出范围"""
        if self.output_mode == "spherical":
            # 球面位置: theta, phi, radius
            theta = torch.sigmoid(nbv[:, 0]) * 2 * math.pi  # [0, 2π]
            phi = torch.sigmoid(nbv[:, 1]) * math.pi        # [0, π]
            radius = torch.sigmoid(nbv[:, 2]) * 2 + 1       # [1, 3]
            position = torch.stack([theta, phi, radius], dim=1)
            
            # 四元数旋转: qx, qy, qz, qw
            quaternion = F.normalize(nbv[:, 3:], p=2, dim=1)
            
            return torch.cat([position, quaternion], dim=1)
            
        elif self.output_mode == "cartesian":
            # 笛卡尔位置: x, y, z
            position = nbv[:, :3]
            # 四元数旋转: qx, qy, qz, qw
            quaternion = F.normalize(nbv[:, 3:], p=2, dim=1)
            return torch.cat([position, quaternion], dim=1)
            
        elif self.output_mode == "euler":
            # 笛卡尔位置: x, y, z
            position = nbv[:, :3]
            # 欧拉角旋转: roll, pitch, yaw (弧度)
            roll = torch.tanh(nbv[:, 3]) * math.pi    # [-π, π]
            pitch = torch.tanh(nbv[:, 4]) * math.pi/2 # [-π/2, π/2]
            yaw = torch.tanh(nbv[:, 5]) * math.pi     # [-π, π]
            rotation = torch.stack([roll, pitch, yaw], dim=1)
            
            return torch.cat([position, rotation], dim=1)
            
        elif self.output_mode == "position_only":
            # 仅笛卡尔位置: x, y, z (姿态将由其他方式自动确定)
            # lower, upper = self.position_bounds
            # 强制限制在[min, max]区间，避免训练过程中位置发散
            # position = torch.tanh(nbv[:, :3])
            # midpoint = (upper + lower) * 0.5
            # half_range = (upper - lower) * 0.5
            # position = position * half_range + midpoint

            # position = torch.clamp(nbv[:, :3], min=lower, max=upper)
            
            position = nbv[:, :3]
            return position
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Parameter):
                nn.init.xavier_uniform_(m)
    
    def spherical_to_cartesian(self, spherical_pose: torch.Tensor) -> torch.Tensor:
        """将球坐标转换为笛卡尔坐标"""
        theta, phi, radius = spherical_pose[:, 0], spherical_pose[:, 1], spherical_pose[:, 2]
        
        x = radius * torch.sin(phi) * torch.cos(theta)
        y = radius * torch.sin(phi) * torch.sin(theta)
        z = radius * torch.cos(phi)
        
        return torch.stack([x, y, z], dim=1)
    
    def euler_to_quaternion(self, euler_angles: torch.Tensor) -> torch.Tensor:
        """将欧拉角转换为四元数 (roll, pitch, yaw) -> (qx, qy, qz, qw)"""
        roll, pitch, yaw = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
        
        # 计算半角
        cr, cp, cy = torch.cos(roll * 0.5), torch.cos(pitch * 0.5), torch.cos(yaw * 0.5)
        sr, sp, sy = torch.sin(roll * 0.5), torch.sin(pitch * 0.5), torch.sin(yaw * 0.5)
        
        # 四元数分量
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        
        return torch.stack([qx, qy, qz, qw], dim=1)
    
    def quaternion_to_euler(self, quaternion: torch.Tensor) -> torch.Tensor:
        """将四元数转换为欧拉角 (qx, qy, qz, qw) -> (roll, pitch, yaw)"""
        qx, qy, qz, qw = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
        
        # 归一化四元数
        norm = torch.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        # 计算欧拉角
        roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = torch.asin(2 * (qw * qy - qz * qx))
        yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        
        return torch.stack([roll, pitch, yaw], dim=1)

class BasicNBVPolicy(BaseNBVPolicy):
    """
    基础NBV策略网络
    
    接收[B, S, 2048]场景特征，通过简单的池化和MLP输出相机位姿
    """
    
    def __init__(self, 
                 scene_feature_dim: int = 2048,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 pooling_mode: str = "mean",
                 output_mode: str = "cartesian",
                 token_pooling_mode: str = "mean"):
        """
        初始化基础NBV策略网络
        
        Args:
            scene_feature_dim: 场景特征维度
            hidden_dim: 隐藏层维度
            num_layers: MLP层数
            pooling_mode: 池化模式 "mean", "max", "attention"
            output_mode: 输出模式 "spherical" 或 "cartesian"
        """
        super().__init__(output_mode, token_pooling_mode)
        
        self.scene_feature_dim = scene_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling_mode = pooling_mode
        
        # 特征池化
        if pooling_mode == "attention":
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=scene_feature_dim,
                num_heads=8,
                batch_first=True
            )
            self.pool_query = nn.Parameter(torch.randn(1, 1, scene_feature_dim))
        
        # 输入归一化
        self.input_norm = nn.LayerNorm(scene_feature_dim)
        
        # MLP网络
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(scene_feature_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
        
        self.backbone = nn.Sequential(*layers)
        
        # 输出头
        self.output_head = nn.Linear(hidden_dim, self.target_dim)
        
        self._initialize_weights()
    
    def forward(self, scene_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            scene_features: 场景特征 [B, S, 2048]
            
        Returns:
            camera_pose: 相机位姿 [B, target_dim]
        """
        scene_features = self._pool_tokens_if_needed(scene_features)  # [B, S, D]
        B, S, D = scene_features.shape
        
        # 归一化
        x = self.input_norm(scene_features)  # [B, S, D]
        
        # 特征池化
        if self.pooling_mode == "mean":
            pooled_features = x.mean(dim=1)  # [B, D]
        elif self.pooling_mode == "max":
            pooled_features = x.max(dim=1)[0]  # [B, D]
        elif self.pooling_mode == "attention":
            query = self.pool_query.expand(B, -1, -1)  # [B, 1, D]
            pooled_features, _ = self.attention_pool(query, x, x)  # [B, 1, D]
            pooled_features = pooled_features.squeeze(1)  # [B, D]
        
        # MLP处理
        features = self.backbone(pooled_features)  # [B, hidden_dim]
        
        # 输出预测
        nbv_raw = self.output_head(features)  # [B, target_dim]
        
        return self._activate_nbv(nbv_raw)


class AttentionNBVPolicy(BaseNBVPolicy):
    """
    基于注意力机制的NBV策略网络
    
    使用Transformer编码器处理序列特征
    """
    
    def __init__(self, 
                 scene_feature_dim: int = 2048,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 output_mode: str = "cartesian",
                 token_pooling_mode: str = "mean"):
        """
        初始化注意力NBV策略网络
        
        Args:
            scene_feature_dim: 场景特征维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            output_mode: 输出模式
        """
        super().__init__(output_mode, token_pooling_mode)
        
        self.scene_feature_dim = scene_feature_dim
        self.hidden_dim = hidden_dim
        
        # 特征投影
        self.feature_projection = nn.Linear(scene_feature_dim, hidden_dim)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, hidden_dim))  # 支持最多100个视角
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 全局特征提取
        self.global_pool = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, self.target_dim)
        )
        
        self._initialize_weights()
    
    def forward(self, scene_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            scene_features: 场景特征 [B, S, 2048]
            
        Returns:
            camera_pose: 相机位姿 [B, target_dim]
        """
        scene_features = self._pool_tokens_if_needed(scene_features)  # [B, S, D]
        B, S, D = scene_features.shape
        
        # 特征投影
        x = self.feature_projection(scene_features)  # [B, S, hidden_dim]
        
        # 添加位置编码
        x = x + self.pos_embedding[:, :S, :]
        
        # Transformer编码
        encoded_features = self.transformer(x)  # [B, S, hidden_dim]
        
        # 全局特征提取
        global_token = self.global_token.expand(B, -1, -1)
        global_features, _ = self.global_pool(global_token, encoded_features, encoded_features)
        global_features = global_features.squeeze(1)  # [B, hidden_dim]
        
        # 输出预测
        nbv_raw = self.output_head(global_features)
        
        return self._activate_nbv(nbv_raw)


class IterativeNBVPolicy(BaseNBVPolicy):
    """
    迭代细化NBV策略网络
    
    通过多次迭代逐步精化NBV预测
    """
    
    def __init__(self,
                 scene_feature_dim: int = 2048,
                 hidden_dim: int = 512,
                 trunk_depth: int = 4,
                 num_heads: int = 8,
                 output_mode: str = "cartesian",
                 num_iterations: int = 4,
                 token_pooling_mode: str = "mean"):
        """
        初始化迭代细化NBV策略网络
        """
        super().__init__(output_mode, token_pooling_mode)
        
        self.scene_feature_dim = scene_feature_dim
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        
        # 场景特征编码器
        self.scene_encoder = AttentionNBVPolicy(
            scene_feature_dim, hidden_dim, num_heads, trunk_depth, output_mode
        )
        
        # 学习的空NBV token
        self.empty_nbv_token = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_nbv = nn.Linear(self.target_dim, hidden_dim)
        
        # 调制模块
        self.nbv_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim)
        )
        
        # 自适应层归一化
        self.adaln_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # Trunk网络
        self.trunk = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(trunk_depth)
        ])
        
        # NBV预测分支
        self.nbv_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, self.target_dim)
        )
        
        self._initialize_weights()
    
    def _modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """调制函数"""
        return x * (1 + scale) + shift
    
    def forward(self, scene_features: torch.Tensor, num_iterations: Optional[int] = None) -> List[torch.Tensor]:
        """
        迭代细化前向传播
        
        Args:
            scene_features: 场景特征 [B, S, 2048]
            num_iterations: 迭代次数
            
        Returns:
            nbv_predictions: NBV预测列表
        """
        if num_iterations is None:
            num_iterations = self.num_iterations
        
        scene_features = self._pool_tokens_if_needed(scene_features)  # [B, S, D]
        B = scene_features.shape[0]
        
        # 场景特征编码
        encoded_scene = self.scene_encoder.feature_projection(scene_features)  # [B, S, hidden_dim]
        encoded_scene = self.scene_encoder.transformer(encoded_scene)
        
        # 全局场景特征
        global_token = self.scene_encoder.global_token.expand(B, -1, -1)
        scene_global, _ = self.scene_encoder.global_pool(global_token, encoded_scene, encoded_scene)
        scene_tokens = scene_global  # [B, 1, hidden_dim]
        
        pred_nbv = None
        nbv_predictions = []
        
        for _ in range(num_iterations):
            # NBV输入
            if pred_nbv is None:
                nbv_input = self.embed_nbv(self.empty_nbv_token.expand(B, -1, -1))
            else:
                pred_nbv_detached = pred_nbv.detach()
                nbv_input = self.embed_nbv(pred_nbv_detached.unsqueeze(1))
            
            # 生成调制参数
            shift, scale, gate = self.nbv_modulation(nbv_input).chunk(3, dim=-1)
            
            # 自适应层归一化和调制
            scene_modulated = gate * self._modulate(self.adaln_norm(scene_tokens), shift, scale)
            scene_modulated = scene_modulated + scene_tokens
            
            # Trunk处理
            processed_tokens = self.trunk(scene_modulated)
            
            # 预测NBV增量
            nbv_delta = self.nbv_branch(processed_tokens.squeeze(1))
            
            if pred_nbv is None:
                pred_nbv = nbv_delta
            else:
                pred_nbv = pred_nbv + nbv_delta
            
            # 激活约束
            activated_nbv = self._activate_nbv(pred_nbv)
            nbv_predictions.append(activated_nbv)
        
        return nbv_predictions


class MultiScaleNBVPolicy(BaseNBVPolicy):
    """
    多尺度特征融合NBV策略网络
    
    从不同尺度的特征中预测NBV
    """
    
    def __init__(self,
                 scene_feature_dim: int = 2048,
                 feature_scales: List[int] = [512, 1024, 1536, 2048],
                 hidden_dim: int = 512,
                 output_mode: str = "cartesian",
                 token_pooling_mode: str = "mean"):
        """
        初始化多尺度NBV策略网络
        """
        super().__init__(output_mode, token_pooling_mode)
        
        self.scene_feature_dim = scene_feature_dim
        self.feature_scales = feature_scales
        self.hidden_dim = hidden_dim
        
        # 序列特征编码器
        self.sequence_encoder = AttentionNBVPolicy(
            scene_feature_dim, hidden_dim, num_layers=2
        )
        
        # 多尺度投影层
        self.scale_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, scale),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(scale)
            ) for scale in feature_scales
        ])
        
        # 特征金字塔融合
        self.pyramid_fusion = self._build_pyramid_fusion()
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(feature_scales[-1], feature_scales[-1] // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_scales[-1] // 2, self.target_dim)
        )
        
        self._initialize_weights()
    
    def _build_pyramid_fusion(self):
        """构建特征金字塔融合模块"""
        fusion_modules = nn.ModuleList()
        
        for i in range(len(self.feature_scales) - 1):
            in_dim = self.feature_scales[i]
            out_dim = self.feature_scales[i + 1]
            
            fusion_module = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.LayerNorm(out_dim)
            )
            fusion_modules.append(fusion_module)
        
        return fusion_modules
    
    def forward(self, scene_features: torch.Tensor) -> torch.Tensor:
        """
        多尺度特征融合前向传播
        
        Args:
            scene_features: 场景特征 [B, S, 2048]
            
        Returns:
            nbv_prediction: NBV预测 [B, target_dim]
        """
        scene_features = self._pool_tokens_if_needed(scene_features)  # [B, S, D]
        B, S, D = scene_features.shape
        
        # 序列特征编码
        encoded_features = self.sequence_encoder.feature_projection(scene_features)
        encoded_features = self.sequence_encoder.transformer(encoded_features)
        
        # 全局特征提取
        global_token = self.sequence_encoder.global_token.expand(B, -1, -1)
        global_features, _ = self.sequence_encoder.global_pool(
            global_token, encoded_features, encoded_features
        )
        global_features = global_features.squeeze(1)  # [B, hidden_dim]
        
        # 多尺度投影
        scale_features = []
        for projector in self.scale_projectors:
            scale_feature = projector(global_features)
            scale_features.append(scale_feature)
        
        # 特征金字塔融合
        fused_feature = scale_features[0]
        for i, fusion_module in enumerate(self.pyramid_fusion):
            next_scale_feature = scale_features[i + 1]
            upsampled_feature = fusion_module(fused_feature)
            fused_feature = upsampled_feature + next_scale_feature
        
        # 输出预测
        nbv_raw = self.output_head(fused_feature)
        
        return self._activate_nbv(nbv_raw)


class HybridNBVPolicy(BaseNBVPolicy):
    """
    混合架构NBV策略网络
    
    结合迭代细化和多尺度特征融合的优势
    """
    
    def __init__(self,
                 scene_feature_dim: int = 2048,
                 hidden_dim: int = 512,
                 feature_scales: List[int] = [1024, 1536, 2048],
                 num_iterations: int = 3,
                 output_mode: str = "cartesian",
                 token_pooling_mode: str = "mean"):
        """
        初始化混合架构NBV策略网络
        """
        super().__init__(output_mode, token_pooling_mode)
        
        self.scene_feature_dim = scene_feature_dim
        self.hidden_dim = hidden_dim
        
        # 多尺度特征提取
        self.multi_scale_extractor = MultiScaleNBVPolicy(
            scene_feature_dim, feature_scales, hidden_dim, output_mode, token_pooling_mode
        )
        
        # 迭代细化模块
        self.iterative_refiner = IterativeNBVPolicy(
            scene_feature_dim, hidden_dim, trunk_depth=2,
            num_iterations=num_iterations, output_mode=output_mode, token_pooling_mode=token_pooling_mode
        )
        
        # 预测融合器
        self.prediction_fusion = nn.Sequential(
            nn.Linear(self.target_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.target_dim)
        )
        
        self._initialize_weights()
    
    def forward(self, scene_features: torch.Tensor) -> torch.Tensor:
        """
        混合架构前向传播
        
        Args:
            scene_features: 场景特征 [B, S, 2048]
            
        Returns:
            nbv_prediction: 最终NBV预测 [B, target_dim]
        """
        # 多尺度预测
        multi_scale_pred = self.multi_scale_extractor(scene_features)
        
        # 迭代细化预测
        iterative_preds = self.iterative_refiner(scene_features)
        final_iterative_pred = iterative_preds[-1]
        
        # 融合两种预测
        combined_pred = torch.cat([multi_scale_pred, final_iterative_pred], dim=1)
        fused_pred = self.prediction_fusion(combined_pred)
        
        return self._activate_nbv(fused_pred)


class GeometryAwareNBVPolicy(BaseNBVPolicy):
    """
    几何感知NBV策略网络
    
    结合场景特征和几何信息进行NBV预测
    """
    
    def __init__(self,
                 scene_feature_dim: int = 2048,
                 geometry_feature_dim: int = 7,
                 hidden_dim: int = 512,
                 output_mode: str = "cartesian",
                 token_pooling_mode: str = "mean"):
        """
        初始化几何感知NBV策略网络
        
        Args:
            scene_feature_dim: 场景特征维度
            geometry_feature_dim: 几何特征维度 (深度特征3 + 点云特征4)
            hidden_dim: 隐藏层维度
            output_mode: 输出模式
        """
        super().__init__(output_mode, token_pooling_mode)
        
        self.scene_feature_dim = scene_feature_dim
        self.geometry_feature_dim = geometry_feature_dim
        self.hidden_dim = hidden_dim
        
        # 场景特征编码器
        self.scene_encoder = AttentionNBVPolicy(
            scene_feature_dim, hidden_dim, num_layers=3
        )
        
        # 几何特征编码器
        self.geometry_encoder = nn.Sequential(
            nn.Linear(geometry_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # 跨模态注意力融合
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.target_dim)
        )
        
        self._initialize_weights()
    
    def forward(self, scene_features: torch.Tensor, geometry_features: torch.Tensor) -> torch.Tensor:
        """
        几何感知前向传播
        
        Args:
            scene_features: 场景特征 [B, S, 2048]
            geometry_features: 几何特征 [B, geometry_feature_dim]
            
        Returns:
            nbv_prediction: NBV预测 [B, target_dim]
        """
        scene_features = self._pool_tokens_if_needed(scene_features)  # [B, S, D]
        B = scene_features.shape[0]
        
        # 场景特征编码
        encoded_scene = self.scene_encoder.feature_projection(scene_features)
        encoded_scene = self.scene_encoder.transformer(encoded_scene)
        
        # 全局场景特征
        global_token = self.scene_encoder.global_token.expand(B, -1, -1)
        scene_global, _ = self.scene_encoder.global_pool(global_token, encoded_scene, encoded_scene)
        scene_global = scene_global.squeeze(1)  # [B, hidden_dim]
        
        # 几何特征编码
        geometry_encoded = self.geometry_encoder(geometry_features)  # [B, hidden_dim//2]
        
        # 跨模态注意力
        scene_attended = scene_global.unsqueeze(1)  # [B, 1, hidden_dim]
        attended_scene, _ = self.cross_attention(scene_attended, scene_attended, scene_attended)
        attended_scene = attended_scene.squeeze(1)  # [B, hidden_dim]
        
        # 融合特征
        combined_features = torch.cat([attended_scene, geometry_encoded], dim=1)
        
        # 预测NBV
        nbv_raw = self.fusion_network(combined_features)
        
        return self._activate_nbv(nbv_raw)
