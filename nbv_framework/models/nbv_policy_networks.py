"""NBV policy networks.

Only the attention-based policy is kept as the supported architecture.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNBVPolicy(nn.Module):
    """Shared utilities for NBV policy heads."""

    def __init__(
        self,
        output_mode: str = "cartesian",
        token_pooling_mode: str = "mean",
        position_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__()
        self.output_mode = output_mode
        self.token_pooling_mode = token_pooling_mode

        if position_bounds is None:
            position_bounds = (-3.0, 3.0)
        if position_bounds[0] >= position_bounds[1]:
            raise ValueError(
                f"Invalid position_bounds: {position_bounds}. Expected (min, max) with min < max"
            )
        self.position_bounds = position_bounds

        if output_mode in {"spherical", "cartesian"}:
            self.target_dim = 7
        elif output_mode == "euler":
            self.target_dim = 6
        elif output_mode == "position_only":
            self.target_dim = 3
        else:
            raise ValueError(
                f"Unknown output_mode: {output_mode}. "
                "Supported: spherical, cartesian, euler, position_only"
            )

    def _pool_tokens_if_needed(self, scene_features: torch.Tensor) -> torch.Tensor:
        """Pool [B, S, P, D] to [B, S, D] when token dimension exists."""
        if scene_features.dim() == 4:
            if self.token_pooling_mode == "mean":
                return scene_features.mean(dim=2)
            if self.token_pooling_mode == "max":
                return scene_features.max(dim=2)[0]
            if self.token_pooling_mode == "camera":
                return scene_features[:, :, 0, :]
            raise ValueError(
                f"Unknown token_pooling_mode: {self.token_pooling_mode}. "
                "Supported: mean, max, camera"
            )
        return scene_features

    def _activate_nbv(self, nbv: torch.Tensor) -> torch.Tensor:
        """Apply output-space constraints."""
        if self.output_mode == "spherical":
            theta = torch.sigmoid(nbv[:, 0]) * 2 * math.pi
            phi = torch.sigmoid(nbv[:, 1]) * math.pi
            radius = torch.sigmoid(nbv[:, 2]) * 2 + 1
            position = torch.stack([theta, phi, radius], dim=1)
            quaternion = F.normalize(nbv[:, 3:], p=2, dim=1)
            return torch.cat([position, quaternion], dim=1)

        if self.output_mode == "cartesian":
            position = nbv[:, :3]
            quaternion = F.normalize(nbv[:, 3:], p=2, dim=1)
            return torch.cat([position, quaternion], dim=1)

        if self.output_mode == "euler":
            position = nbv[:, :3]
            roll = torch.tanh(nbv[:, 3]) * math.pi
            pitch = torch.tanh(nbv[:, 4]) * (math.pi / 2)
            yaw = torch.tanh(nbv[:, 5]) * math.pi
            rotation = torch.stack([roll, pitch, yaw], dim=1)
            return torch.cat([position, rotation], dim=1)

        # position_only
        return nbv[:, :3]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class FourierEmbedding(nn.Module):
    def __init__(self, input_dim: int, mapping_size: int = 64, scale: float = 10.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = float(scale)
        self.register_buffer("B", torch.randn(input_dim, mapping_size) * self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 0:
            return x

        x_proj = (2.0 * math.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class AttentionNBVPolicy(BaseNBVPolicy):
    """Attention-based NBV policy conditioned on scene features and camera extrinsics."""

    def __init__(
        self,
        scene_feature_dim: int = 768,
        hidden_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        output_mode: str = "cartesian",
        token_pooling_mode: str = "mean",
        input_extrinsic_dim: int = 7,
    ) -> None:
        super().__init__(output_mode, token_pooling_mode)
        self.hidden_dim = hidden_dim

        fourier_mapping_size = 64
        self.cam_fourier_embed = FourierEmbedding(input_extrinsic_dim, mapping_size=fourier_mapping_size)
        fourier_dim = fourier_mapping_size * 2

        self.camera_embedding = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.feature_projection = nn.Sequential(
            nn.Linear(scene_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=hidden_dim,
            dropout=dropout,
            max_len=5000,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.global_pool = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.target_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.global_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, scene_features: torch.Tensor, camera_extrinsics: torch.Tensor) -> torch.Tensor:
        scene_features = self._pool_tokens_if_needed(scene_features)
        batch_size, _, _ = scene_features.shape

        scene_tokens = self.feature_projection(scene_features)

        cam_fourier = self.cam_fourier_embed(camera_extrinsics)
        cam_emb = self.camera_embedding(cam_fourier)

        x = scene_tokens + cam_emb
        x = self.pos_encoder(x)
        encoded_features = self.transformer(x)

        global_token = self.global_token.expand(batch_size, -1, -1)
        global_features, _ = self.global_pool(
            query=global_token,
            key=encoded_features,
            value=encoded_features,
        )
        global_features = global_features.squeeze(1)

        nbv_raw = self.output_head(global_features)
        return self._activate_nbv(nbv_raw)


__all__ = ["BaseNBVPolicy", "AttentionNBVPolicy"]
