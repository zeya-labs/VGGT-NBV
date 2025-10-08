"""MapAnything 基础模型封装.

该包装器将预训练的 MapAnything 模型集成到 NBV 框架中, 提供:
1. 场景编码特征抽取接口, 返回每视角的 Transformer token 特征
2. 基于图像的 3D 几何推理接口, 输出与原 VGGTWrapper 相同关键字段

当前实现仅启用图像输入路径, 不使用外部相机或深度先验.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tvf
from PIL import Image

# 将 map-anything 仓库加入 Python 搜索路径
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MAP_ANYTHING_ROOT = os.path.join(_REPO_ROOT, "map-anything")
if _MAP_ANYTHING_ROOT not in sys.path:
    sys.path.append(_MAP_ANYTHING_ROOT)

try:  # noqa: SIM105
    from mapanything.models import MapAnything  # type: ignore
    from mapanything.utils.image import IMAGE_NORMALIZATION_DICT, find_closest_aspect_ratio  # type: ignore
    from mapanything.utils.cropping import crop_resize_if_necessary  # type: ignore
    from uniception.models.info_sharing.base import MultiViewTransformerInput  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - 运行时缺依赖由用户环境负责
    missing = "uniception" if "uniception" in str(exc) else "mapanything"
    raise ModuleNotFoundError(
        f"无法导入 {missing} 模块, 请确认 map-anything 及其依赖已正确安装"
    ) from exc


_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

TensorDict = Dict[str, torch.Tensor]
PredList = List[TensorDict]


class MapAnythingWrapper(nn.Module):
    """MapAnything 模型封装.

    参数被冻结, 仅用于特征抽取与几何推理; 梯度仅需对输入图像开放, 以支撑策略网络训练.
    """

    def __init__(
        self,
        model_name: str = "facebook/map-anything",
        device: str = "cuda",
        data_norm_type: str = "dinov2",
        memory_efficient_inference: bool = False,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.data_norm_type = data_norm_type
        self.memory_efficient_inference = memory_efficient_inference

        print(f"Loading MapAnything model: {model_name}")
        self.base_model: MapAnything = MapAnything.from_pretrained(model_name).to(self.device)
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
        print("MapAnything model loaded and frozen successfully")

        encoder_dim = getattr(self.base_model.encoder, "enc_embed_dim", None)
        self._default_feature_dim: Optional[int] = int(encoder_dim) if encoder_dim is not None else None
        self._feature_dim: Optional[int] = None

        self.resolution_set: int = 518

        self._capture_gradients: bool = False
        self._grad_capture_keys: Tuple[str, ...] = tuple()
        self._captured_tensors: Dict[str, torch.Tensor] = {}
        
        self._capture_input_grad: bool = False
        self._captured_input: Optional[torch.Tensor] = None

        self._disable_geometric_inputs()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def configure_gradient_capture(
        self,
        enable: bool = True,
        keys: Optional[Sequence[str]] = None,
        capture_input: bool = True,
    ) -> None:
        """保留接口以兼容原训练器, 当前实现仅记录配置状态."""
        self._capture_gradients = enable
        self._capture_input_grad = capture_input
        if keys is not None:
            self._grad_capture_keys = tuple(keys)
        if not enable:
            self._captured_tensors.clear()
            self._captured_input = None

    def collect_gradient_stats(self) -> Dict[str, float]:
        """返回最近一次记录的梯度统计. 当前 MapAnything 仅作为冻结特征提取器, 直接返回空字典."""
        if not self._capture_gradients:
            return {}
        grad_stats: Dict[str, float] = {}
        if self._capture_input_grad and self._captured_input is not None:
            grad = self._captured_input.grad
            if grad is not None:
                grad_stats["input/grad_norm"] = grad.norm().detach().item()
                grad_stats["input/grad_mean_abs"] = grad.abs().mean().detach().item()
        for key, tensor in list(self._captured_tensors.items()):
            grad = tensor.grad
            if grad is None:
                continue
            grad_stats[f"{key}/grad_norm"] = grad.norm().detach().item()
            grad_stats[f"{key}/grad_mean_abs"] = grad.abs().mean().detach().item()
        self._captured_tensors.clear()
        self._captured_input = None
        return grad_stats

    def extract_scene_features(self, images: torch.Tensor) -> torch.Tensor:
        """提取多视角场景特征, 返回形状 [B, S, P, D]."""
        views, normalized = self._prepare_views(images)
        batch_size = normalized.shape[0]
        # print("views shape:", views[0]["img"].shape)
        encoder_features = self.base_model._encode_n_views(views)
        # print("encoder_features shape:", [i.shape for i in encoder_features])
        with torch.autocast("cuda", enabled=False):
            fused_features = self.base_model._encode_and_fuse_optional_geometric_inputs(
                views, encoder_features
            )
        scale_token = (
            self.base_model.scale_token.unsqueeze(0)
            .unsqueeze(-1)
            .repeat(batch_size, 1, 1)
        )
        
        # print("fused_features shape:", [i.shape for i in fused_features])
        # print("scale_token shape:", scale_token.shape)
        info_sharing_input = MultiViewTransformerInput(
            features=fused_features, additional_input_tokens=scale_token
        )
        if self.base_model.info_sharing_return_type == "no_intermediate_features":
            final_feat = self.base_model.info_sharing(info_sharing_input)
        else:
            final_feat, _ = self.base_model.info_sharing(info_sharing_input)
        # print("info_sharing features shape:", [i.shape for i in final_feat.features])
        scene_features = self._gather_tokens(final_feat.features) # [B, S, P, D]
        self._feature_dim = scene_features.shape[-1]
        self._maybe_retain_grad(scene_features, normalized)
        # print("scene_features shape:", scene_features.shape)
        return scene_features

    def reconstruct_and_evaluate(self, images: torch.Tensor) -> TensorDict:
        """运行 MapAnything 前向, 返回与 VGGTWrapper 对齐的关键输出."""
        views, normalized = self._prepare_views(images)
        predictions = self.base_model.forward(
            views, memory_efficient_inference=self.memory_efficient_inference
        )
        recon = self._stack_predictions(predictions)
        self._maybe_retain_grad_from_result(recon, normalized)
        return recon

    def forward(
        self, images: torch.Tensor, mode: str = "encode"
    ) -> Union[torch.Tensor, TensorDict]:
        if mode == "encode":
            return self.extract_scene_features(images)
        if mode == "reconstruct":
            return self.reconstruct_and_evaluate(images)
        raise ValueError(f"Unknown mode: {mode}. Supported modes: encode, reconstruct")

    # ------------------------------------------------------------------
    # 内部工具函数
    # ------------------------------------------------------------------
    def _disable_geometric_inputs(self) -> None:
        if hasattr(self.base_model, "_configure_geometric_input_config"):
            self.base_model._configure_geometric_input_config(
                use_calibration=False,
                use_depth=False,
                use_pose=False,
                use_depth_scale=False,
                use_pose_scale=False,
            )

    def _ensure_batch(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 4:
            return images.unsqueeze(0)
        if images.dim() != 5:
            raise ValueError(
                f"Expected images with shape [B, S, 3, H, W] or [S, 3, H, W], got {images.shape}"
            )
        return images

    def _prepare_views(self, images: torch.Tensor) -> Tuple[List[TensorDict], torch.Tensor]:
        images = self._ensure_batch(images)
        images = images.clamp(0.0, 1.0)

        batch_size, num_views, _, _, _ = images.shape
        is_metric = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # 计算所有视图的平均长宽比
        aspect_ratios: List[float] = []
        for b in range(batch_size):
            for v in range(num_views):
                img = images[b, v]
                h, w = int(img.shape[-2]), int(img.shape[-1])
                if h > 0 and w > 0:
                    aspect_ratios.append(w / h)

        if not aspect_ratios:
            raise ValueError("无法从输入图像中计算长宽比")

        average_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
        # target_width, target_height = find_closest_aspect_ratio(average_aspect_ratio, self.resolution_set)
        target_width, target_height = (224,224)

        patch_size = getattr(self.base_model.encoder, "patch_size", 14)
        if patch_size <= 0:
            patch_size = 14
        target_width = max(patch_size, (target_width // patch_size) * patch_size)
        target_height = max(patch_size, (target_height // patch_size) * patch_size)

        norm_cfg = IMAGE_NORMALIZATION_DICT.get(self.data_norm_type)
        if norm_cfg is None:
            mean = _DEFAULT_MEAN
            std = _DEFAULT_STD
        else:
            mean = norm_cfg.mean
            std = norm_cfg.std

        ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=mean, std=std)])

        views: List[TensorDict] = []
        for view_idx in range(num_views):
            normalized_stack: List[torch.Tensor] = []
            for batch_idx in range(batch_size):
                tensor = images[batch_idx, view_idx].detach().cpu()
                np_img = tensor.permute(1, 2, 0).numpy()
                pil_img = Image.fromarray((np_img * 255.0).clip(0, 255).astype(np.uint8))

                resized_img = crop_resize_if_necessary(
                    image=pil_img,
                    resolution=(target_width, target_height),
                )[0]

                norm_img = ImgNorm(resized_img)
                normalized_stack.append(norm_img)

            view_tensor = torch.stack(normalized_stack, dim=0).to(self.device)
            views.append(
                {
                    "img": view_tensor,
                    "data_norm_type": [self.data_norm_type],
                    "is_metric_scale": is_metric.clone(),
                }
            )

        normalized = torch.stack(
            [view["img"] for view in views], dim=1
        )  # [B, S, 3, H, W]

        return views, normalized

    def infer_feature_dim(self, image_size: int, num_views: int) -> int:
        """使用零样本推理确认当前配置下的特征维度."""
        if self._feature_dim is not None:
            return self._feature_dim

        patch_size = getattr(self.base_model.encoder, "patch_size", 14)
        if patch_size <= 0:
            patch_size = 14
        height = max(patch_size, (image_size // patch_size) * patch_size)
        width = height
        dummy = torch.zeros(
            1,
            max(1, num_views),
            3,
            height,
            width,
            device=self.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            _ = self.extract_scene_features(dummy)
        if self._feature_dim is None:
            raise RuntimeError("无法推断 MapAnything 特征维度")
        return self._feature_dim

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is not None:
            return self._feature_dim
        if self._default_feature_dim is not None:
            return self._default_feature_dim
        raise RuntimeError("MapAnythingWrapper 尚未确定特征维度")

    def _gather_tokens(self, feature_list: Sequence[torch.Tensor]) -> torch.Tensor:
        if not feature_list:
            raise ValueError("MapAnything 返回空特征列表")

        processed: List[torch.Tensor] = []
        for feat in feature_list:
            if feat.dim() != 4:
                raise ValueError(
                    f"期望特征维度为 [B, C, H, W], 实际收到 {feat.shape}"
                )
            tokens = feat.flatten(2).transpose(1, 2)
            processed.append(tokens)

        return torch.stack(processed, dim=1)

    def _stack_predictions(self, predictions: PredList) -> TensorDict:
        stacked: Dict[str, List[torch.Tensor]] = {}
        for view_pred in predictions:
            for key, value in view_pred.items():
                if not torch.is_tensor(value):
                    continue
                stacked.setdefault(key, []).append(value)

        result: TensorDict = {}
        for key, values in stacked.items():
            tensor = torch.stack(values, dim=1)
            result[key] = tensor

        if "pts3d" in result:
            pts3d = result["pts3d"]
            result["world_points_from_depth"] = pts3d
            result["world_points"] = pts3d
        if "conf" in result:
            conf = result["conf"]
            result["depth_conf"] = conf
            result["world_points_conf"] = conf
        if "pts3d_cam" in result and "depth" not in result:
            result["depth"] = result["pts3d_cam"][..., 2:3]
        elif "depth_along_ray" in result and "depth" not in result:
            result["depth"] = result["depth_along_ray"]
        return result

    def _maybe_retain_grad(self, scene_features: torch.Tensor, images: torch.Tensor) -> None:
        if not self._capture_gradients:
            return
        for key in self._grad_capture_keys:
            if key == "scene_features":
                target = scene_features
            else:
                continue
            if target.requires_grad:
                target.retain_grad()
                self._captured_tensors[key] = target
        if self._capture_input_grad and images.requires_grad:
            images.retain_grad()
            self._captured_input = images

    def _maybe_retain_grad_from_result(
        self, result: TensorDict, images: torch.Tensor
    ) -> None:
        if not self._capture_gradients:
            return
        for key in self._grad_capture_keys:
            tensor = result.get(key)
            if tensor is None or not torch.is_tensor(tensor):
                continue
            if tensor.requires_grad:
                tensor.retain_grad()
                self._captured_tensors[key] = tensor
        if self._capture_input_grad and images.requires_grad:
            images.retain_grad()
            self._captured_input = images


__all__ = ["MapAnythingWrapper"]
