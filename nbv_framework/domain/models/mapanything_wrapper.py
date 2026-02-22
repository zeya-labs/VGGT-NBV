"""MapAnything 基础模型封装.

该包装器将预训练的 MapAnything 模型集成到 NBV 框架中, 提供:
1. 场景编码特征抽取接口, 返回每视角的 Transformer token 特征
2. 基于图像的 3D 几何推理接口, 输出与原 VGGTWrapper 相同关键字段

当前实现默认启用图像 + 标定(内参/位姿)多模态输入, 暂不使用深度先验.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from loguru import logger

from mapanything.models import MapAnything
from uniception.models.info_sharing.base import MultiViewTransformerInput

from nbv_framework.infrastructure.utils.mapanything_views import (
    dump_mapanything_views_for_debug,
    prepare_mapanything_views,
)

TensorDict = Dict[str, torch.Tensor]
PredList = List[TensorDict]


class MapAnythingWrapper(nn.Module):
    """MapAnything 模型封装.

    参数被冻结, 仅用于特征抽取与几何推理; 梯度仅需对输入图像开放, 以支撑策略网络训练.
    """

    def __init__(
        self,
        model_name: str = "facebook/map-anything",
        revision: Optional[str] = "6f3a25bfbb8fcc799176bb01e9d07dfb49d5416a",
        local_files_only: bool = True,
        data_norm_type: str = "dinov2",
        memory_efficient_inference: bool = False,
    ) -> None:
        super().__init__()
        self.data_norm_type = data_norm_type
        self.memory_efficient_inference = memory_efficient_inference

        logger.info(f"Loading MapAnything model: {model_name}")
        from_pretrained_kwargs = {"local_files_only": bool(local_files_only)}
        if revision is not None:
            from_pretrained_kwargs["revision"] = revision
        self.base_model: MapAnything = MapAnything.from_pretrained(
            model_name,
            **from_pretrained_kwargs,
        )
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
        logger.info("MapAnything model loaded and frozen successfully")

        self.default_fov_degrees: float = 60.0

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def extract_scene_features(
        self,
        images: torch.Tensor,
        camera_poses: torch.Tensor,
        *,
        depth_z: Optional[torch.Tensor] = None,
        is_metric_scale: bool = False,
        fov_degrees: Optional[float] = None,
        view_save_dir: Optional[str] = None,
        mesh_paths: Optional[Sequence[Optional[str]]] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """提取多视角场景特征, 返回形状 [B, S, P, D]."""
        effective_fov = self.default_fov_degrees if fov_degrees is None else fov_degrees
        views, normalized = prepare_mapanything_views(
            images,
            camera_poses,
            data_norm_type=self.data_norm_type,
            fov_degrees=effective_fov,
            is_metric_scale=is_metric_scale,
            depth_z=depth_z,
        )
        if view_save_dir is not None:
            dump_mapanything_views_for_debug(
                images=images,
                camera_poses=camera_poses,
                fov_degrees=effective_fov,
                is_metric_scale=is_metric_scale,
                save_dir=view_save_dir,
                depth_z=depth_z,
                mesh_paths=mesh_paths,
            )
        batch_size = normalized.shape[0]
        self._configure_geometric_inputs(
            use_calibration=True,
            use_pose=True,
            use_depth=depth_z is not None,
        )
        try:
            encoder_features = self.base_model._encode_n_views(views)
            with torch.autocast("cuda", enabled=False):
                fused_features = self.base_model._encode_and_fuse_optional_geometric_inputs(
                    views, encoder_features
                )
        finally:
            self._restore_geometric_inputs()
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
        # [Slist][B,C,Hf,Wf]
        scene_features = self._gather_tokens(final_feat.features) # [B, S, P=Hf*Wf, C]
        # print("scene_features shape:", scene_features.shape)
        return scene_features, views

    def reconstruct_and_evaluate(
        self,
        images: torch.Tensor,
        camera_poses: torch.Tensor,
        *,
        depth_z: Optional[torch.Tensor] = None,
        is_metric_scale: bool = False,
        fov_degrees: Optional[float] = None,
        view_save_dir: Optional[str] = None,
        mesh_paths: Optional[Sequence[Optional[str]]] = None,
    ) -> TensorDict:
        """运行 MapAnything 前向, 返回与 VGGTWrapper 对齐的关键输出."""
        effective_fov = self.default_fov_degrees if fov_degrees is None else fov_degrees
        views, normalized = prepare_mapanything_views(
            images,
            camera_poses,
            data_norm_type=self.data_norm_type,
            fov_degrees=effective_fov,
            is_metric_scale=is_metric_scale,
            depth_z=depth_z,
        )
        if view_save_dir is not None:
            dump_mapanything_views_for_debug(
                images=images,
                camera_poses=camera_poses,
                fov_degrees=effective_fov,
                is_metric_scale=is_metric_scale,
                save_dir=view_save_dir,
                depth_z=depth_z,
                mesh_paths=mesh_paths,
            )
        self._configure_geometric_inputs(
            use_calibration=True,
            use_pose=True,
            use_depth=depth_z is not None,
        )
        # 输出views(list)的键
        # print("views keys:", views[0].keys())

        try:
            predictions = self.base_model.forward(
                views, memory_efficient_inference=self.memory_efficient_inference
            )
        finally:
            self._restore_geometric_inputs()
        # 列出 predictions 中的所有键
        # print("predictions keys:", predictions[0].keys())
        # predictions keys: dict_keys(['pts3d', 'pts3d_cam', 'ray_directions', 'depth_along_ray', 'cam_trans', 'cam_quats', 'metric_scaling_factor', 'conf', 'non_ambiguous_mask', 'non_ambiguous_mask_logits'])
        recon = self._stack_predictions(predictions)
        return recon

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
        return result

    def forward(
        self,
        images: torch.Tensor,
        camera_poses: torch.Tensor,
        *,
        mode: str = "encode",
        depth_z: Optional[torch.Tensor] = None,
        is_metric_scale: bool = False,
        fov_degrees: Optional[float] = None,
        view_save_dir: Optional[str] = None,
        mesh_paths: Optional[Sequence[Optional[str]]] = None,
    ) -> Union[torch.Tensor, TensorDict]:
        if mode == "encode":
            return self.extract_scene_features(
                images,
                camera_poses,
                depth_z=depth_z,
                is_metric_scale=is_metric_scale,
                fov_degrees=fov_degrees,
                view_save_dir=view_save_dir,
                mesh_paths=mesh_paths,
            )
        if mode == "reconstruct":
            return self.reconstruct_and_evaluate(
                images,
                camera_poses,
                depth_z=depth_z,
                is_metric_scale=is_metric_scale,
                fov_degrees=fov_degrees,
                view_save_dir=view_save_dir,
                mesh_paths=mesh_paths,
            )
        raise ValueError(f"Unknown mode: {mode}. Supported modes: encode, reconstruct")

    # ------------------------------------------------------------------
    # 内部工具函数
    # ------------------------------------------------------------------
    def _configure_geometric_inputs(
        self,
        use_calibration: bool,
        use_pose: bool,
        use_depth: bool = False,
        use_depth_scale: bool = False,
        use_pose_scale: bool = False,
    ) -> None:
        if hasattr(self.base_model, "_configure_geometric_input_config"):
            self.base_model._configure_geometric_input_config(
                use_calibration=use_calibration,
                use_depth=use_depth,
                use_pose=use_pose,
                use_depth_scale=use_depth_scale,
                use_pose_scale=use_pose_scale,
            )

    def _restore_geometric_inputs(self) -> None:
        if hasattr(self.base_model, "_restore_original_geometric_input_config"):
            self.base_model._restore_original_geometric_input_config()


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

__all__ = ["MapAnythingWrapper"]
