"""MapAnything implementation of SceneEncoderPort."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from nbv_framework.infrastructure.models.scene_encoder.mapanything_encoder import MapAnythingWrapper


class MapAnythingSceneEncoderAdapter:
    def __init__(self, wrapper: MapAnythingWrapper) -> None:
        self.wrapper = wrapper

    def extract_scene_features(
        self,
        images: torch.Tensor,
        camera_poses: torch.Tensor,
        *,
        depth_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        return self.wrapper.extract_scene_features(
            images,
            camera_poses,
            depth_z=depth_z,
            is_metric_scale=False,
        )
