"""Request/response models for the MapAnything reconstruction WebUI backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SamplingConfig(BaseModel):
    view_sampling_mode: str = "deterministic_per_call"
    seed: int = 42
    camera_radius: float = 1.6
    camera_radius_variation: float = 0.0
    camera_radius_mode: str = "random"
    up_axis: str = "Y"
    scene_index: int = 0
    use_manual_camera: bool = False
    manual_camera_position: Optional[Any] = None
    manual_camera_look_at: Optional[Any] = None


class RenderConfig(BaseModel):
    image_size: int = Field(default=518, ge=64, le=2048)
    fov: float = Field(default=60.0, ge=10.0, le=150.0)
    normalize_method: str = "unit_sphere"
    num_samples: int = Field(default=32768, ge=256, le=2000000)


class PrepareInputsRequest(BaseModel):
    mesh_path: str
    num_views: int = Field(default=2, ge=1, le=32)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    show_depth: bool = False


class PrepareInputsResponse(BaseModel):
    run_id: str
    mesh_path: str
    model_name: str
    num_views: int
    rgb_urls: List[str]
    depth_urls: List[str]
    camera_poses: List[List[float]]
    created_at: str
    device: str
    timings: Dict[str, float]


class ReconstructRequest(BaseModel):
    run_id: str
    conf_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    max_points: int = Field(default=300000, ge=1000, le=2000000)
    use_depth_input: bool = True


class ReconstructResponse(BaseModel):
    run_id: str
    ply_url: str
    num_points: int
    num_points_before_sampling: int
    used_depth_input: bool
    created_at: str
    timings: Dict[str, float]


class RunRecord(BaseModel):
    run_id: str
    created_at: str
    mesh_path: str
    model_name: str
    num_views: int
    device: str
    rgb_urls: List[str]
    depth_urls: List[str]
    camera_poses: List[List[float]]
    reconstruct: Optional[Dict[str, Any]] = None
