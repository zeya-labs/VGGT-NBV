from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class MeshListItem(BaseModel):
    id: str
    relative_path: str
    name: str


class ViewerCamera(BaseModel):
    position: List[float] = Field(..., min_length=3, max_length=3)
    target: List[float] = Field(..., min_length=3, max_length=3)


class MeshInfoResponse(BaseModel):
    mesh_path: str
    centroid: List[float]
    scale: float


class CaptureImageRequest(BaseModel):
    mesh_path: str
    camera: ViewerCamera
    image_size: int = Field(default=512, ge=64, le=2048)
    fov: float = Field(default=60.0, ge=15.0, le=120.0)

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, value: int) -> int:
        if value % 2 != 0:
            raise ValueError("image_size must be an even integer.")
        return value


class RenderVideoRequest(BaseModel):
    mesh_path: str
    camera: ViewerCamera
    trajectory_mode: Literal["orbit", "swing"] = "orbit"
    duration_sec: float = Field(default=4.0, ge=1.0, le=30.0)
    fps: int = Field(default=24, ge=4, le=60)
    image_size: int = Field(default=512, ge=64, le=2048)
    fov: float = Field(default=60.0, ge=15.0, le=120.0)

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, value: int) -> int:
        if value % 2 != 0:
            raise ValueError("image_size must be an even integer.")
        return value


class HistoryRecord(BaseModel):
    record_id: str
    kind: Literal["image", "video"]
    mesh_path: str
    created_at: str
    preview_url: Optional[str] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    metadata_url: str

