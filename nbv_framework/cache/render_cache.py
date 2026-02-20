from __future__ import annotations

import hashlib
from loguru import logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from pytorch3d.structures import Meshes, join_meshes_as_batch




class RenderCache:
    def __init__(
        self,
        *,
        renderer: Optional[Any] = None,
        root: Optional[Path] = None,
        version: int = 1,
        render_signature: Optional[str] = None,
    ) -> None:
        self.renderer = renderer
        self.version = int(version)
        self.root = root or self._default_root()
        self._render_signature_override = render_signature

    @staticmethod
    def build_signature(
        *,
        version: int,
        image_size: Optional[int],
        fov: Optional[float],
        faces_per_pixel: Optional[int],
        blur_radius: Optional[float],
        perspective_correct: Optional[bool],
        cull_backfaces: Optional[bool],
    ) -> str:
        return (
            f"v{int(version)}|"
            f"img{image_size}|"
            f"fov{fov}|"
            f"fpp{faces_per_pixel}|"
            f"blur{blur_radius}|"
            f"pc{perspective_correct}|"
            f"cull{cull_backfaces}"
        )

    def _default_root(self) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "models" / ".cache"

    def _render_signature(self) -> str:
        if self._render_signature_override is not None:
            return self._render_signature_override
        if self.renderer is None:
            raise ValueError("RenderCache requires a renderer or an explicit render_signature.")
        raster_settings = self.renderer.rasterizer.raster_settings
        return self.build_signature(
            version=self.version,
            image_size=getattr(self.renderer, "image_size", None),
            fov=getattr(self.renderer, "default_fov", None),
            faces_per_pixel=raster_settings.faces_per_pixel,
            blur_radius=raster_settings.blur_radius,
            perspective_correct=raster_settings.perspective_correct,
            cull_backfaces=raster_settings.cull_backfaces,
        )

    def _cache_key(
        self,
        *,
        mesh_path: str,
        normalize_method: Optional[str],
        camera_poses: torch.Tensor,
    ) -> str:
        hasher = hashlib.sha1()
        hasher.update(str(mesh_path).encode("utf-8"))
        hasher.update(b"|")
        hasher.update(str(normalize_method).encode("utf-8"))
        hasher.update(b"|")
        hasher.update(self._render_signature().encode("utf-8"))
        hasher.update(b"|")
        cam_bytes = (
            camera_poses.detach()
            .to(dtype=torch.float32)
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        )
        hasher.update(cam_bytes)
        return hasher.hexdigest()

    def build_paths(
        self,
        *,
        mesh_paths: Optional[Sequence[Optional[str]]],
        normalize_methods: Optional[Sequence[Optional[str]]],
        camera_poses_batch: Optional[torch.Tensor],
    ) -> Optional[List[Path]]:
        if mesh_paths is None or camera_poses_batch is None:
            return None
        if len(mesh_paths) != camera_poses_batch.shape[0]:
            return None
        cache_paths: List[Path] = []
        for idx, mesh_path in enumerate(mesh_paths):
            if not mesh_path:
                return None
            normalize_method = None
            if normalize_methods is not None and idx < len(normalize_methods):
                normalize_method = normalize_methods[idx]
            key = self._cache_key(
                mesh_path=mesh_path,
                normalize_method=normalize_method,
                camera_poses=camera_poses_batch[idx],
            )
            cache_paths.append(self.root / f"{key}.pt")
        return cache_paths

    def _load_item(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.is_file():
            return None
        try:
            item = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            logger.warning("Failed to load render cache: {}", path)
            return None
        if not isinstance(item, dict):
            return None
        if item.get("version") != self.version:
            return None
        if "mesh" not in item or "initial_images" not in item or "gt_mesh_data" not in item:
            return None
        if not isinstance(item["mesh"], Meshes):
            return None
        gt_mesh_data = item.get("gt_mesh_data", {})
        required_keys = ("gt_point_maps", "gt_valid_masks", "depth_z", "depth_z_viz")
        if not all(key in gt_mesh_data for key in required_keys):
            return None
        return item

    def load_batch(
        self,
        *,
        cache_paths: Sequence[Path],
        device: torch.device,
        dtype: torch.dtype,
        base_gt_mesh_data: Dict[str, torch.Tensor],
    ) -> Optional[Tuple[Meshes, torch.Tensor, Dict[str, torch.Tensor]]]:
        cache_items: List[Dict[str, Any]] = []
        for path in cache_paths:
            item = self._load_item(path)
            if item is None:
                return None
            cache_items.append(item)

        meshes = [item["mesh"].to(device) for item in cache_items]
        mesh_batch = join_meshes_as_batch(meshes)

        initial_images = torch.stack(
            [item["initial_images"] for item in cache_items], dim=0
        ).to(device)
        if initial_images.is_floating_point() and initial_images.dtype != dtype:
            initial_images = initial_images.to(dtype=dtype)

        cached_gt = [item["gt_mesh_data"] for item in cache_items]
        gt_mesh_data = dict(base_gt_mesh_data)
        gt_mesh_data["gt_point_maps"] = torch.stack(
            [item["gt_point_maps"] for item in cached_gt], dim=0
        ).to(device)
        gt_mesh_data["gt_valid_masks"] = torch.stack(
            [item["gt_valid_masks"] for item in cached_gt], dim=0
        ).to(device, dtype=torch.bool)
        gt_mesh_data["depth_z"] = torch.stack(
            [item["depth_z"] for item in cached_gt], dim=0
        ).to(device)
        gt_mesh_data["depth_z_viz"] = torch.stack(
            [item["depth_z_viz"] for item in cached_gt], dim=0
        ).to(device)

        return mesh_batch, initial_images, gt_mesh_data

    def load_item(
        self,
        *,
        cache_path: Path,
        base_gt_mesh_data: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[Tuple[Meshes, torch.Tensor, Dict[str, torch.Tensor]]]:
        item = self._load_item(cache_path)
        if item is None:
            return None

        target_device = device if device is not None else torch.device("cpu")
        mesh = item["mesh"].to(target_device)
        initial_images = item["initial_images"].to(target_device)
        if dtype is not None and initial_images.is_floating_point() and initial_images.dtype != dtype:
            initial_images = initial_images.to(dtype=dtype)

        cached_gt = item["gt_mesh_data"]
        gt_mesh_data = dict(base_gt_mesh_data)
        gt_mesh_data["gt_point_maps"] = cached_gt["gt_point_maps"].to(target_device)
        gt_mesh_data["gt_valid_masks"] = cached_gt["gt_valid_masks"].to(target_device, dtype=torch.bool)
        gt_mesh_data["depth_z"] = cached_gt["depth_z"].to(target_device)
        gt_mesh_data["depth_z_viz"] = cached_gt["depth_z_viz"].to(target_device)

        return mesh, initial_images, gt_mesh_data

    def save_batch(
        self,
        *,
        cache_paths: Sequence[Path],
        mesh_batch: Meshes,
        initial_images: Optional[torch.Tensor],
        gt_mesh_data: Dict[str, torch.Tensor],
        is_global_zero: bool = True,
    ) -> None:
        if initial_images is None:
            return
        required_keys = ("gt_point_maps", "gt_valid_masks", "depth_z", "depth_z_viz")
        if any(gt_mesh_data.get(key) is None for key in required_keys):
            return
        if not is_global_zero:
            return

        self.root.mkdir(parents=True, exist_ok=True)
        for idx, path in enumerate(cache_paths):
            if path.exists():
                continue
            item = {
                "version": self.version,
                "mesh": mesh_batch[idx].to("cpu"),
                "initial_images": initial_images[idx].detach().to("cpu"),
                "gt_mesh_data": {
                    "gt_point_maps": gt_mesh_data["gt_point_maps"][idx].detach().to("cpu"),
                    "gt_valid_masks": gt_mesh_data["gt_valid_masks"][idx].detach().to("cpu"),
                    "depth_z": gt_mesh_data["depth_z"][idx].detach().to("cpu"),
                    "depth_z_viz": gt_mesh_data["depth_z_viz"][idx].detach().to("cpu"),
                },
            }
            tmp_path = path.with_suffix(".tmp")
            torch.save(item, tmp_path)
            tmp_path.replace(path)
