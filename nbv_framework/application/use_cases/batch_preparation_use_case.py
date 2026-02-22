"""Batch preparation use case for NBV train/eval steps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from pytorch3d.structures import Meshes

from nbv_framework.application.dto import NBVBatch, PreparedBatch
from nbv_framework.domain.data.batch_utils import parse_mesh_metadata, trim_gt_mesh_data
from nbv_framework.domain.data.view_selection import select_initial_views
from nbv_framework.application.ports import DepthVisualizationPort, MeshRepositoryPort, RendererPort


class BatchPreparationUseCase:
    """Prepare heterogeneous dataset batches into tensor-ready structures."""

    REQUIRED_GT_KEYS = ("gt_point_maps", "gt_valid_masks", "depth_z", "depth_z_viz")

    def __init__(
        self,
        *,
        renderer: RendererPort,
        mesh_repository: MeshRepositoryPort,
        depth_visualizer: DepthVisualizationPort,
        mesh_load_workers: int,
        min_initial_views: int,
        max_initial_views: int,
        randomize_initial_views: bool,
    ) -> None:
        self.renderer = renderer
        self.mesh_repository = mesh_repository
        self.depth_visualizer = depth_visualizer
        self.mesh_load_workers = int(mesh_load_workers)
        self.min_initial_views = int(min_initial_views)
        self.max_initial_views = int(max_initial_views)
        self.randomize_initial_views = bool(randomize_initial_views)

    def parse_batch(self, batch: Dict[str, Any]) -> NBVBatch:
        inputs = batch.get("inputs", {})
        targets = batch.get("targets", {})
        mesh_data = batch.get("mesh", {})
        meta = batch.get("meta")

        initial_images = inputs.get("images")
        camera_poses = inputs.get("camera_poses")
        if camera_poses is None:
            raise ValueError("Batch inputs.camera_poses is required.")

        gt_mesh_data = targets.get("gt_mesh_data", {})
        mesh_paths, normalize_methods = parse_mesh_metadata(meta)
        mesh_batch = mesh_data.get("normalized")
        if isinstance(mesh_batch, list):
            mesh_batch = None

        return NBVBatch(
            initial_images=initial_images,
            camera_poses=camera_poses,
            gt_mesh_data=gt_mesh_data,
            mesh_batch=mesh_batch,
            mesh_paths=mesh_paths,
            normalize_methods=normalize_methods,
            meta=meta,
        )

    def prepare(self, batch: Dict[str, Any], *, randomize: bool) -> PreparedBatch:
        parsed = self.parse_batch(batch)
        camera_poses_batch = parsed.camera_poses

        mesh_batch = parsed.mesh_batch
        if mesh_batch is None:
            mesh_batch = self.mesh_repository.load_meshes_as_batch(
                mesh_paths=parsed.mesh_paths,
                normalize_methods=parsed.normalize_methods,
                device=camera_poses_batch.device,
                num_workers=self.mesh_load_workers,
            )

        gt_mesh_data = dict(parsed.gt_mesh_data)
        initial_images = parsed.initial_images

        batch_size = camera_poses_batch.shape[0]
        missing_indices = self._find_missing_indices(initial_images, gt_mesh_data, batch_size)

        if missing_indices:
            initial_images, gt_mesh_data = self._render_missing(
                initial_images=initial_images,
                gt_mesh_data=gt_mesh_data,
                camera_poses_batch=camera_poses_batch,
                mesh_batch=mesh_batch,
                missing_indices=missing_indices,
            )
        else:
            initial_images = self._ensure_batch_tensor(initial_images, batch_size)
            for key in self.REQUIRED_GT_KEYS:
                gt_mesh_data[key] = self._ensure_batch_tensor(gt_mesh_data.get(key), batch_size)

        depth_z_batch = gt_mesh_data.get("depth_z")

        (
            initial_images,
            camera_poses_batch,
            depth_z_batch,
            selection,
            active_view_count,
        ) = select_initial_views(
            initial_images,
            camera_poses_batch,
            depth_z=depth_z_batch,
            randomize=randomize,
            min_initial_views=self.min_initial_views,
            max_initial_views=self.max_initial_views,
            randomize_initial_views=self.randomize_initial_views,
        )

        trimmed_gt_mesh_data = trim_gt_mesh_data(gt_mesh_data, selection)

        return PreparedBatch(
            initial_images=initial_images,
            camera_poses=camera_poses_batch,
            depth_z=depth_z_batch,
            gt_mesh_data=gt_mesh_data,
            trimmed_gt_mesh_data=trimmed_gt_mesh_data,
            mesh_batch=mesh_batch,
            mesh_paths=parsed.mesh_paths,
            selection=selection,
            active_view_count=active_view_count,
        )

    def _find_missing_indices(
        self,
        initial_images: Optional[torch.Tensor],
        gt_mesh_data: Dict[str, Any],
        batch_size: int,
    ) -> List[int]:
        cache_ready = [True] * batch_size
        if initial_images is None:
            cache_ready = [False] * batch_size
        elif isinstance(initial_images, list):
            for idx, item in enumerate(initial_images):
                if item is None:
                    cache_ready[idx] = False

        for key in self.REQUIRED_GT_KEYS:
            value = gt_mesh_data.get(key)
            if value is None:
                return list(range(batch_size))
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    if item is None:
                        cache_ready[idx] = False
            elif torch.is_tensor(value):
                if value.shape[0] != batch_size:
                    return list(range(batch_size))

        return [idx for idx, ready in enumerate(cache_ready) if not ready]

    def _ensure_batch_tensor(self, value: Any, batch_size: int) -> torch.Tensor:
        if value is None:
            raise ValueError("Missing required batch tensor value.")
        if torch.is_tensor(value):
            return value
        if isinstance(value, list):
            return torch.stack(value, dim=0)
        return torch.stack([value for _ in range(batch_size)], dim=0)

    def _as_list(self, value: Any, batch_size: int) -> List[Any]:
        if value is None:
            return [None] * batch_size
        if isinstance(value, list):
            return list(value)
        if torch.is_tensor(value) and value.shape[0] == batch_size:
            return list(value.unbind(0))
        return [value] * batch_size

    def _render_missing(
        self,
        *,
        initial_images: Optional[torch.Tensor],
        gt_mesh_data: Dict[str, Any],
        camera_poses_batch: torch.Tensor,
        mesh_batch: "Meshes",
        missing_indices: List[int],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = camera_poses_batch.shape[0]
        idx_tensor = torch.as_tensor(missing_indices, device=camera_poses_batch.device, dtype=torch.long)

        subset_mesh_batch = mesh_batch[missing_indices]
        subset_camera_poses = camera_poses_batch.index_select(0, idx_tensor)

        subset_render_out = self.renderer.render_views(
            mesh_batch=subset_mesh_batch,
            camera_poses=subset_camera_poses,
            out_rgb=True,
            out_points=True,
            out_mask=True,
            out_depth=True,
        )

        subset_images = subset_render_out["rgb"]
        if subset_images.is_floating_point() and subset_images.dtype != torch.float32:
            subset_images = subset_images.to(dtype=torch.float32)

        subset_points = subset_render_out["points"]
        subset_masks = subset_render_out["mask"].to(dtype=torch.bool)
        subset_depth = subset_render_out["depth"]
        subset_depth_viz = self.depth_visualizer.normalize_depth_for_visualization(
            subset_depth,
            subset_masks,
        )

        initial_images_list = self._as_list(initial_images, batch_size)
        points_list = self._as_list(gt_mesh_data.get("gt_point_maps"), batch_size)
        masks_list = self._as_list(gt_mesh_data.get("gt_valid_masks"), batch_size)
        depth_list = self._as_list(gt_mesh_data.get("depth_z"), batch_size)
        depth_viz_list = self._as_list(gt_mesh_data.get("depth_z_viz"), batch_size)

        for offset, idx in enumerate(missing_indices):
            initial_images_list[idx] = subset_images[offset]
            points_list[idx] = subset_points[offset]
            masks_list[idx] = subset_masks[offset]
            depth_list[idx] = subset_depth[offset]
            depth_viz_list[idx] = subset_depth_viz[offset]

        gt_mesh_data["gt_point_maps"] = torch.stack(points_list, dim=0)
        gt_mesh_data["gt_valid_masks"] = torch.stack(masks_list, dim=0)
        gt_mesh_data["depth_z"] = torch.stack(depth_list, dim=0)
        gt_mesh_data["depth_z_viz"] = torch.stack(depth_viz_list, dim=0)
        return torch.stack(initial_images_list, dim=0), gt_mesh_data

