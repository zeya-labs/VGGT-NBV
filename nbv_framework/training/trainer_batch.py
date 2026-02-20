"""Batch preparation helpers for NBVTrainer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from pytorch3d.structures import Meshes

from ..data.batch_utils import parse_mesh_metadata, trim_gt_mesh_data
from ..pipeline.step_ops import render_inputs, select_initial_views
from ..pipeline.types import PreparedBatch
from ..utils.mesh_utils import load_meshes_as_batch


class NBVTrainerBatchMixin:
    """Batch preparation utilities for NBVTrainer."""

    def _render_inputs(
        self,
        initial_images: Optional[torch.Tensor],
        camera_poses_batch: torch.Tensor,
        gt_mesh_data: Dict[str, torch.Tensor],
        mesh_batch: Optional[Meshes],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return render_inputs(
            renderer=self.renderer,
            initial_images=initial_images,
            camera_poses_batch=camera_poses_batch,
            gt_mesh_data=gt_mesh_data,
            mesh_batch=mesh_batch,
            dtype=self.dtype,
        )

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> PreparedBatch:
        inputs = batch.get("inputs", {})
        targets = batch.get("targets", {})
        mesh_data = batch.get("mesh", {})
        meta = batch.get("meta")

        initial_images = inputs.get("images")
        camera_poses_batch = inputs.get("camera_poses")

        gt_mesh_data = targets.get("gt_mesh_data", {})
        mesh_paths, normalize_methods = parse_mesh_metadata(meta)
        cache_paths = None
        if self.render_cache is not None:
            cache_paths = self.render_cache.build_paths(
                mesh_paths=mesh_paths,
                normalize_methods=normalize_methods,
                camera_poses_batch=camera_poses_batch,
            )

        mesh_batch = mesh_data.get("normalized")
        if isinstance(mesh_batch, list):
            mesh_batch = None

        if mesh_batch is None:
            mesh_batch = load_meshes_as_batch(
                mesh_paths=mesh_paths,
                normalize_methods=normalize_methods,
                device=camera_poses_batch.device,
                num_workers=self.mesh_load_workers,
            )

        batch_size = camera_poses_batch.shape[0]
        required_keys = ("gt_point_maps", "gt_valid_masks", "depth_z", "depth_z_viz")

        def _as_list(value: Any) -> List[Any]:
            if value is None:
                return [None] * batch_size
            if isinstance(value, list):
                return list(value)
            if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                return list(value.unbind(0))
            return [value] * batch_size

        cache_ready = [True] * batch_size
        if initial_images is None:
            cache_ready = [False] * batch_size
        elif isinstance(initial_images, list):
            for idx, item in enumerate(initial_images):
                if item is None:
                    cache_ready[idx] = False

        for key in required_keys:
            value = gt_mesh_data.get(key)
            if value is None:
                cache_ready = [False] * batch_size
                break
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    if item is None:
                        cache_ready[idx] = False
            elif isinstance(value, torch.Tensor):
                if value.shape[0] != batch_size:
                    cache_ready = [False] * batch_size
                    break

        missing_indices = [idx for idx, ready in enumerate(cache_ready) if not ready]
        rendered = False
        if missing_indices:
            idx_tensor = torch.as_tensor(
                missing_indices, device=camera_poses_batch.device, dtype=torch.long
            )
            subset_mesh_batch = mesh_batch[missing_indices]
            subset_camera_poses = camera_poses_batch.index_select(0, idx_tensor)

            subset_gt_mesh_data: Dict[str, Any] = {}
            for key, value in gt_mesh_data.items():
                if key in required_keys:
                    continue
                if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                    subset_gt_mesh_data[key] = value.index_select(0, idx_tensor)
                elif isinstance(value, list):
                    subset_gt_mesh_data[key] = [value[i] for i in missing_indices]
                else:
                    subset_gt_mesh_data[key] = value

            subset_initial_images, subset_gt_mesh_data = self._render_inputs(
                initial_images=None,
                camera_poses_batch=subset_camera_poses,
                gt_mesh_data=subset_gt_mesh_data,
                mesh_batch=subset_mesh_batch,
            )
            rendered = True

            initial_images_list = _as_list(initial_images)
            for offset, idx in enumerate(missing_indices):
                initial_images_list[idx] = subset_initial_images[offset]
            initial_images = torch.stack(initial_images_list, dim=0)

            for key in required_keys:
                existing_list = _as_list(gt_mesh_data.get(key))
                subset_value = subset_gt_mesh_data.get(key)
                for offset, idx in enumerate(missing_indices):
                    existing_list[idx] = subset_value[offset]
                gt_mesh_data[key] = torch.stack(existing_list, dim=0)
        else:
            if isinstance(initial_images, list):
                initial_images = torch.stack(initial_images, dim=0)
            for key in required_keys:
                value = gt_mesh_data.get(key)
                if isinstance(value, list):
                    gt_mesh_data[key] = torch.stack(value, dim=0)

        if rendered and cache_paths and self.render_cache is not None:
            self.render_cache.save_batch(
                cache_paths=cache_paths,
                mesh_batch=mesh_batch,
                initial_images=initial_images,
                gt_mesh_data=gt_mesh_data,
                is_global_zero=getattr(self.trainer, "is_global_zero", True),
            )
        depth_z_batch = gt_mesh_data.get("depth_z")

        initial_images, camera_poses_batch, depth_z_batch, selection, active_view_count = self._select_initial_views(
            initial_images,
            camera_poses_batch,
            depth_z=depth_z_batch,
            randomize=self.trainer.training,
        )

        trimmed_gt_mesh_data = trim_gt_mesh_data(gt_mesh_data, selection)

        return PreparedBatch(
            initial_images=initial_images,
            camera_poses=camera_poses_batch,
            depth_z=None,
            gt_mesh_data=gt_mesh_data,
            trimmed_gt_mesh_data=trimmed_gt_mesh_data,
            mesh_batch=mesh_batch,
            mesh_paths=mesh_paths,
            selection=selection,
            active_view_count=active_view_count,
        )

    def _select_initial_views(
        self,
        initial_images: torch.Tensor,
        camera_poses: torch.Tensor,
        *,
        depth_z: Optional[torch.Tensor] = None,
        randomize: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, int]:
        initial_images, camera_poses, depth_z, selection, num_views = select_initial_views(
            initial_images,
            camera_poses,
            depth_z=depth_z,
            randomize=randomize,
            min_initial_views=self.min_initial_views,
            max_initial_views=self.max_initial_views,
            randomize_initial_views=self.randomize_initial_views,
        )
        self._last_initial_view_count = num_views
        self._last_initial_view_indices = selection.detach().cpu()
        return initial_images, camera_poses, depth_z, selection, num_views

    def _get_log_batch_size(self) -> Optional[int]:
        if self._last_batch_size is None:
            return None
        return int(self._last_batch_size)
