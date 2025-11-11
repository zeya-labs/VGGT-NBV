"""Composite reconstruction loss that bundles geometric terms."""

import os
from typing import Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

import logging
import torch
import torch.nn as nn

from pytorch3d.structures import Pointclouds

from .chamfer import ChamferDistance
from .viewpoint import ViewpointLoss

if TYPE_CHECKING:
    from ..rendering import DifferentiableRenderer


class ReconstructionLoss(nn.Module):
    """Combine Chamfer, confidence, and viewpoint regularisers."""

    def __init__(
        self,
        chamfer_weight: float = 1.0,
        confidence_weight: float = 0.0,
        viewpoint_weight: float = 0.0,
        pose_penalty_weight: float = 0.02,
        renderer: Optional["DifferentiableRenderer"] = None,
        pose_up_axis: str = "Y",
        save_point_clouds: bool = True,
        point_cloud_dir_name: str = "point_clouds",
        max_points_per_cloud: int = 4096,
        log_tensorboard: bool = False,
    ) -> None:
        super().__init__()

        self.chamfer_weight = chamfer_weight
        self.confidence_weight = confidence_weight
        self.viewpoint_weight = viewpoint_weight
        self.pose_penalty_weight = pose_penalty_weight
        self.renderer = renderer
        self.train_flag = None
        self.pose_up_axis = pose_up_axis.upper()
        if self.pose_up_axis not in {"X", "Y", "Z"}:
            raise ValueError("pose_up_axis must be one of {'X', 'Y', 'Z'}.")

        self.save_point_clouds = bool(save_point_clouds)
        self.point_cloud_dir_name = point_cloud_dir_name
        self.chamfer_loss = ChamferDistance(max_points_per_cloud=max_points_per_cloud)
        self.chamfer_loss.configure_point_cloud_logging(
            enable_save=self.save_point_clouds,
            subdir_name=self.point_cloud_dir_name,
            max_points_per_cloud=max_points_per_cloud,
            log_to_tensorboard=log_tensorboard,
        )
        self.viewpoint_loss = ViewpointLoss()
        self.pose_outer_radius = 4.0
        self.pose_inner_radius = 2.0
        self.pose_floor_margin = 1.0

    @staticmethod
    def _infer_device(tensor_dict: Dict[str, torch.Tensor]) -> Optional[torch.device]:
        for value in tensor_dict.values():
            if isinstance(value, torch.Tensor):
                return value.device
        return None

    @staticmethod
    def _first_tensor(tensor_dict: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        for value in tensor_dict.values():
            if isinstance(value, torch.Tensor):
                return value
        return None

    def extract_point_cloud_from_reconstruction(
        self,
        recon_data: Dict[str, torch.Tensor],
        combined_images_batch: torch.Tensor,
        confidence_threshold: float = 0.0,
        source: Literal["vggt", "depth"] = "depth",
        gt_valid_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[Pointclouds, torch.Tensor]:
        if source == "vggt":
            points_data = recon_data.get("world_points")
            conf_data = recon_data.get("world_points_conf")
            if points_data is None or conf_data is None:
                raise KeyError(
                    "Source 'vggt' selected, but 'world_points' or 'world_points_conf' not found in recon_data."
                )
        elif source == "depth":
            points_data = recon_data.get("world_points_from_depth")
            conf_data = recon_data.get("depth_conf")
            if points_data is None or conf_data is None:
                raise KeyError(
                    "Source 'depth' selected, but 'world_points_from_depth' or 'depth_conf' not found in recon_data."
                )
        else:
            raise ValueError(f"未知的 source: {source}。应为 'vggt' 或 'depth'。")

        if points_data is None or conf_data is None:
            return Pointclouds(points=[])

        B, S, H, W, _ = points_data.shape

        with torch.no_grad():
            if confidence_threshold == 0.0:
                conf_threshold_value = 0.0
            else:
                conf_flat = conf_data.reshape(-1)
                conf_threshold_value = torch.quantile(
                    conf_flat, confidence_threshold / 100.0
                )

            high_conf_mask = (
                (conf_data >= conf_threshold_value) & (conf_data > 1e-5)
            )

            if combined_images_batch is not None:
                pixel_intensity = combined_images_batch.mean(dim=2)
                black_threshold = 0.1
                non_black_mask = pixel_intensity > black_threshold
                combined_mask = high_conf_mask & non_black_mask
            else:
                combined_mask = high_conf_mask

            if gt_valid_masks is not None:
                if gt_valid_masks.shape != combined_mask.shape:
                    raise ValueError(
                        "gt_valid_masks shape {gt_valid_masks.shape} does not match combined mask "
                        f"shape {combined_mask.shape}"
                    )
                combined_mask = combined_mask & gt_valid_masks

        point_clouds_list = []
        for i in range(B):
            mask_i = combined_mask[i]
            if mask_i.any():
                points_i = points_data[i][mask_i]
                point_clouds_list.append(points_i)
            else:
                point_clouds_list.append(
                    torch.empty((0, 3), device=points_data.device, dtype=points_data.dtype)
                )

        return Pointclouds(points=point_clouds_list), combined_mask

    def _compute_chamfer_component(
        self,
        recon_data: Dict[str, torch.Tensor],
        gt_data: Dict[str, torch.Tensor],
        combined_images_batch: Optional[torch.Tensor],
        combined_camera_poses: Optional[torch.Tensor],
        writer,
        step,
        device: torch.device,
        point_cloud_dir: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        zero = torch.tensor(0.0, device=device)
        if self.chamfer_weight <= 0 or "gt_points" not in gt_data:
            return zero, zero, None

        if combined_camera_poses is None:
            raise ValueError(
                "combined_camera_poses must be provided when Chamfer loss is enabled."
            )

        gt_point_maps = gt_data.get("gt_point_maps")
        gt_valid_masks = gt_data.get("gt_valid_masks")
        if gt_point_maps is None or gt_valid_masks is None:
            raise KeyError(
                "gt_mesh_data must contain 'gt_point_maps' and 'gt_valid_masks' for Chamfer loss."
            )

        sample_tensor = self._first_tensor(recon_data)
        if sample_tensor is None:
            raise ValueError("recon_data must contain tensor values for device inference.")

        target_device = sample_tensor.device
        gt_point_maps = gt_point_maps.to(device=target_device, dtype=torch.float32)
        gt_valid_masks = gt_valid_masks.to(device=target_device)

        pred_pointclouds, correspondence_mask = self.extract_point_cloud_from_reconstruction(
            recon_data,
            combined_images_batch,
            source="vggt",
            gt_valid_masks=gt_valid_masks,
        )

        gt_points_batch = gt_data["gt_points"]
        gt_pointclouds = Pointclouds(points=[p for p in gt_points_batch])

        correspondence_points: List[torch.Tensor] = []
        for i in range(correspondence_mask.shape[0]):
            mask_i = correspondence_mask[i]
            if mask_i.any():
                gt_points_i = gt_point_maps[i][mask_i]
            else:
                gt_points_i = torch.empty(
                    (0, 3),
                    device=gt_point_maps.device,
                    dtype=gt_point_maps.dtype,
                )
            correspondence_points.append(gt_points_i)

        if len(pred_pointclouds) != len(gt_pointclouds):
            logging.warning(
                "预测点云列表的批次大小与GT点云不匹配。跳过Chamfer损失计算。"
            )
            return zero, zero, correspondence_mask

        chamfer_loss_value = self.chamfer_loss(
            pred_pointclouds,
            gt_pointclouds,
            correspondence_points=correspondence_points,
            writer=writer,
            step=step,
            point_cloud_dir=point_cloud_dir,
        )

        weighted_loss = self.chamfer_weight * chamfer_loss_value
        return weighted_loss, chamfer_loss_value, correspondence_mask

    def _compute_confidence_component(
        self,
        recon_data: Dict[str, torch.Tensor],
        device: torch.device,
        confidence_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zero = torch.tensor(0.0, device=device)
        if self.confidence_weight <= 0:
            return zero, zero

        world_points_conf = recon_data.get("world_points_conf")
        depth_conf = recon_data.get("depth_conf")

        loss = zero
        if world_points_conf is not None:
            if confidence_mask is not None:
                if confidence_mask.shape != world_points_conf.shape:
                    raise ValueError(
                        "confidence_mask shape {confidence_mask.shape} does not match "
                        f"world_points_conf shape {world_points_conf.shape}"
                    )
                mask = confidence_mask.to(device=world_points_conf.device, dtype=world_points_conf.dtype)
                valid_count = mask.sum()
                if valid_count.item() > 0:
                    masked_mean = (world_points_conf * mask).sum() / valid_count
                else:
                    masked_mean = world_points_conf.new_tensor(1.0)
                loss = loss - torch.log(masked_mean + 1e-8)
            else:
                loss = loss - torch.log(world_points_conf.mean() + 1e-8)

        # if depth_conf is not None:
        #     loss = loss - torch.log(depth_conf.mean() + 1e-8)

        weighted_loss = self.confidence_weight * loss
        return weighted_loss, loss

    def _compute_viewpoint_component(
        self,
        combined_images_batch: Optional[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zero = torch.tensor(0.0, device=device)
        if self.viewpoint_weight <= 0 or combined_images_batch is None:
            return zero, zero

        new_images = combined_images_batch[:, -1, :, :, :]
        viewpoint_loss_value = self.viewpoint_loss(new_images)
        weighted_loss = self.viewpoint_weight * viewpoint_loss_value
        return weighted_loss, viewpoint_loss_value

    def _compute_pose_penalty_component(
        self,
        combined_camera_poses: Optional[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes a pose penalty based on the Euclidean distance from the origin.
        This constrains poses to a spherical shell.
        """
        zero = torch.tensor(0.0, device=device)
        pose_penalty_weight = getattr(self, "pose_penalty_weight", 0.0)
        if pose_penalty_weight <= 0 or combined_camera_poses is None:
            return zero, zero, {
                "pose_penalty_inner": zero,
                "pose_penalty_outer": zero,
                "pose_penalty_floor": zero,
            }

        # Get target positions (e.g., camera centers)
        if combined_camera_poses.dim() == 2:
            target_positions = combined_camera_poses[:, :3]
        else:
            target_positions = combined_camera_poses[:, -1, :3]

        axis_to_index = {"X": 0, "Y": 1, "Z": 2}
        up_axis_index = axis_to_index.get(getattr(self, "pose_up_axis", "Y"), 1)

        # Define the spherical shell boundaries
        outer_radius = float(getattr(self, "pose_outer_radius", 4.0))  # 最大允许半径
        inner_radius = float(getattr(self, "pose_inner_radius", 2.0))  # 最小允许半径
        floor_margin = float(getattr(self, "pose_floor_margin", 2.0))  # 允许的最低高度偏移

        inner_radius = max(inner_radius, 1e-3)
        outer_radius = max(outer_radius, 1e-3)
        floor_margin = max(floor_margin, 1e-3)

        # Calculate the Euclidean distance (L2 norm) from the origin for each pose
        # torch.linalg.norm is efficient for this
        distances = torch.linalg.norm(target_positions, ord=2, dim=-1)

        # Calculate violations using smooth quadratic penalties without singularities.
        # Inner shell: encourage cameras to stay outside the minimum radius.
        inner_violation = torch.relu(inner_radius - distances)
        inner_penalty = (inner_violation / inner_radius).pow(2).mean()

        # Outer shell: discourage cameras from drifting beyond the maximum radius.
        outer_violation = torch.relu(distances - outer_radius)
        outer_penalty = (outer_violation / outer_radius).pow(2).mean()

        # Floor constraint: gently push poses back above the configured floor plane.
        up_axis_values = target_positions[..., up_axis_index]
        floor_violation = torch.relu(-(up_axis_values + floor_margin))
        floor_penalty = (floor_violation / floor_margin).pow(2).mean()

        penalty_terms = {
            "pose_penalty_inner": inner_penalty,
            "pose_penalty_outer": outer_penalty,
            "pose_penalty_floor": floor_penalty,
        }
        # print("====================================================")
        # print("target_positions:",target_positions)
        # print("distances:",distances)

        # print("pose_penalty_inner:", inner_violation)
        # print("pose_penalty_outer:", outer_violation)
        # print("pose_penalty_floor:", floor_violation)

        # print("====================================================")
        penalty_value = torch.stack(list(penalty_terms.values())).sum()
        weighted_penalty = pose_penalty_weight * penalty_value

        return weighted_penalty, penalty_value, penalty_terms

    def forward(
        self,
        recon_data: Dict[str, torch.Tensor],
        gt_data: Dict[str, torch.Tensor],
        combined_images_batch: Optional[torch.Tensor],
        combined_camera_poses: Optional[torch.Tensor],
        return_components: bool = False,
        writer=None,
        step=None,
        train_flag: bool = False,
        point_cloud_dir: Optional[str] = None,
    ) -> torch.Tensor:
        self.train_flag = train_flag

        device = self._infer_device(recon_data)
        if device is None:
            if combined_images_batch is not None:
                device = combined_images_batch.device
            else:
                device = torch.device("cpu")

        total_loss = torch.tensor(0.0, device=device)
        loss_components: Dict[str, float] = {}

        chamfer_save_dir: Optional[str] = None
        if self.save_point_clouds and point_cloud_dir is not None:
            chamfer_save_dir = point_cloud_dir

        weighted_chamfer, chamfer_raw, confidence_mask = self._compute_chamfer_component(
            recon_data,
            gt_data,
            combined_images_batch,
            combined_camera_poses,
            writer,
            step,
            device,
            point_cloud_dir=chamfer_save_dir,
        )
        total_loss = total_loss + weighted_chamfer
        loss_components["chamfer_loss"] = chamfer_raw.item()
        loss_components["weighted_chamfer_loss"] = weighted_chamfer.item()

        weighted_confidence, confidence_raw = self._compute_confidence_component(
            recon_data,
            device,
            confidence_mask=confidence_mask,
        )
        total_loss = total_loss + weighted_confidence
        loss_components["confidence_loss"] = confidence_raw.item()
        loss_components["weighted_confidence_loss"] = weighted_confidence.item()

        weighted_viewpoint, viewpoint_raw = self._compute_viewpoint_component(
            combined_images_batch,
            device,
        )
        total_loss = total_loss + weighted_viewpoint
        loss_components["viewpoint_loss"] = viewpoint_raw.item()
        loss_components["weighted_viewpoint_loss"] = weighted_viewpoint.item()

        weighted_pose_penalty, pose_penalty_raw, pose_penalty_terms = self._compute_pose_penalty_component(
            combined_camera_poses,
            device,
        )
        total_loss = total_loss + weighted_pose_penalty
        loss_components["pose_penalty_loss"] = pose_penalty_raw.item()
        loss_components["weighted_pose_penalty_loss"] = weighted_pose_penalty.item()
        for term_name, term_value in pose_penalty_terms.items():
            loss_components[term_name] = term_value.item()

        loss_components["total_loss"] = total_loss.item()

        # print(loss_components)

        if return_components:
            return total_loss, loss_components
        return total_loss


__all__ = ["ReconstructionLoss"]
