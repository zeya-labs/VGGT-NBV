"""Depth Anything 3 scene-encoder and reconstruction wrapper."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from loguru import logger
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.utils.camera_conversions import opencv_from_cameras_projection
from safetensors.torch import load_file

from nbv_framework.reconstruction import ReconstructionData

TensorDict = Dict[str, torch.Tensor]
ViewList = List[Dict[str, Any]]

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def _ensure_addict_compat() -> None:
    try:
        import addict  # noqa: F401

        return
    except ImportError:
        pass

    class _CompatDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.update(*args, **kwargs)

        def __getattr__(self, key: str) -> Any:
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key: str, value: Any) -> None:
            if key.startswith("_"):
                object.__setattr__(self, key, value)
                return
            self[key] = value

        def __delattr__(self, key: str) -> None:
            try:
                del self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setitem__(self, key: str, value: Any) -> None:
            super().__setitem__(key, self._wrap(value))

        def update(self, *args, **kwargs) -> None:  # type: ignore[override]
            data = dict(*args, **kwargs)
            for key, value in data.items():
                self[key] = value

        @classmethod
        def _wrap(cls, value: Any) -> Any:
            if isinstance(value, cls):
                return value
            if isinstance(value, dict):
                return cls(value)
            if isinstance(value, list):
                return [cls._wrap(item) for item in value]
            if isinstance(value, tuple):
                return tuple(cls._wrap(item) for item in value)
            return value

    module = types.ModuleType("addict")
    module.Dict = _CompatDict
    sys.modules["addict"] = module


def _ensure_depthanything3_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    src_dir = repo_root / "third_party" / "Depth-Anything-3" / "src"
    if not src_dir.exists():
        raise FileNotFoundError(
            f"Depth Anything 3 source not found: {src_dir}. "
            "Expected third_party/Depth-Anything-3/src to exist."
        )
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


def _ensure_depthanything3_runtime() -> None:
    _ensure_addict_compat()
    _ensure_depthanything3_import_path()


def _resolve_pretrained_artifact(
    model_name_or_path: str,
    filename: str,
    *,
    revision: Optional[str],
    local_files_only: bool,
) -> Path:
    path = Path(model_name_or_path).expanduser()
    if path.exists():
        if path.is_dir():
            artifact = path / filename
        else:
            artifact = path
        if not artifact.exists():
            raise FileNotFoundError(f"Expected DA3 artifact `{filename}` under {path}")
        return artifact.resolve()

    downloaded = hf_hub_download(
        repo_id=model_name_or_path,
        filename=filename,
        revision=revision,
        local_files_only=bool(local_files_only),
    )
    return Path(downloaded).resolve()


def _load_pretrained_config(
    model_name_or_path: str,
    *,
    revision: Optional[str],
    local_files_only: bool,
) -> Dict[str, Any]:
    config_path = _resolve_pretrained_artifact(
        model_name_or_path,
        "config.json",
        revision=revision,
        local_files_only=local_files_only,
    )
    return json.loads(config_path.read_text(encoding="utf-8"))


def _instantiate_da3_model(config_payload: Dict[str, Any]) -> nn.Module:
    _ensure_depthanything3_runtime()
    from depth_anything_3.cfg import create_object
    from omegaconf import OmegaConf

    config = config_payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("Depth Anything 3 config.json missing top-level `config` object")
    return create_object(OmegaConf.create(config))


def _find_first_nested_value(obj: Any, *path: str) -> Any:
    if not path:
        return obj
    if isinstance(obj, dict):
        head = path[0]
        if head in obj:
            value = _find_first_nested_value(obj[head], *path[1:])
            if value is not None:
                return value
        for value in obj.values():
            result = _find_first_nested_value(value, *path)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for value in obj:
            result = _find_first_nested_value(value, *path)
            if result is not None:
                return result
    return None


def _extract_scene_feature_dim(config_payload: Dict[str, Any]) -> int:
    cam_dim = _find_first_nested_value(config_payload, "cam_enc", "dim_out")
    if cam_dim is not None:
        return int(cam_dim)

    head_dim = _find_first_nested_value(config_payload, "head", "dim_in")
    cat_token = _find_first_nested_value(config_payload, "net", "cat_token")
    if head_dim is not None:
        if bool(cat_token):
            return int(head_dim) // 2
        return int(head_dim)

    raise ValueError("Unable to infer DA3 scene feature dim from config.json")


def _extract_default_feature_layer(config_payload: Dict[str, Any]) -> int:
    out_layers = _find_first_nested_value(config_payload, "net", "out_layers")
    if not isinstance(out_layers, list) or not out_layers:
        raise ValueError("Unable to infer DA3 export feature layer from config.json")
    return int(max(out_layers))


def _normalize_images(images: torch.Tensor) -> torch.Tensor:
    mean = _IMAGENET_MEAN.to(device=images.device, dtype=images.dtype).view(1, 1, 3, 1, 1)
    std = _IMAGENET_STD.to(device=images.device, dtype=images.dtype).view(1, 1, 3, 1, 1)
    return (images - mean) / std


def _compute_pinhole_intrinsics(
    *,
    height: int,
    width: int,
    fov_degrees: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    fov_radians = torch.deg2rad(torch.tensor(float(fov_degrees), device=device, dtype=dtype))
    fy = 0.5 * float(height) / torch.tan(fov_radians / 2.0)
    fx = 0.5 * float(width) / torch.tan(fov_radians / 2.0)
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    return torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )


def _pose7d_to_opencv_cam2world(
    pose: torch.Tensor,
    *,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    if pose.dim() != 2 or pose.shape[-1] != 7:
        raise ValueError(f"Expected flattened pose shape [N, 7], got {tuple(pose.shape)}")

    position_c2w = pose[:, :3]
    quaternion_xyzw = pose[:, 3:]
    quaternion_wxyz = quaternion_xyzw[:, [3, 0, 1, 2]]

    rotation_w2c = quaternion_to_matrix(quaternion_wxyz)
    translation_w2c = -torch.bmm(position_c2w.unsqueeze(1), rotation_w2c).squeeze(1)

    cameras = PerspectiveCameras(R=rotation_w2c, T=translation_w2c, device=pose.device)
    image_size_tensor = torch.as_tensor(
        image_size,
        device=pose.device,
        dtype=pose.dtype,
    ).view(1, 2)
    image_size_tensor = image_size_tensor.expand(pose.shape[0], -1)
    rotation_w2c_cv, translation_w2c_cv, _ = opencv_from_cameras_projection(
        cameras,
        image_size_tensor,
    )

    rotation_c2w_cv = rotation_w2c_cv.transpose(1, 2)
    position_c2w_cv = -torch.bmm(
        rotation_c2w_cv,
        translation_w2c_cv.unsqueeze(-1),
    ).squeeze(-1)

    cam2world = torch.eye(4, device=pose.device, dtype=pose.dtype).unsqueeze(0).repeat(
        pose.shape[0], 1, 1
    )
    cam2world[:, :3, :3] = rotation_c2w_cv
    cam2world[:, :3, 3] = position_c2w_cv
    return cam2world


def _affine_inverse(matrix: torch.Tensor) -> torch.Tensor:
    rotation = matrix[..., :3, :3]
    translation = matrix[..., :3, 3:]
    bottom = matrix[..., 3:, :]
    return torch.cat(
        [
            torch.cat([rotation.transpose(-1, -2), -rotation.transpose(-1, -2) @ translation], dim=-1),
            bottom,
        ],
        dim=-2,
    )


def _normalize_extrinsics_to_ref0(
    extrinsics_w2c: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if extrinsics_w2c.dim() != 4 or extrinsics_w2c.shape[-2:] != (4, 4):
        raise ValueError(
            f"extrinsics_w2c must have shape [B, S, 4, 4], got {tuple(extrinsics_w2c.shape)}"
        )

    ref0_c2w = _affine_inverse(extrinsics_w2c[:, :1])
    normalized = extrinsics_w2c @ ref0_c2w
    c2w_normalized = _affine_inverse(normalized)
    distances = c2w_normalized[..., :3, 3].norm(dim=-1)
    median_distance = torch.median(distances, dim=1).values.clamp_min(1e-1)
    normalized = normalized.clone()
    normalized[..., :3, 3] = normalized[..., :3, 3] / median_distance.view(-1, 1, 1)
    return normalized, ref0_c2w.squeeze(1), median_distance


def _backproject_depth_to_world(
    *,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics_w2c: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if depth.dim() != 4:
        raise ValueError(f"depth must have shape [B, S, H, W], got {tuple(depth.shape)}")

    batch_size, num_views, height, width = depth.shape
    device = depth.device
    dtype = depth.dtype

    u = torch.arange(width, device=device, dtype=dtype)
    v = torch.arange(height, device=device, dtype=dtype)
    try:
        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    except TypeError:  # pragma: no cover
        v_grid, u_grid = torch.meshgrid(v, u)

    pixels = torch.stack([u_grid, v_grid, torch.ones_like(u_grid)], dim=-1)
    pixels = pixels.view(1, 1, height * width, 3).expand(batch_size, num_views, -1, -1)

    k_inv = torch.linalg.inv(intrinsics)
    rays = torch.matmul(k_inv.unsqueeze(2), pixels.unsqueeze(-1)).squeeze(-1)
    depth_flat = depth.view(batch_size, num_views, height * width, 1)
    cam_points = rays * depth_flat

    c2w = _affine_inverse(extrinsics_w2c)
    rotation = c2w[..., :3, :3]
    translation = c2w[..., :3, 3]

    world_points = torch.matmul(
        rotation.unsqueeze(2),
        cam_points.unsqueeze(-1),
    ).squeeze(-1) + translation.unsqueeze(2)
    world_points = world_points.view(batch_size, num_views, height, width, 3)

    mask = torch.isfinite(depth) & (depth > 0)
    world_points = world_points.masked_fill(~mask.unsqueeze(-1), 0.0)
    return world_points, mask


def _transform_normalized_points_to_world(
    points_ref0_scaled: torch.Tensor,
    *,
    ref0_c2w: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    batch_size = points_ref0_scaled.shape[0]
    scaled = points_ref0_scaled * scale.view(batch_size, 1, 1, 1, 1)
    flat = scaled.view(batch_size, -1, 3)
    rotation = ref0_c2w[:, :3, :3]
    translation = ref0_c2w[:, :3, 3]
    transformed = torch.einsum("bij,bnj->bni", rotation, flat) + translation.unsqueeze(1)
    return transformed.view_as(scaled)


def _camera_centers_from_extrinsics(extrinsics_w2c: torch.Tensor) -> torch.Tensor:
    if extrinsics_w2c.dim() != 4:
        raise ValueError(
            f"extrinsics_w2c must have shape [B, S, 3|4, 4], got {tuple(extrinsics_w2c.shape)}"
        )
    if extrinsics_w2c.shape[-2:] == (3, 4):
        pad = torch.zeros(
            *extrinsics_w2c.shape[:-2],
            4,
            4,
            device=extrinsics_w2c.device,
            dtype=extrinsics_w2c.dtype,
        )
        pad[..., :3, :4] = extrinsics_w2c
        pad[..., 3, 3] = 1.0
        extrinsics_w2c = pad
    elif extrinsics_w2c.shape[-2:] != (4, 4):
        raise ValueError(
            f"extrinsics_w2c must have shape [B, S, 3|4, 4], got {tuple(extrinsics_w2c.shape)}"
        )
    c2w = _affine_inverse(extrinsics_w2c)
    return c2w[..., :3, 3]


def _estimate_umeyama_scale(
    reference_extrinsics_w2c: torch.Tensor,
    input_extrinsics_w2c: torch.Tensor,
) -> torch.Tensor:
    if reference_extrinsics_w2c.dim() != 4 or input_extrinsics_w2c.dim() != 4:
        raise ValueError(
            "reference_extrinsics_w2c and input_extrinsics_w2c must be rank-4 tensors: "
            f"{tuple(reference_extrinsics_w2c.shape)} vs {tuple(input_extrinsics_w2c.shape)}"
        )
    if reference_extrinsics_w2c.shape[:2] != input_extrinsics_w2c.shape[:2]:
        raise ValueError(
            "reference_extrinsics_w2c and input_extrinsics_w2c must share batch/view dims: "
            f"{tuple(reference_extrinsics_w2c.shape[:2])} vs {tuple(input_extrinsics_w2c.shape[:2])}"
        )

    reference_centers = _camera_centers_from_extrinsics(reference_extrinsics_w2c)
    input_centers = _camera_centers_from_extrinsics(input_extrinsics_w2c)

    ref_mean = reference_centers.mean(dim=1, keepdim=True)
    input_mean = input_centers.mean(dim=1, keepdim=True)
    ref_centered = reference_centers - ref_mean
    input_centered = input_centers - input_mean

    num_views = reference_centers.shape[1]
    covariance = torch.matmul(ref_centered.transpose(-1, -2), input_centered) / float(num_views)
    u, singular_values, vh = torch.linalg.svd(covariance)

    sign = torch.ones(
        reference_centers.shape[0],
        3,
        device=reference_centers.device,
        dtype=reference_centers.dtype,
    )
    det = torch.linalg.det(torch.matmul(u, vh))
    sign[:, -1] = torch.where(det < 0, -1.0, 1.0)

    input_variance = (input_centered.square().sum(dim=(-1, -2)) / float(num_views)).clamp_min(1e-12)
    scale = (singular_values * sign).sum(dim=-1) / input_variance
    return scale.clamp_min(1e-6)


def _build_policy_views(images: torch.Tensor, camera_poses: torch.Tensor) -> ViewList:
    return [
        {
            "img": images[:, view_idx],
            "camera_pose_trans": camera_poses[:, view_idx, :3],
            "camera_pose_quats": camera_poses[:, view_idx, 3:],
        }
        for view_idx in range(images.shape[1])
    ]


class _CheckpointContainer(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model


class DepthAnything3Wrapper(nn.Module):
    """Depth Anything 3 wrapper matching the MapAnything wrapper interface."""

    def __init__(
        self,
        model_name_or_path: str = "depth-anything/DA3-BASE",
        revision: Optional[str] = None,
        local_files_only: bool = False,
        feature_layer: Optional[int] = None,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> None:
        super().__init__()
        _ensure_depthanything3_runtime()

        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.local_files_only = bool(local_files_only)
        self.use_ray_pose = bool(use_ray_pose)
        self.ref_view_strategy = str(ref_view_strategy)
        self.default_fov_degrees = 60.0
        self._depth_input_warned = False

        config_payload = _load_pretrained_config(
            model_name_or_path,
            revision=revision,
            local_files_only=self.local_files_only,
        )
        self.model_name = str(config_payload.get("model_name", model_name_or_path))
        self.scene_feature_dim = _extract_scene_feature_dim(config_payload)
        self.feature_layer = (
            _extract_default_feature_layer(config_payload)
            if feature_layer is None
            else int(feature_layer)
        )

        logger.info(f"Loading Depth Anything 3 model: {model_name_or_path}")
        raw_model = _instantiate_da3_model(config_payload)
        state_dict_path = _resolve_pretrained_artifact(
            model_name_or_path,
            "model.safetensors",
            revision=revision,
            local_files_only=self.local_files_only,
        )
        state_dict = load_file(str(state_dict_path), device="cpu")

        if any(key.startswith("model.") for key in state_dict.keys()):
            checkpoint_model: nn.Module = _CheckpointContainer(raw_model)
        else:
            checkpoint_model = raw_model

        missing, unexpected = checkpoint_model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Depth Anything 3 missing keys: {missing}")
        if unexpected:
            logger.warning(f"Depth Anything 3 unexpected keys: {unexpected}")

        self.base_model = raw_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
        logger.info("Depth Anything 3 model loaded and frozen successfully")

    def extract_scene_features(
        self,
        images: torch.Tensor,
        camera_poses: torch.Tensor,
        *,
        depth_z: Optional[torch.Tensor] = None,
        is_metric_scale: bool = True,
        fov_degrees: Optional[float] = None,
        view_save_dir: Optional[str] = None,
        mesh_paths: Optional[Sequence[Optional[str]]] = None,
    ) -> Tuple[torch.Tensor, ViewList]:
        del is_metric_scale, view_save_dir, mesh_paths
        if depth_z is not None and not self._depth_input_warned:
            logger.warning("DepthAnything3Wrapper ignores depth_z input during feature extraction")
            self._depth_input_warned = True

        normalized_images, extrinsics_norm, intrinsics_norm, _, _, _ = self._prepare_inputs(
            images=images,
            camera_poses=camera_poses,
            fov_degrees=fov_degrees,
        )

        output = self.base_model(
            normalized_images,
            extrinsics_norm,
            intrinsics_norm,
            export_feat_layers=[self.feature_layer],
            infer_gs=False,
            use_ray_pose=self.use_ray_pose,
            ref_view_strategy=self.ref_view_strategy,
        )
        aux_key = f"feat_layer_{self.feature_layer}"
        aux = output.get("aux")
        if aux is None or aux_key not in aux:
            raise KeyError(f"Depth Anything 3 output missing auxiliary feature `{aux_key}`")

        feature_map = aux[aux_key]
        if feature_map.dim() != 5:
            raise ValueError(
                f"Expected DA3 auxiliary feature shape [B, S, Hf, Wf, C], got {tuple(feature_map.shape)}"
            )
        batch_size, num_views, feat_h, feat_w, feat_dim = feature_map.shape
        scene_features = feature_map.reshape(batch_size, num_views, feat_h * feat_w, feat_dim)
        return scene_features, _build_policy_views(images, camera_poses)

    def reconstruct_and_evaluate(
        self,
        images: torch.Tensor,
        camera_poses: torch.Tensor,
        *,
        depth_z: Optional[torch.Tensor] = None,
        is_metric_scale: bool = True,
        fov_degrees: Optional[float] = None,
        view_save_dir: Optional[str] = None,
        mesh_paths: Optional[Sequence[Optional[str]]] = None,
        align_pts3d_to_input_world: bool = True,
    ) -> ReconstructionData:
        del is_metric_scale, view_save_dir, mesh_paths
        if depth_z is not None and not self._depth_input_warned:
            logger.warning("DepthAnything3Wrapper ignores depth_z input during reconstruction")
            self._depth_input_warned = True

        (
            normalized_images,
            extrinsics_norm,
            intrinsics_norm,
            ref0_c2w,
            scale,
            extrinsics_input,
        ) = self._prepare_inputs(
            images=images,
            camera_poses=camera_poses,
            fov_degrees=fov_degrees,
        )

        output = self.base_model(
            normalized_images,
            extrinsics_norm,
            intrinsics_norm,
            export_feat_layers=[],
            infer_gs=False,
            use_ray_pose=self.use_ray_pose,
            ref_view_strategy=self.ref_view_strategy,
        )

        depth = output.get("depth")
        if depth is None or not torch.is_tensor(depth):
            raise KeyError("Depth Anything 3 output missing tensor `depth`")
        depth = depth.to(dtype=normalized_images.dtype)

        backproject_extrinsics = extrinsics_norm
        if align_pts3d_to_input_world:
            predicted_extrinsics = output.get("extrinsics")
            if torch.is_tensor(predicted_extrinsics):
                umeyama_scale = _estimate_umeyama_scale(predicted_extrinsics, extrinsics_input)
                depth = depth / umeyama_scale.view(-1, 1, 1, 1)
                backproject_extrinsics = extrinsics_input
            else:
                logger.warning(
                    "Depth Anything 3 output missing `extrinsics`; falling back to normalized-pose "
                    "backprojection without official pose-scale alignment"
                )

        world_points, mask = _backproject_depth_to_world(
            depth=depth,
            intrinsics=intrinsics_norm,
            extrinsics_w2c=backproject_extrinsics,
        )
        if align_pts3d_to_input_world and backproject_extrinsics is extrinsics_norm:
            world_points = _transform_normalized_points_to_world(
                world_points,
                ref0_c2w=ref0_c2w,
                scale=scale,
            )

        conf = output.get("depth_conf")
        if conf is None:
            conf = mask.to(dtype=world_points.dtype)
        else:
            conf = conf.to(device=world_points.device, dtype=world_points.dtype)

        mask = mask.to(device=world_points.device, dtype=torch.bool)
        conf = conf.masked_fill(~mask, 0.0)
        world_points = world_points.masked_fill(~mask.unsqueeze(-1), 0.0)

        return ReconstructionData(
            recon_world_points=world_points,
            recon_conf=conf,
            recon_mask=mask,
        )

    def forward(
        self,
        images: torch.Tensor,
        camera_poses: torch.Tensor,
        *,
        mode: str = "encode",
        depth_z: Optional[torch.Tensor] = None,
        is_metric_scale: bool = True,
        fov_degrees: Optional[float] = None,
        view_save_dir: Optional[str] = None,
        mesh_paths: Optional[Sequence[Optional[str]]] = None,
        align_pts3d_to_input_world: bool = True,
    ) -> Union[Tuple[torch.Tensor, ViewList], ReconstructionData]:
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
                align_pts3d_to_input_world=align_pts3d_to_input_world,
            )
        raise ValueError(f"Unknown mode: {mode}. Supported modes: encode, reconstruct")

    def _prepare_inputs(
        self,
        *,
        images: torch.Tensor,
        camera_poses: torch.Tensor,
        fov_degrees: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if images.dim() != 5 or images.shape[2] != 3:
            raise ValueError(f"images must have shape [B, S, 3, H, W], got {tuple(images.shape)}")
        if camera_poses.dim() != 3 or camera_poses.shape[-1] != 7:
            raise ValueError(
                f"camera_poses must have shape [B, S, 7], got {tuple(camera_poses.shape)}"
            )
        if camera_poses.shape[:2] != images.shape[:2]:
            raise ValueError(
                "camera_poses batch/view dims must match images: "
                f"{tuple(camera_poses.shape[:2])} vs {tuple(images.shape[:2])}"
            )

        batch_size, num_views, _, height, width = images.shape
        effective_fov = self.default_fov_degrees if fov_degrees is None else float(fov_degrees)
        normalized_images = _normalize_images(images.to(dtype=torch.float32))

        intrinsics_single = _compute_pinhole_intrinsics(
            height=height,
            width=width,
            fov_degrees=effective_fov,
            device=images.device,
            dtype=torch.float32,
        )
        intrinsics = intrinsics_single.view(1, 1, 3, 3).repeat(batch_size, num_views, 1, 1)

        pose_flat = camera_poses.reshape(batch_size * num_views, 7).to(dtype=torch.float32)
        cam2world = _pose7d_to_opencv_cam2world(pose_flat, image_size=(height, width))
        extrinsics_input = _affine_inverse(cam2world).view(batch_size, num_views, 4, 4)
        extrinsics_norm, ref0_c2w, scale = _normalize_extrinsics_to_ref0(extrinsics_input)

        return normalized_images, extrinsics_norm, intrinsics, ref0_c2w, scale, extrinsics_input


__all__ = ["DepthAnything3Wrapper"]
