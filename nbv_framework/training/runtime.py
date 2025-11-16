"""High-level orchestration of NBV experiments."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from nbv_framework import (
    AttentionNBVPolicy,
    BaseNBVPolicy,
    DifferentiableRenderer,
    MapAnythingWrapper,
    NBVTrainer,
)
from nbv_framework.datasets import MixedDataset, RepeatedDataset, SyntheticDataset
from nbv_framework.datasets.data_loaders import create_train_loader, create_val_loader
from nbv_framework.training.config import NBVExperimentConfig
from nbv_framework.training.loss import ReconstructionLoss
from nbv_framework.training.runtime_utils import set_random_seed
from nbv_framework.utils.data_utils import create_synthetic_training_data
from nbv_framework.utils.evaluation import compare_with_baselines, evaluate_nbv_policy
from nbv_framework.utils.logging_utils import configure_logging, get_logger


LOGGER = get_logger(__name__)


class NBVExperiment:
    """Encapsulates the full experiment lifecycle."""

    def __init__(self, cfg: NBVExperimentConfig):
        self.cfg = cfg
        self.mapanything_wrapper: Optional[MapAnythingWrapper] = None
        self.policy_network: Optional[BaseNBVPolicy] = None
        self.renderer: Optional[DifferentiableRenderer] = None
        self.loss_fn: Optional[ReconstructionLoss] = None
        self.train_loader = None
        self.val_loader = None

    # ------------------------------------------------------------------ //
    # Public entrypoint
    # ------------------------------------------------------------------ //
    def launch(self) -> None:
        """Configure logging, build everything, and execute."""
        self._configure_logging()
        set_random_seed(self.cfg.seed + self.cfg.rank)
        self._log_run_header()
        self._maybe_create_synthetic_data()
        self._barrier_if_needed()
        self._build_models()
        self._maybe_wrap_policy_for_ddp()

        if self.cfg.mode in {"train", "all"}:
            self._build_data_loaders()
            self._train_policy()

        if self.cfg.mode in {"eval", "all"}:
            self._run_evaluation()

        if self.cfg.is_main_process:
            LOGGER.info("Demo completed successfully!")

    # ------------------------------------------------------------------ //
    # Internal helpers
    # ------------------------------------------------------------------ //
    def _configure_logging(self) -> None:
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        log_file = os.path.join(self.cfg.log_dir, f"train_rank{self.cfg.rank}.log")
        configure_logging(
            level=logging.INFO,
            log_file=log_file,
            rank=self.cfg.rank,
            enable_console=self.cfg.is_main_process,
            replace_print=True,
        )

    def _log_run_header(self) -> None:
        if not self.cfg.is_main_process:
            return

        LOGGER.info("NBV Framework Demo")
        LOGGER.info("=" * 50)
        LOGGER.info("Device: %s", self.cfg.device)
        LOGGER.info("Mode: %s", self.cfg.mode)
        LOGGER.info("Max meshes: %s", self.cfg.max_meshes)
        LOGGER.info(
            "Initial views: min=%s, max=%s, randomize=%s",
            self.cfg.min_initial_views,
            self.cfg.max_initial_views,
            self.cfg.randomize_initial_views,
        )
        if self.cfg.resume_checkpoint:
            LOGGER.info("Resume checkpoint specified: %s", self.cfg.resume_checkpoint)
        if not self.cfg.auto_resume:
            LOGGER.info("Auto-resume disabled")

    def _maybe_create_synthetic_data(self) -> None:
        if not self.cfg.is_main_process:
            return
        if not (self.cfg.create_data or not os.path.exists(self.cfg.synthetic_data_root)):
            return

        LOGGER.info("Creating synthetic training data...")
        create_synthetic_training_data(
            output_dir=self.cfg.synthetic_data_root,
            num_objects=20,
            num_views_per_object=15,
            image_size=self.cfg.image_size,
        )
        LOGGER.info("Synthetic data created successfully!")

    def _barrier_if_needed(self) -> None:
        if self.cfg.distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _build_models(self) -> None:
        LOGGER.info("Setting up models on device: %s", self.cfg.device)
        self.mapanything_wrapper = MapAnythingWrapper(
            model_name="facebook/map-anything",
            device=self.cfg.device,
        )
        self.policy_network = AttentionNBVPolicy(
            scene_feature_dim=self.cfg.scene_feature_dim,
            hidden_dim=self.cfg.policy_hidden_dim,
            num_heads=self.cfg.policy_num_heads,
            num_layers=self.cfg.policy_num_layers,
            output_mode=self.cfg.policy_output_mode,
        ).to(self.cfg.device)

        self.renderer = DifferentiableRenderer(
            image_size=self.cfg.image_size,
            device=self.cfg.device,
            quality="high",
            downsample_factor=2,
        )

        self.loss_fn = ReconstructionLoss(
            renderer=self.renderer,
            pose_up_axis=self.cfg.up_axis,
        )
        LOGGER.info("Models setup completed!")

    def _maybe_wrap_policy_for_ddp(self) -> None:
        if not self.cfg.distributed or self.policy_network is None:
            return

        device_index = self.cfg.device.index if isinstance(self.cfg.device, torch.device) else None
        if device_index is None:
            raise RuntimeError("CUDA device index is required for DistributedDataParallel")

        self.policy_network = DDP(
            self.policy_network,
            device_ids=[device_index],
            output_device=device_index,
            find_unused_parameters=True,
        )

    def _build_data_loaders(self) -> None:
        LOGGER.info("Setting up data loaders...")
        train_dataset = MixedDataset(
            dataset_configs=[
                {
                    "name": "House3KDataset",
                    "type": "house3k",
                    "data_root": "/mnt/sdb/chenmohan/VGGT-NBV/models/test",
                    "num_initial_views": self.cfg.max_initial_views,
                    "image_size": self.cfg.image_size,
                    "normalize_method": self.cfg.normalize_method,
                    "num_samples": self.cfg.num_samples,
                    "split": "train",
                    "max_meshes": self.cfg.max_meshes,
                    "use_cache": True,
                    "up_axis": self.cfg.up_axis,
                    "manual_camera_position": self.cfg.manual_camera_position,
                    "manual_camera_look_at": self.cfg.manual_camera_look_at,
                    "use_manual_camera": self.cfg.use_manual_camera,
                    "randomize_views_per_call": self.cfg.randomize_views_per_call,
                    "process_rank": self.cfg.rank,
                },
            ],
            seed=42,
        )

        train_repeat_factor = max(1, int(self.cfg.train_repeat_factor))
        if train_repeat_factor > 1:
            original_len = len(train_dataset)
            train_dataset = RepeatedDataset(train_dataset, train_repeat_factor)
            LOGGER.info(
                "Train dataset repeated %dx: %d -> %d samples",
                train_repeat_factor,
                original_len,
                len(train_dataset),
            )

        val_dataset = MixedDataset(
            dataset_configs=[
                {
                    "name": "House3KDataset",
                    "type": "house3k",
                    "data_root": "/mnt/sdb/chenmohan/VGGT-NBV/models/House3K_obj",
                    "num_initial_views": self.cfg.max_initial_views,
                    "image_size": self.cfg.image_size,
                    "normalize_method": self.cfg.normalize_method,
                    "num_samples": self.cfg.num_samples,
                    "split": "val",
                    "max_meshes": self.cfg.max_meshes,
                    "use_cache": True,
                    "up_axis": self.cfg.up_axis,
                    "randomize_views_per_call": False,
                    "process_rank": self.cfg.rank,
                },
            ],
            seed=42,
        )

        val_repeat_factor = max(1, int(self.cfg.val_repeat_factor))
        if val_repeat_factor > 1:
            original_val_len = len(val_dataset)
            val_dataset = RepeatedDataset(val_dataset, val_repeat_factor)
            LOGGER.info(
                "Val dataset repeated %dx: %d -> %d samples",
                val_repeat_factor,
                original_val_len,
                len(val_dataset),
            )

        train_sampler = None
        val_sampler = None
        if self.cfg.distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.cfg.world_size,
                rank=self.cfg.rank,
                shuffle=True,
                seed=self.cfg.seed,
                drop_last=True,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.cfg.world_size,
                rank=self.cfg.rank,
                shuffle=False,
                seed=self.cfg.seed,
                drop_last=False,
            )

        self.train_loader = create_train_loader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=10,
            sampler=train_sampler,
        )
        self.val_loader = create_val_loader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=10,
            sampler=val_sampler,
        )
        LOGGER.info(
            "Data loaders created - Train: %d, Val: %d",
            len(train_dataset),
            len(val_dataset),
        )

    def _train_policy(self) -> None:
        if any(
            component is None
            for component in (self.mapanything_wrapper, self.policy_network, self.renderer, self.loss_fn)
        ):
            raise RuntimeError("Models must be built before training.")

        if self.cfg.is_main_process:
            LOGGER.info("Starting NBV policy training...")

        trainer = NBVTrainer(
            vggt_wrapper=self.mapanything_wrapper,
            policy_network=self.policy_network,
            renderer=self.renderer,
            loss_fn=self.loss_fn,
            num_epochs=self.cfg.num_epochs,
            learning_rate=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            device=self.cfg.device,
            log_dir=self.cfg.log_dir,
            min_initial_views=self.cfg.min_initial_views,
            max_initial_views=self.cfg.max_initial_views,
            randomize_initial_views=self.cfg.randomize_initial_views,
            enable_validation=self.cfg.enable_validation,
            use_epoch_seed=self.cfg.use_epoch_seed,
            distributed=self.cfg.distributed,
            world_size=self.cfg.world_size,
            rank=self.cfg.rank,
        )

        resume_checkpoint = self._resolve_resume_checkpoint()
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            try:
                trainer.load_checkpoint(resume_checkpoint)
                LOGGER.info("Successfully resumed from epoch %d", trainer.current_epoch)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.error("Failed to load checkpoint: %s", exc)
                LOGGER.info("Starting training from scratch...")
        elif self.cfg.is_main_process:
            LOGGER.info("No checkpoint found, starting training from scratch...")

        trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            save_dir=self.cfg.save_dir,
        )

    def _resolve_resume_checkpoint(self) -> Optional[str]:
        if self.cfg.resume_checkpoint:
            LOGGER.info("Resuming from specified checkpoint: %s", self.cfg.resume_checkpoint)
            return self.cfg.resume_checkpoint
        if not self.cfg.auto_resume:
            return None

        if not os.path.exists(self.cfg.save_dir):
            return None

        checkpoint_files = []
        for file_name in os.listdir(self.cfg.save_dir):
            if not (file_name.startswith("checkpoint_epoch_") and file_name.endswith(".pth")):
                continue
            try:
                epoch_num = int(file_name.replace("checkpoint_epoch_", "").replace(".pth", ""))
            except ValueError:
                continue
            checkpoint_files.append((epoch_num, os.path.join(self.cfg.save_dir, file_name)))

        if not checkpoint_files:
            return None

        latest_checkpoint = max(checkpoint_files, key=lambda item: item[0])[1]
        LOGGER.info("Auto-resuming from latest checkpoint: %s", latest_checkpoint)
        return latest_checkpoint

    def _run_evaluation(self) -> None:
        if not self.cfg.is_main_process or self.policy_network is None or self.renderer is None:
            return

        policy = self.policy_network
        if hasattr(policy, "module"):
            policy = policy.module  # type: ignore[attr-defined]

        LOGGER.info("Running evaluation...")
        test_dataset = SyntheticDataset(
            data_root=self.cfg.synthetic_data_root,
            num_initial_views=self.cfg.max_initial_views,
            image_size=self.cfg.image_size,
            split="val",
        )
        test_data = [test_dataset[i] for i in range(min(5, len(test_dataset)))]

        evaluation_results = evaluate_nbv_policy(
            policy_network=policy,
            vggt_wrapper=self.mapanything_wrapper,
            renderer=self.renderer,
            test_data=test_data,
            max_views=self.cfg.max_initial_views,
            device=self.cfg.device,
        )
        LOGGER.info("Evaluation Results:")
        for metric, value in evaluation_results.items():
            LOGGER.info("  %s: %.6f", metric, value)

        LOGGER.info("Comparing with baseline methods...")
        comparison_results = compare_with_baselines(
            policy_network=policy,
            vggt_wrapper=self.mapanything_wrapper,
            renderer=self.renderer,
            test_data=test_data[:3],
            device=self.cfg.device,
        )
        LOGGER.info("Comparison Results:")
        for method, results in comparison_results.items():
            LOGGER.info("%s:", method.upper())
            for metric, value in results.items():
                LOGGER.info("  %s: %.6f", metric, value)
