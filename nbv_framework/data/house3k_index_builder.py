"""Index-building helpers for House3K dataset splits."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from .house3k_utils import find_batch_directories, scan_house3k_batches, split_house3k_dataset


@dataclass(frozen=True)
class House3KIndexConfig:
    data_root: str
    seed: int
    split: str
    train_ratio: float
    val_ratio: float
    max_meshes: Optional[int]


def apply_max_mesh_limit(
    objects: List[Dict],
    *,
    max_meshes: Optional[int],
    seed: int,
) -> List[Dict]:
    if not max_meshes or len(objects) <= int(max_meshes):
        return list(objects)

    limited = list(objects)
    rng = random.Random(seed)
    rng.shuffle(limited)
    return limited[: int(max_meshes)]


def build_house3k_split_objects(config: House3KIndexConfig) -> List[Dict]:
    logger.info("正在扫描House3K数据集: {}，seed={}", config.data_root, config.seed)
    data_root_path = Path(config.data_root)

    batch_dirs = find_batch_directories(data_root_path)
    logger.info("找到 {} 个批次目录: {}", len(batch_dirs), [d.name for d in batch_dirs])

    all_objects, total_scanned = scan_house3k_batches(batch_dirs, logger=logger)
    logger.info("[House3K数据集] 总共扫描 {} 个3D模型", total_scanned)
    logger.info("[House3K数据集] 加载 {} 个有效3D模型", len(all_objects))

    limited_objects = apply_max_mesh_limit(
        all_objects,
        max_meshes=config.max_meshes,
        seed=config.seed,
    )
    if len(limited_objects) != len(all_objects):
        logger.info(
            "[House3K数据集] 应用全局mesh限制，从 {} 个减少到 {} 个",
            len(all_objects),
            len(limited_objects),
        )

    split_objects, split_stats = split_house3k_dataset(
        limited_objects,
        split=config.split,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )
    logger.info(
        "数据集分割 - 总计: {}, 训练: {}, 验证: {}, 测试: {}",
        split_stats["total"],
        split_stats["train"],
        split_stats["val"],
        split_stats["test"],
    )
    logger.info("当前分割 '{}': 加载了 {} 个样本", config.split, split_stats["current_split"])
    return split_objects
