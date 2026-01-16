"""House3K dataset scanning and splitting utilities."""

import random
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger

SplitStats = Dict[str, int]


def find_batch_directories(data_root: Path) -> List[Path]:
    """Return sorted batch directories under the dataset root."""
    return sorted([d for d in data_root.iterdir() if d.is_dir() and "BATCH" in d.name.upper()])


def check_texture_files(mtl_path: Path, logger=logger) -> bool:
    """Check whether all texture files referenced in an MTL file exist."""
    if not mtl_path.exists():
        return False

    set_path = mtl_path.parent

    try:
        with mtl_path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        texture_keywords = ("map_Kd", "map_Ka", "map_Ks", "map_Ns", "map_d", "map_bump", "bump")
        texture_files_found = []

        for line in lines:
            line = line.strip()
            if line.startswith(texture_keywords):
                parts = line.split()
                if len(parts) > 1:
                    texture_files_found.append(parts[-1])

        if not texture_files_found:
            return False

        return all((set_path / tex_file).exists() for tex_file in texture_files_found)

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(f"读取MTL文件失败 {mtl_path}: {exc}")
        return False


def scan_batch_directory(batch_path: Path, logger=logger) -> List[Dict]:
    """Scan a single batch directory and return all model entries."""
    batch_objects: List[Dict] = []
    obj_files = list(batch_path.glob("Set*/*.obj")) + list(batch_path.glob("SET*/*.obj"))

    for obj_path in obj_files:
        mtl_path = obj_path.with_suffix(".mtl")
        has_valid_textures = check_texture_files(mtl_path, logger=logger)

        batch_objects.append(
            {
                "batch_name": batch_path.name,
                "set_name": obj_path.parent.name,
                "model_name": obj_path.stem,
                "obj_path": str(obj_path),
                "mtl_path": str(mtl_path),
                "set_path": str(obj_path.parent),
                "has_texture": has_valid_textures,
            }
        )

        logger.info(f"批次 {batch_path.name}: 找到 {len(batch_objects)} 个模型")
    return batch_objects


def scan_house3k_batches(batch_dirs: List[Path], logger=logger) -> Tuple[List[Dict], int]:
    """Scan all batches, returning textured objects and total scanned count."""
    all_objects: List[Dict] = []
    total_scanned = 0

    for batch_dir in batch_dirs:
        batch_objects = scan_batch_directory(batch_dir, logger=logger)
        total_scanned += len(batch_objects)
        valid_objects = [obj for obj in batch_objects if obj.get("has_texture", False)]
        all_objects.extend(valid_objects)

    return all_objects, total_scanned


def split_house3k_dataset(
    all_objects: List[Dict],
    split: str,
    train_ratio: float,
    val_ratio: float,
    rng_seed: int = 42,
) -> Tuple[List[Dict], SplitStats]:
    """Shuffle and split objects according to the configured ratios."""
    shuffled_objects = all_objects.copy()
    rng = random.Random(rng_seed)
    rng.shuffle(shuffled_objects)

    total_count = len(shuffled_objects)

    if split == "train":
        split_end = int(total_count * train_ratio)
        split_objects = shuffled_objects[:split_end]
    elif split == "val":
        train_end = int(total_count * train_ratio)
        val_end = train_end + int(total_count * val_ratio)
        split_objects = shuffled_objects[train_end:val_end]
    elif split == "test":
        train_end = int(total_count * train_ratio)
        val_end = train_end + int(total_count * val_ratio)
        split_objects = shuffled_objects[val_end:]
    else:
        raise ValueError(f"未知的分割类型: {split}")

    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count

    stats: SplitStats = {
        "total": total_count,
        "train": train_count,
        "val": val_count,
        "test": test_count,
        "current_split": len(split_objects),
    }

    return split_objects, stats
