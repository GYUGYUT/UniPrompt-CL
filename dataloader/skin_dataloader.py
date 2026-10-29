from __future__ import annotations

from pathlib import Path
from typing import List

from torch.utils.data import DataLoader

from dataloader.medical_cache import build_transforms, make_loader, split_csv


DEFAULT_SKIN_CACHE_DIR = Path("dataloader") / "tensor_cache" / "skin"
SKIN_DOMAINS = ["ISIC", "HAM10000", "Derm7pt"]


def _none_to_default(value, default):
    return default if value in (None, "", "None") else value


def skin_stage_domains() -> List[str]:
    return SKIN_DOMAINS


def skin_num_classes(args) -> int:
    return int(getattr(args, "skin_classes", getattr(args, "num_classes", 2)))


def skin_split_dir(args) -> Path:
    split_dir = getattr(args, "skin_stage_split_dir", None)
    split_dir = _none_to_default(split_dir, getattr(args, "stage_split_dir", None))
    split_dir = _none_to_default(
        split_dir,
        Path("dataloader") / "stage_splits" / "skin",
    )
    return Path(split_dir)


def skin_cache_dir(args) -> Path:
    cache_dir = getattr(args, "skin_cache_dir", None)
    cache_dir = _none_to_default(cache_dir, getattr(args, "cache_dir", None))
    cache_dir = _none_to_default(
        cache_dir,
        DEFAULT_SKIN_CACHE_DIR,
    )
    return Path(cache_dir)


def skin_data_root(args) -> Path:
    data_root = getattr(args, "skin_data_root", None)
    data_root = _none_to_default(data_root, getattr(args, "medical_data_root", None))
    data_root = _none_to_default(data_root, "data")
    return Path(data_root)


def skin_create_dataloader(args, stage: int, is_train: str = "train") -> DataLoader:
    if not 1 <= stage <= len(SKIN_DOMAINS):
        raise IndexError(f"skin stage {stage} out of range")
    if is_train not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {is_train}")

    domain = SKIN_DOMAINS[stage - 1]
    cache_mode = getattr(args, "cache_mode", "none")
    image_size = int(getattr(args, "img_size", 224))
    train_transform, eval_transform = build_transforms(image_size, cache_mode=cache_mode)
    transform = train_transform if is_train == "train" else eval_transform
    batch_size = (
        int(getattr(args, "batch_size", 64))
        if is_train != "test"
        else int(getattr(args, "test_batch_size", 1))
    )

    return make_loader(
        csv_path=split_csv(skin_split_dir(args), stage, domain, is_train),
        data_root=skin_data_root(args),
        transform=transform,
        batch_size=batch_size,
        num_workers=int(getattr(args, "num_workers", 4)),
        shuffle=(is_train == "train"),
        max_samples=getattr(args, "max_samples_per_split", None),
        cache_mode=cache_mode,
        cache_dir=skin_cache_dir(args),
        image_size=image_size,
        cache_shard_size=int(getattr(args, "cache_shard_size", 2048)),
        build_cache=bool(getattr(args, "build_cache", False)),
        cache_overwrite=bool(getattr(args, "cache_overwrite", False)),
    )
