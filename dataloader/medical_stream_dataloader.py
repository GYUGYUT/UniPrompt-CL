from __future__ import annotations

from pathlib import Path
from typing import List

DEFAULT_DR_CACHE_DIR = Path("dataloader") / "tensor_cache" / "dr"

from torch.utils.data import DataLoader

from dataloader.medical_cache import build_transforms, make_loader, split_csv
from dataloader.skin_dataloader import (
    skin_create_dataloader,
    skin_num_classes,
    skin_stage_domains,
)


STAGE_DOMAINS = {
    "dr": ["aptos", "ddr", "diabetic"],
    "skin": ["ISIC", "HAM10000", "Derm7pt"],
}


def _none_to_default(value, default):
    return default if value in (None, "", "None") else value


def get_dataset_name(args) -> str:
    dataset = getattr(args, "dataset", "dr")
    if dataset not in STAGE_DOMAINS:
        raise ValueError(f"Unsupported dataset: {dataset}. Expected one of {sorted(STAGE_DOMAINS)}")
    return dataset


def get_stage_domains(args) -> List[str]:
    if get_dataset_name(args) == "skin":
        return skin_stage_domains()
    return STAGE_DOMAINS[get_dataset_name(args)]


def get_stage_num_classes(args) -> int:
    dataset = get_dataset_name(args)
    if dataset == "skin":
        return skin_num_classes(args)
    return int(getattr(args, "diabetic_classes", getattr(args, "num_classes", 3)))


def get_split_dir(args) -> Path:
    dataset = get_dataset_name(args)
    split_dir = getattr(args, "stage_split_dir", None)
    split_dir = _none_to_default(split_dir, Path("dataloader") / "stage_splits" / dataset)
    return Path(split_dir)


def get_cache_dir(args) -> Path:
    dataset = get_dataset_name(args)
    cache_dir = getattr(args, "cache_dir", None)
    cache_dir = _none_to_default(cache_dir, DEFAULT_DR_CACHE_DIR)
    return Path(cache_dir)


def get_data_root(args) -> Path:
    data_root = getattr(args, "medical_data_root", None)
    data_root = _none_to_default(data_root, "data")
    return Path(data_root)


def create_stage_dataloader(args, stage: int, split: str) -> DataLoader:
    if get_dataset_name(args) == "skin":
        return skin_create_dataloader(args, stage=stage, is_train=split)

    domains = get_stage_domains(args)
    if not 1 <= stage <= len(domains):
        raise IndexError(f"stage {stage} out of range for {get_dataset_name(args)}")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    domain = domains[stage - 1]
    cache_mode = getattr(args, "cache_mode", "none")
    image_size = int(getattr(args, "img_size", 224))
    train_transform, eval_transform = build_transforms(image_size, cache_mode=cache_mode)
    transform = train_transform if split == "train" else eval_transform
    batch_size = int(getattr(args, "batch_size", 64)) if split != "test" else int(getattr(args, "test_batch_size", 1))
    num_workers = int(getattr(args, "num_workers", 4))

    return make_loader(
        csv_path=split_csv(get_split_dir(args), stage, domain, split),
        data_root=get_data_root(args),
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == "train"),
        max_samples=getattr(args, "max_samples_per_split", None),
        cache_mode=cache_mode,
        cache_dir=get_cache_dir(args),
        image_size=image_size,
        cache_shard_size=int(getattr(args, "cache_shard_size", 2048)),
        build_cache=bool(getattr(args, "build_cache", False)),
        cache_overwrite=bool(getattr(args, "cache_overwrite", False)),
    )
