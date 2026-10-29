"""Image and uint8 tensor-cache loaders for the stage-split CSVs.

Each split CSV has the columns ``filepath,label`` with ``filepath`` relative to
the data root. With ``cache_mode="uint8"`` the decoded, resized images are stored
once as uint8 CHW tensor shards and reused by later runs.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm.auto import tqdm


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class CSVImageDataset(Dataset):
    def __init__(self, csv_path: Path, data_root: Path, transform: Any) -> None:
        self.csv_path = csv_path
        self.data_root = data_root
        self.transform = transform
        self.df = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = self.data_root / str(row["filepath"])
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image)
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y


class CachedTensorTransform:
    def __init__(self, train: bool) -> None:
        self.train = train

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.train and random.random() < 0.5:
            image = torch.flip(image, dims=[2])
        image = image.float().div_(255.0)
        mean = IMAGENET_MEAN.to(device=image.device)
        std = IMAGENET_STD.to(device=image.device)
        return (image - mean) / std


class CachedTensorDataset(Dataset):
    def __init__(self, manifest_path: Path, transform: Any) -> None:
        self.manifest_path = manifest_path
        self.transform = transform
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.manifest = manifest
        images: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        for shard in tqdm(manifest["shards"], desc=f"load cache {manifest_path.stem}", leave=False, dynamic_ncols=True):
            payload = torch.load(manifest_path.parent / shard["file"], map_location="cpu")
            images.append(payload["images"])
            labels.append(payload["labels"])
        if not images:
            raise RuntimeError(f"empty cache manifest: {manifest_path}")
        self.images = torch.cat(images, dim=0).contiguous()
        self.labels = torch.cat(labels, dim=0).long()

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


def build_transforms(image_size: int, cache_mode: str = "none") -> Tuple[Any, Any]:
    if cache_mode == "uint8":
        return CachedTensorTransform(train=True), CachedTensorTransform(train=False)

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def split_csv(split_dir: Path, stage: int, domain: str, split: str) -> Path:
    return split_dir / f"stage{stage}_{domain}_{split}.csv"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_stem(csv_path: Path, image_size: int, max_samples: int | None) -> str:
    suffix = f"_first{max_samples}" if max_samples is not None else "_full"
    return f"{csv_path.stem}_size{image_size}{suffix}"


def cache_manifest_path(cache_dir: Path, csv_path: Path, image_size: int, max_samples: int | None) -> Path:
    return cache_dir / f"{cache_stem(csv_path, image_size, max_samples)}_manifest.json"


def cache_is_ready(manifest_path: Path, csv_path: Path, image_size: int, max_samples: int | None) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if int(manifest.get("image_size", -1)) != int(image_size):
        return False
    if manifest.get("max_samples") != max_samples:
        return False
    try:
        csv_count = len(pd.read_csv(csv_path))
    except Exception:
        return False
    if max_samples is not None:
        csv_count = min(csv_count, max_samples)
    if int(manifest.get("count", -1)) != int(csv_count):
        return False
    return all((manifest_path.parent / shard["file"]).exists() for shard in manifest.get("shards", []))


def pil_to_uint8_chw(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def build_cache_for_csv(
    csv_path: Path,
    data_root: Path,
    cache_dir: Path,
    image_size: int,
    shard_size: int,
    max_samples: int | None,
    overwrite: bool,
) -> Path:
    manifest_path = cache_manifest_path(cache_dir, csv_path, image_size, max_samples)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and cache_is_ready(manifest_path, csv_path, image_size, max_samples):
        print(f"[tensor_cache] ready: {manifest_path}")
        return manifest_path

    df = pd.read_csv(csv_path)
    if max_samples is not None:
        df = df.iloc[:min(max_samples, len(df))].copy()

    stem = cache_stem(csv_path, image_size, max_samples)
    shards: List[Dict[str, Any]] = []
    images: List[torch.Tensor] = []
    labels: List[int] = []
    filepaths: List[str] = []

    def flush_shard(shard_idx: int, start_idx: int) -> None:
        if not images:
            return
        shard_name = f"{stem}_shard{shard_idx:05d}.pt"
        shard_path = cache_dir / shard_name
        tmp_path = cache_dir / f".{shard_name}.tmp"
        payload = {
            "images": torch.stack(images, dim=0),
            "labels": torch.tensor(labels, dtype=torch.long),
            "filepaths": list(filepaths),
        }
        torch.save(payload, tmp_path)
        tmp_path.replace(shard_path)
        shards.append({
            "file": shard_name,
            "start": start_idx,
            "count": len(labels),
        })
        images.clear()
        labels.clear()
        filepaths.clear()

    start_idx = 0
    shard_idx = 0
    iterator = tqdm(df.itertuples(index=False), total=len(df), desc=f"build cache {csv_path.stem}", dynamic_ncols=True)
    for row_idx, row in enumerate(iterator):
        rel_path = str(getattr(row, "filepath"))
        image_path = data_root / rel_path
        with Image.open(image_path) as image:
            images.append(pil_to_uint8_chw(image, image_size))
        labels.append(int(getattr(row, "label")))
        filepaths.append(rel_path)
        if len(images) >= shard_size:
            flush_shard(shard_idx, start_idx)
            start_idx = row_idx + 1
            shard_idx += 1
    flush_shard(shard_idx, start_idx)

    manifest = {
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "data_root": str(data_root),
        "image_size": image_size,
        "max_samples": max_samples,
        "count": int(len(df)),
        "dtype": "uint8",
        "layout": "CHW",
        "shard_size": shard_size,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "shards": shards,
    }
    tmp_manifest = cache_dir / f".{manifest_path.name}.tmp"
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_manifest.replace(manifest_path)
    print(f"[tensor_cache] wrote: {manifest_path}")
    return manifest_path


def make_loader(
    csv_path: Path,
    data_root: Path,
    transform: Any,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    max_samples: int | None,
    cache_mode: str = "none",
    cache_dir: Path | None = None,
    image_size: int = 224,
    cache_shard_size: int = 2048,
    build_cache: bool = False,
    cache_overwrite: bool = False,
) -> DataLoader:
    if not csv_path.exists():
        raise FileNotFoundError(f"split CSV not found: {csv_path}")
    if cache_mode == "uint8":
        if cache_dir is None:
            raise ValueError("cache_dir must be set when cache_mode is not none")
        manifest_path = cache_manifest_path(cache_dir, csv_path, image_size, max_samples)
        if build_cache or cache_overwrite or not cache_is_ready(manifest_path, csv_path, image_size, max_samples):
            if not build_cache and not cache_overwrite:
                raise FileNotFoundError(
                    f"cache missing for {csv_path}. Re-run with --build_cache to create it."
                )
            manifest_path = build_cache_for_csv(
                csv_path=csv_path,
                data_root=data_root,
                cache_dir=cache_dir,
                image_size=image_size,
                shard_size=cache_shard_size,
                max_samples=max_samples,
                overwrite=cache_overwrite,
            )
        dataset = CachedTensorDataset(manifest_path, transform)
    else:
        dataset = CSVImageDataset(csv_path, data_root, transform)
        if max_samples is not None:
            dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )
