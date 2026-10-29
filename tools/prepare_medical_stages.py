"""Prepare DR and Skin stage CSVs for the baseline runners.

The baseline runners expect CSV files named ``stage{N}_{domain}_{split}.csv``
with columns ``filepath,label``.  The legacy ECCV dataloaders use dataset-specific
CSV schemas and image roots, so this script normalizes them into that common
format.  Image paths are written as absolute paths; the runner can then use
``--data-root /`` for these medical datasets.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Iterable


def dr_label(label: str) -> int:
    value = int(label)
    if value == 0:
        return 0
    if value in {1, 2, 3}:
        return 1
    if value == 4:
        return 2
    raise ValueError(f"unexpected DR label: {label}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label"])
        writer.writeheader()
        writer.writerows(rows)


def convert_csv(
    src_csv: Path,
    dst_csv: Path,
    image_root: Path,
    filename_fn: Callable[[dict[str, str]], str],
    label_fn: Callable[[str], int],
    label_col: str,
) -> None:
    rows = []
    for row in read_rows(src_csv):
        rows.append({
            "filepath": str((image_root / filename_fn(row)).resolve()),
            "label": label_fn(row[label_col]),
        })
    write_rows(dst_csv, rows)


def prepare_dr(args: argparse.Namespace) -> None:
    dl = args.dataloader_dir
    specs = [
        ("aptos", {
            "train": (dl / "atop" / "train_1.csv", args.aptos_train_root, lambda r: f"{r['id_code']}.png", "diagnosis"),
            "val": (dl / "atop" / "valid.csv", args.aptos_val_root, lambda r: f"{r['id_code']}.png", "diagnosis"),
            "test": (dl / "atop" / "test.csv", args.aptos_test_root, lambda r: f"{r['id_code']}.png", "diagnosis"),
        }),
        ("ddr", {
            "train": (dl / "ddr" / "train.csv", args.ddr_root, lambda r: r["id_code"], "diagnosis"),
            "val": (dl / "ddr" / "val.csv", args.ddr_root, lambda r: r["id_code"], "diagnosis"),
            "test": (dl / "ddr" / "test.csv", args.ddr_root, lambda r: r["id_code"], "diagnosis"),
        }),
        ("diabetic", {
            "train": (dl / "diabetic" / "train.csv", args.diabetic_root, lambda r: f"{r['image']}.jpeg", "level"),
            "val": (dl / "diabetic" / "val.csv", args.diabetic_root, lambda r: f"{r['image']}.jpeg", "level"),
            "test": (dl / "diabetic" / "test.csv", args.diabetic_root, lambda r: f"{r['image']}.jpeg", "level"),
        }),
    ]
    for stage, (domain, split_map) in enumerate(specs, start=1):
        for split, (src, image_root, filename_fn, label_col) in split_map.items():
            convert_csv(src, args.out_split_dir / f"stage{stage}_{domain}_{split}.csv", image_root, filename_fn, dr_label, label_col)


def prepare_skin(args: argparse.Namespace) -> None:
    dl = args.dataloader_dir
    suffix = "_undersampled" if args.undersampled else ""
    specs = [
        ("ISIC", dl / "ISIC", args.isic_root),
        ("HAM10000", dl / "HAM10000", args.ham10000_root),
        ("Derm7pt", dl / "Derm7pt", args.derm7pt_root),
    ]
    for stage, (domain, csv_root, image_root) in enumerate(specs, start=1):
        for split in ("train", "val", "test"):
            src = csv_root / f"{split}{suffix}.csv"
            dst = args.out_split_dir / f"stage{stage}_{domain}_{split}.csv"
            convert_csv(src, dst, image_root, lambda r: r["filepath"], lambda x: int(x), "label")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DR/Skin stage CSVs for copied baselines.")
    parser.add_argument("--dataset", choices=["dr", "skin"], required=True)
    parser.add_argument("--dataloader-dir", type=Path, default=Path("dataloader"))
    parser.add_argument("--out-split-dir", type=Path, required=True)

    parser.add_argument("--aptos-train-root", type=Path, default=Path("data/aptos_train"))
    parser.add_argument("--aptos-val-root", type=Path, default=Path("data/aptos_val"))
    parser.add_argument("--aptos-test-root", type=Path, default=Path("data/aptos_test"))
    parser.add_argument("--ddr-root", type=Path, default=Path("data/ddr"))
    parser.add_argument("--diabetic-root", type=Path, default=Path("data/drd"))

    parser.add_argument("--isic-root", type=Path, default=Path("data/skin/ISIC"))
    parser.add_argument("--ham10000-root", type=Path, default=Path("data/skin/HAM10000"))
    parser.add_argument("--derm7pt-root", type=Path, default=Path("data/skin/Derm7pt"))
    parser.add_argument("--undersampled", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dataset == "dr":
        prepare_dr(args)
    else:
        prepare_skin(args)
    print(f"[prepare_medical_stages] wrote {args.dataset} splits to {args.out_split_dir}")


if __name__ == "__main__":
    main()
