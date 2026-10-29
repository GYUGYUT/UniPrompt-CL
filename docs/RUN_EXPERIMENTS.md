# Running Experiments

This project supports two medical continual-learning streams:

- `dr`: APTOS -> DDR -> DRD
- `skin`: ISIC -> HAM10000 -> Derm7pt

All paths are relative so the code can be submitted anonymously.

## Expected Layout

Stage split CSVs are already stored under:

```text
dataloader/stage_splits/dr/
dataloader/stage_splits/skin/
```

If using cached tensors, cache files must be present under:

```text
dataloader/tensor_cache/dr/
dataloader/tensor_cache/skin/
```

If not using cache, images are expected under:

```text
data/aptos_train/
data/aptos_val/
data/aptos_test/
data/ddr/
data/drd/
data/skin/ISIC/
data/skin/HAM10000/
data/skin/Derm7pt/
```

## Run DR

With cached tensors:

```bash
python main.py \
  --dataset dr \
  --cache_mode uint8 \
  --num_classes 3 \
  --diabetic_classes 3 \
  --img_size 224
```

Without cached tensors:

```bash
python main.py \
  --dataset dr \
  --num_classes 3 \
  --diabetic_classes 3 \
  --img_size 224
```

## Run Skin-Cancer

With cached tensors:

```bash
python main.py \
  --dataset skin \
  --cache_mode uint8 \
  --skin_classes 2 \
  --num_classes 2 \
  --diabetic_classes 2 \
  --img_size 224
```

Without cached tensors:

```bash
python main.py \
  --dataset skin \
  --skin_classes 2 \
  --num_classes 2 \
  --diabetic_classes 2 \
  --img_size 224
```

## Outputs

Each run creates a timestamped folder under `results/`.

Main continual-learning metric files:

```text
results/<timestamp>/cl_accuracy_matrix.csv
results/<timestamp>/cl_macro_f1_matrix.csv
results/<timestamp>/cl_metrics.json
results/<timestamp>/cl_metrics_summary.txt
```

`cl_metrics_summary.txt` stores AvgACC, BWT, and AvgF for both accuracy and macro-F1.

Each stage also stores checkpoints and test artifacts under:

```text
results/<timestamp>/step1/
results/<timestamp>/step2/
results/<timestamp>/step3/
```

## Notes

- `scripts/train_dr.sh` and `scripts/train_skin.sh` launch the paper configuration; the explicit commands above show the minimal invocation.
- Do not pass `--build_cache` unless you intentionally want to create missing cache files.
- To use a custom data root, pass `--medical_data_root <path>`.
- To use custom split/cache locations, pass `--stage_split_dir`, `--cache_dir`, `--skin_stage_split_dir`, or `--skin_cache_dir`.
