#!/usr/bin/env bash
# Diabetic retinopathy stream: APTOS -> DDR -> DRD, paper settings (Sup. Table 6).
set -e
cd "$(dirname "$0")/.."

python main.py \
  --dataset dr \
  --num_classes 3 --diabetic_classes 3 \
  --img_size 224 --batch_size 64 \
  --num_prompts 500 --expand_per_stage 100 --prompt_dim 768 --lambda_qr 0.001 \
  --optimizer adamw --lr 0.001 --weight_decay 1e-4 \
  --scheduler cosine --epochs 100 --patience 5 \
  --gpu_ids "${GPU_IDS:-0}" \
  "$@"
