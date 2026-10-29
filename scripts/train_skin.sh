#!/usr/bin/env bash
# Skin cancer stream: ISIC -> HAM10000 -> Derm7pt, paper settings (Sup. Table 6; batch size 16).
set -e
cd "$(dirname "$0")/.."

python main.py \
  --dataset skin \
  --skin_classes 2 --num_classes 2 --diabetic_classes 2 \
  --img_size 224 --batch_size 16 \
  --num_prompts 500 --expand_per_stage 100 --prompt_dim 768 --lambda_qr 0.001 \
  --optimizer adamw --lr 0.001 --weight_decay 1e-4 \
  --scheduler cosine --epochs 100 --patience 5 \
  --gpu_ids "${GPU_IDS:-0}" \
  "$@"
