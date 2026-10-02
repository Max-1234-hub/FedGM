#!/usr/bin/env bash
set -euo pipefail

# Usage: bash examples/train.sh /path/to/dataset-root [Fundus|Polyp|Prostate|FL_Breast_Ultrasound] [FedGM|FedAvg|FedProx|MOON|FedNova|PN|FedAWA]
DATA_ROOT=${1:?Provide the dataset root}
DATASET=${2:-Fundus}
METHOD=${3:-FedGM}
GPU=${GPU:-0}
cd "$(dirname "$0")/.."

python -u main_seg.py \
  --data_path "$DATA_ROOT" \
  --dataset "$DATASET" \
  --method "$METHOD" \
  --device "$GPU" \
  --save_path "results/${DATASET}/${METHOD}" \
  --local_model UNet2D \
  --T 200 --E 2 --batchsize 8 \
  --optimizer adam --loss dice_bce \
  --lr 0.001 --gm_lr 0.005
