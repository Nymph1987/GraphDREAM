#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
PORT="${PORT:-1698}"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "$PYTHON_BIN" -u run.py \
  --gpu "$GPU" \
  --port "$PORT" \
  --classify emotion \
  --dataset MELD \
  --epochs 60 \
  --textf_mode textf0 \
  --loss_type emo_sen_sft \
  --lr 6.9e-5 \
  --batch_size 16 \
  --hidden_dim 384 \
  --win 5 5 \
  --heter_n_layers 4 4 4 \
  --drop 0.30 \
  --shift_win 10 \
  --lambd 1.22 0.68 0.34 \
  --select_best_by dev \
  --refine_heads 4 \
  --dropedge 0.0 \
  --gcn_residual_drop 0.15 \
  --contrastive_weight 0.016 \
  --decouple_weight 0.0035 \
  --warmup_epochs 10 \
  --supcon_temperature 0.070 \
  --label_smoothing 0.07 \
  --class_balance_beta 0.9996 \
  --hard_negative_weight 1.0 \
  --pair_aux_weight_04 0.0 \
  --pair_aux_weight_25 0.0 \
  --proto_aux_weight 0.0 \
  --hier_aux_weight 0.0 \
  --hier_kl_weight 0.0 \
  --use_temporal_fusion \
  --temporal_logit_scale 0.36 \
  --temporal_aux_weight 0.09 \
  --pair04_sample_boost 1.0 \
  --pair04_margin 0.18 \
  --pair04_margin_weight 0.0 \
  "$@"
