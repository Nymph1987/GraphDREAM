#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
PORT="${PORT:-1693}"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "$PYTHON_BIN" -u run.py \
  --gpu "$GPU" \
  --port "$PORT" \
  --classify emotion \
  --dataset IEMOCAP \
  --epochs 60 \
  --textf_mode textf0 \
  --loss_type emo_sen_sft \
  --lr 8.0e-5 \
  --batch_size 16 \
  --hidden_dim 384 \
  --win 11 11 \
  --heter_n_layers 6 6 6 \
  --drop 0.28 \
  --shift_win 15 \
  --lambd 1.22 0.68 0.40 \
  --select_best_by dev \
  --refine_heads 4 \
  --dropedge 0.0 \
  --gcn_residual_drop 0.15 \
  --contrastive_weight 0.032 \
  --decouple_weight 0.0 \
  --warmup_epochs 12 \
  --supcon_temperature 0.068 \
  --label_smoothing 0.10 \
  --class_balance_beta 0.9998 \
  --hard_negative_weight 1.7 \
  --pair_aux_weight_04 0.045 \
  --pair_aux_weight_25 0.040 \
  --proto_aux_weight 0.10 \
  --hier_aux_weight 0.12 \
  --hier_kl_weight 0.05 \
  --use_temporal_fusion \
  --temporal_logit_scale 0.47 \
  --temporal_aux_weight 0.11 \
  --pair04_sample_boost 2.0 \
  --pair04_margin 0.20 \
  --pair04_margin_weight 0.12 \
  --modality_adv_weight 0.0 \
  --grl_alpha 0.0 \
  --data_path data/newdata_iemocap_features.pkl \
  "$@"
