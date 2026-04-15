#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-has0}"
shift || true
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-data/MOSEI/mosei_dialogue_features_metricstop.pkl}"

case "$MODE" in
  has0)
    PORT="${PORT:-1910}"
    exec "$PYTHON_BIN" -u run.py \
      --gpu "$GPU" \
      --port "$PORT" \
      --classify emotion \
      --dataset MOSEI2_HAS0 \
      --epochs 70 \
      --early_stop_patience 10 \
      --textf_mode textf0 \
      --loss_type emo_sft \
      --lr 8.5e-05 \
      --l2 0.0003 \
      --batch_size 32 \
      --hidden_dim 256 \
      --win 5 5 \
      --heter_n_layers 3 3 3 \
      --drop 0.3 \
      --shift_win 3 \
      --lambd 1.1 0.0 0.25 \
      --select_best_by dev \
      --refine_heads 4 \
      --dropedge 0.0 \
      --gcn_residual_drop 0.12 \
      --contrastive_weight 0.01 \
      --decouple_weight 0.001 \
      --warmup_epochs 8 \
      --supcon_temperature 0.070 \
      --label_smoothing 0.05 \
      --class_balance_beta 0.9995 \
      --hard_negative_weight 1.0 \
      --pair_aux_weight_04 0.0 \
      --pair_aux_weight_25 0.0 \
      --pair04_margin_weight 0.0 \
      --proto_aux_weight 0.0 \
      --hier_aux_weight 0.0 \
      --hier_kl_weight 0.0 \
      --use_temporal_fusion \
      --temporal_logit_scale 0.3 \
      --temporal_aux_weight 0.06 \
      --modality_adv_weight 0.0 \
      --disable_tsne \
      --data_path "$DATA_PATH" \
      "$@"
    ;;
  non0)
    PORT="${PORT:-1911}"
    exec "$PYTHON_BIN" -u run.py \
      --gpu "$GPU" \
      --port "$PORT" \
      --classify emotion \
      --dataset MOSEI2_NON0 \
      --epochs 70 \
      --early_stop_patience 10 \
      --textf_mode textf0 \
      --loss_type emo_sft \
      --lr 8.5e-05 \
      --l2 0.0001 \
      --batch_size 16 \
      --hidden_dim 256 \
      --win 7 7 \
      --heter_n_layers 3 3 3 \
      --drop 0.35 \
      --shift_win 3 \
      --lambd 1.1 0.0 0.25 \
      --select_best_by dev \
      --refine_heads 4 \
      --dropedge 0.0 \
      --gcn_residual_drop 0.12 \
      --contrastive_weight 0.0 \
      --decouple_weight 0.0 \
      --warmup_epochs 10 \
      --supcon_temperature 0.070 \
      --label_smoothing 0.02 \
      --class_balance_beta 0.9995 \
      --hard_negative_weight 1.0 \
      --pair_aux_weight_04 0.0 \
      --pair_aux_weight_25 0.0 \
      --pair04_margin_weight 0.0 \
      --proto_aux_weight 0.0 \
      --hier_aux_weight 0.0 \
      --hier_kl_weight 0.0 \
      --use_temporal_fusion \
      --temporal_logit_scale 0.24 \
      --temporal_aux_weight 0.04 \
      --modality_adv_weight 0.0 \
      --disable_tsne \
      --data_path "$DATA_PATH" \
      "$@"
    ;;
  *)
    echo "Usage: $0 {has0|non0} [extra args...]" >&2
    exit 1
    ;;
esac
