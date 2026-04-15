#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-5}"
shift || true
GPU="${GPU:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-data/CH-SIMSv2/chsims_v2_supervised_dialogue_features_paper_labels.pkl}"

case "$MODE" in
  2)
    PORT="${PORT:-1767}"
    exec "$PYTHON_BIN" -u run.py \
      --gpu "$GPU" \
      --port "$PORT" \
      --classify emotion \
      --dataset CHSIMS2 \
      --epochs 100 \
      --early_stop_patience 20 \
      --textf_mode textf0 \
      --loss_type emo_sen_sft \
      --lr 9.5e-5 \
      --batch_size 32 \
      --hidden_dim 256 \
      --win 7 7 \
      --heter_n_layers 3 3 3 \
      --drop 0.30 \
      --shift_win 4 \
      --lambd 1.18 0.68 0.28 \
      --select_best_by dev \
      --refine_heads 4 \
      --dropedge 0.0 \
      --gcn_residual_drop 0.10 \
      --contrastive_weight 0.010 \
      --decouple_weight 0.0020 \
      --warmup_epochs 6 \
      --supcon_temperature 0.070 \
      --label_smoothing 0.025 \
      --class_balance_beta 0.9990 \
      --hard_negative_weight 1.0 \
      --use_temporal_fusion \
      --temporal_logit_scale 0.28 \
      --temporal_aux_weight 0.04 \
      --disable_tsne \
      --data_path "$DATA_PATH" \
      "$@"
    ;;
  3)
    PORT="${PORT:-1731}"
    exec "$PYTHON_BIN" -u run.py \
      --gpu "$GPU" \
      --port "$PORT" \
      --classify emotion \
      --dataset CHSIMS3 \
      --epochs 80 \
      --textf_mode textf0 \
      --loss_type emo_sen_sft \
      --lr 9.0e-5 \
      --batch_size 32 \
      --hidden_dim 256 \
      --win 5 5 \
      --heter_n_layers 3 3 3 \
      --drop 0.32 \
      --shift_win 3 \
      --lambd 1.15 0.65 0.30 \
      --select_best_by dev \
      --refine_heads 4 \
      --dropedge 0.0 \
      --gcn_residual_drop 0.10 \
      --contrastive_weight 0.012 \
      --decouple_weight 0.002 \
      --warmup_epochs 8 \
      --supcon_temperature 0.070 \
      --label_smoothing 0.05 \
      --class_balance_beta 0.9993 \
      --hard_negative_weight 1.1 \
      --use_temporal_fusion \
      --temporal_logit_scale 0.30 \
      --temporal_aux_weight 0.06 \
      --data_path "$DATA_PATH" \
      "$@"
    ;;
  5)
    PORT="${PORT:-1732}"
    exec "$PYTHON_BIN" -u run.py \
      --gpu "$GPU" \
      --port "$PORT" \
      --classify emotion \
      --dataset CHSIMS5 \
      --textf_mode textf0 \
      --loss_type emo_sen_sft \
      --hidden_dim 256 \
      --heter_n_layers 3 3 3 \
      --refine_heads 4 \
      --dropedge 0.0 \
      --disable_tsne \
      --select_best_by dev \
      --data_path "$DATA_PATH" \
      --epochs 80 \
      --lr 9.0e-5 \
      --batch_size 32 \
      --win 5 5 \
      --drop 0.32 \
      --shift_win 3 \
      --lambd 1.15 0.65 0.30 \
      --gcn_residual_drop 0.10 \
      --contrastive_weight 0.012 \
      --decouple_weight 0.0020 \
      --warmup_epochs 8 \
      --supcon_temperature 0.070 \
      --label_smoothing 0.050 \
      --class_balance_beta 0.9993 \
      --hard_negative_weight 1.1 \
      --use_temporal_fusion \
      --temporal_logit_scale 0.30 \
      --temporal_aux_weight 0.06 \
      "$@"
    ;;
  *)
    echo "Usage: $0 {2|3|5} [extra args...]" >&2
    exit 1
    ;;
esac
