#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-results/moment_detr_gmr/best.ckpt}
SPLIT=${SPLIT:-test}
EVAL_PATH=${EVAL_PATH:-data/label/Standard/test.jsonl}
TEXT_FEAT_DIR=${TEXT_FEAT_DIR:-features/soccer_gmr/clip_text}
CLIP_FEAT_DIR=${CLIP_FEAT_DIR:-features/soccer_gmr/clip}
SLOWFAST_FEAT_DIR=${SLOWFAST_FEAT_DIR:-features/soccer_gmr/slowfast}
RESULTS_DIR=${RESULTS_DIR:-results/moment_detr_gmr/${SPLIT}}

python training/moment_detr_gmr/evaluate.py \
  --dataset soccer_gmr \
  --feature clip_slowfast \
  --model_path "${MODEL_PATH}" \
  --split "${SPLIT}" \
  --eval_path "${EVAL_PATH}" \
  --t_feat_dir "${TEXT_FEAT_DIR}" \
  --v_feat_dirs "${CLIP_FEAT_DIR}" "${SLOWFAST_FEAT_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  "$@"
