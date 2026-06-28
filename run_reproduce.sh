#!/bin/bash
set -euo pipefail
GPU_ID=0
RES_DIR="results/moment_detr_gmr"

echo "Starting reproduction of pure Idea 1 (Dual-Track)..."

CUDA_VISIBLE_DEVICES=${GPU_ID} ./scripts/train_moment_detr_gmr.sh \
    --bsz 2048 --eval_bsz 2048 --n_epoch 200

echo "Training finished. Running inference..."
CUDA_VISIBLE_DEVICES=${GPU_ID} MODEL_PATH=${RES_DIR}/best.ckpt SPLIT=test RESULTS_DIR=${RES_DIR} \
    ./scripts/infer_moment_detr_gmr.sh

echo "Evaluating..."
python eval/eval_main.py \
    --submission_path ${RES_DIR}/moment_detr_gmr_test_submission.jsonl \
    --gt_path data/label/Standard/test.jsonl \
    --save_path ${RES_DIR}/test_metrics.json \
    --gmiou_cls_threshold 0.6

echo "Reproduction complete!"
