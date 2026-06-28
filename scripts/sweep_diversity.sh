#!/usr/bin/env bash
set -euo pipefail

# 扫描脚本: 自动跑不同的 div_coef, 并生成所有的图表和指标
# 用法: ./scripts/sweep_diversity.sh

GPU_ID=$1
shift
COEFS=("$@")
MARGIN=0.5
BSZ=2048

# 保存对比表格
SUMMARY_FILE="experiments/stage3_sweep_summary.md"
mkdir -p experiments/stage3_sweep_summary
if [ ! -f ${SUMMARY_FILE} ]; then
    echo "| div_coef | mAP | mR@5 | mR+@5 | G-mIoU@1 | Rej-F1@0.6 | GT 覆盖率 | Active (GT=2) |" > ${SUMMARY_FILE}
    echo "|---|---|---|---|---|---|---|---|" >> ${SUMMARY_FILE}
fi

for COEF in "${COEFS[@]}"; do
    echo "=========================================================="
    echo "▶▶▶ 开始在 GPU ${GPU_ID} 跑 div_coef=${COEF} 的实验..."
    echo "=========================================================="
    
    EXP_DIR="experiments/20260627_stage3_coef${COEF}"
    RES_DIR="${EXP_DIR}/results"
    
    # 1. 训练
    CUDA_VISIBLE_DEVICES=${GPU_ID} ./scripts/train_moment_detr_gmr.sh \
        --use_sa True --query_dropout 0.25 \
        --use_diversity True --div_coef ${COEF} --div_margin ${MARGIN} \
        --bsz ${BSZ} --eval_bsz ${BSZ} --n_epoch 400 \
        --results_dir ${RES_DIR}
        
    # 2. 推理 (带NMS) 生成指标
    CUDA_VISIBLE_DEVICES=${GPU_ID} MODEL_PATH=${RES_DIR}/best.ckpt SPLIT=test RESULTS_DIR=${RES_DIR} \
        ./scripts/infer_moment_detr_gmr.sh \
        --use_sa True --query_dropout 0.0 \
        --use_nms --nms_thr 0.5 \
        --use_diversity True --div_coef ${COEF} --div_margin ${MARGIN}
        
    python eval/eval_main.py \
        --submission_path ${RES_DIR}/moment_detr_gmr_test_submission.jsonl \
        --gt_path data/label/Standard/test.jsonl \
        --save_path ${RES_DIR}/test_metrics.json \
        --gmiou_cls_threshold 0.6
        
    # 3. 诊断 Active Queries
    CUDA_VISIBLE_DEVICES=${GPU_ID} python diagnose.py \
        --model_path ${RES_DIR}/best.ckpt \
        --split test \
        --eval_path data/label/Standard/test.jsonl \
        --use_sa True --query_dropout 0.0
    
    # 将 diagnose 生成的文件移到实验目录
    mkdir -p ${EXP_DIR}/diag
    mv experiments/diag/active_vs_moment.npy ${EXP_DIR}/diag/
    
    # 4. 推理 (无NMS) 拿 prediction dump 用来画图
    CUDA_VISIBLE_DEVICES=${GPU_ID} MODEL_PATH=${RES_DIR}/best.ckpt SPLIT=test RESULTS_DIR=${RES_DIR} \
        ./scripts/infer_moment_detr_gmr.sh \
        --use_sa True --query_dropout 0.0 \
        --use_diversity True --div_coef ${COEF} --div_margin ${MARGIN}
        
    # 5. 画 GT 覆盖率图
    mkdir -p ${EXP_DIR}/span_viz
    python experiments/20260627_stage2_run1/plot_query_span.py \
        --dump ${RES_DIR}/test_pred_dump.pt \
        --out_dir ${EXP_DIR}/span_viz \
        --tau 0.05 --max_samples 24 > ${EXP_DIR}/span_viz/coverage_stats.txt 2>&1
        
    # 提取关键指标
    mAP=$(jq -r '.brief."mAP"' ${RES_DIR}/test_metrics.json)
    mR5=$(jq -r '.brief."mR@5"' ${RES_DIR}/test_metrics.json)
    mRp5=$(jq -r '.brief."mR+@5"' ${RES_DIR}/test_metrics.json)
    G_mIoU1=$(jq -r '.brief."G-mIoU@1"' ${RES_DIR}/test_metrics.json)
    RejF1=$(jq -r '.brief."Rej-F1@0.6"' ${RES_DIR}/test_metrics.json)
    
    cov=$(cat ${EXP_DIR}/span_viz/coverage_stats.txt | grep "平均 GT 覆盖率" | awk -F'= ' '{print $2}' | awk '{print $1}')
    
    # 提取 GT=2 时的 active 数（这里用一个小 python script 快速读一下）
    ACTIVE2=$(python -c "import numpy as np; d=np.load('${EXP_DIR}/diag/active_vs_moment.npy', allow_pickle=True).item(); print(f'{np.mean(d.get(2, [0])): .2f}' if len(d.get(2, [])) > 0 else '0.00')")
    
    # 追加到表格
    echo "| ${COEF} | ${mAP} | ${mR5} | ${mRp5} | ${G_mIoU1} | ${RejF1} | ${cov} | ${ACTIVE2} |" >> ${SUMMARY_FILE}
    
    echo "=========================================================="
    echo "✓ div_coef=${COEF} 已完成! 核心指标: mR+@5=${mRp5}, 覆盖率=${cov}"
    echo "=========================================================="
done

echo "🎉 所有扫描完成! 对比结果已保存在 ${SUMMARY_FILE}"
cat ${SUMMARY_FILE}
