# Error Analysis Summary — Moment-DETR-GMR Test Set

> **Dataset**: Soccer-GMR Standard Test | 1036 queries (544 positive: 160 multi + 384 single | 492 negative)

---

## Error Type Breakdown

| # | Error Type | Affected Samples | Rate | Key Stat |
|---|-----------|:----------------:|:----:|---------|
| 2.2 | **Rejection FP** (null-set accepted) | 492 (τ=0.4) / 123 (τ=0.55) | 100% / 25% (of negatives) | Best Rej-F1=75.1% at τ=0.72 |
| 2.3 | **Rejection FN** (positive rejected) | 238 (τ=0.55) | 43.8% (of positives) | 131 single + 107 multi |
| 2.4 | **Multi-moment miss** (only 1st retrieved) | 23/160 multi queries | 14.4% | mR+@5=0.97% vs mR@5=14.14% |
| 2.5 | **Over-detection** (excess preds) | 544/544 positives | 100.0% | Extra pred score mean=0.447 |
| 2.6 | **Boundary inaccuracy** (near-miss 0.3≤IoU<0.5) | 47 matched pairs | 6.3% | Mean IoU=0.092, Δstart=+7.42s, Δend=+8.16s |

---

## Score Distribution (pred_exist_score)

| | Mean | Median | Std |
|---|---:|---:|---:|
| Positive queries | 0.7016 | 0.5705 | 0.2285 |
| Negative queries | 0.5186 | 0.5144 | 0.0540 |

---

## Multi-moment Analysis (160 queries with |GT|≥2)

| Metric | Count | Rate |
|-------|------:|-----:|
| Any moment hit @k=5 | 59 | 36.9% |
| First moment hit @k=5 | 30 | 18.8% |
| ONLY first hit (miss rest) | 23 | 14.4% |
| Has subsequent hits | 36 | 22.5% |

---

## 🔍 主要瓶颈分析 (待 Phase 3 Oracle 修复后确认)

基于当前 error analysis 数据的初步判断:

1. **拒识误报 (FP)** — 所有492个负样本在默认阈值下被误接受, 直接导致 G-mIoU@1 从39.31降至4.49 (35分差距)。这是 G-mIoU 退化的**直接原因**, 但通过调阈值即可部分修复, 属于 calibration 问题。

2. **多时刻漏检 (Multi-miss)** — mR+@5仅0.97%, 与mR@5=14.14的巨大差距说明模型几乎完全无法检索多时刻。在160个多时刻查询中, 仅36个(22.5%)命中了后续时刻。这是**定位层面**的核心瓶颈, 且与 FlashVTG-GMR 的差距最大 (mR+@5: 0.97 vs 19.10)。

3. **边界不准** — 平均 IoU 较低, 存在一定比例的近失样本可通过 refinement 提升。

**初步结论**: 多时刻漏检是需要模型层面改进的最大瓶颈 (mR+差距18倍); 拒识误报虽然对 G-mIoU 数值影响最大, 但阈值调整即可修复。需等待 Phase 3 Oracle 排序来量化各类错误对指标的贡献。

---

## 图表索引

| 图表 | 路径 |
|------|------|
| Existence score分布 + FP/FN曲线 | figures/exist_score_distribution.png |
| FN/FP率随阈值变化 | figures/fn_ratio_curve.png |
| 多时刻命中率 | figures/multi_moment_hit_rate.png |
| 多余预测分数分布 | figures/extra_pred_score_dist.png |
| IoU直方图 + 边界偏移 | figures/iou_histogram.png |
| 边界偏移散点图 | figures/boundary_offset.png |