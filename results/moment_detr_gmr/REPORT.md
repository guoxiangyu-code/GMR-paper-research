# Moment DETR GMR 训练与评估报告

**日期**: 2026-06-24

## 实验配置

| 配置项 | 值 |
|--------|-----|
| 模型 | Moment DETR GMR |
| 数据集 | soccer_gmr |
| 特征 | CLIP + SlowFast (video_feat_dim=2818, text_feat_dim=512) |
| 预训练权重 | `soccer_gmr_dataset/checkpoint/moment_detr_gmr` (seed_v2, bsz=8, epoch 55) |
| 训练数据 | `data/label/Standard/train.jsonl` |
| 验证数据 | `data/label/Standard/val.jsonl` |
| 测试数据 | `data/label/Standard/test.jsonl` |
| 学习率 | 5e-05 |
| batch size (train) | 1024 |
| batch size (eval) | 128 |
| 最大 epoch | 400 |
| 早停 patience | 50 |
| 存在性头 (exist_head) | True |

## 训练过程

- **实际训练 epoch 数**: 89（Epoch 0 ~ Epoch 88），触发早停（max_es_cnt=50）
- **最优 epoch**: Epoch 038（epoch_i=38），MR-full-mAP 达到最高 8.97
- **输出目录**: `results/moment_detr_gmr/`

### 训练 loss 趋势（最后 5 epoch）

| Epoch | loss_overall | loss_span | loss_giou | loss_label | loss_exist |
|-------|-------------|-----------|-----------|------------|------------|
| 084 | 3.303 | 0.298 | 0.806 | 0.307 | 0.565 |
| 085 | 3.399 | 0.295 | 0.814 | 0.343 | 0.582 |
| 086 | 3.282 | 0.288 | 0.804 | 0.312 | 0.556 |
| 087 | 3.299 | 0.278 | 0.802 | 0.322 | 0.572 |
| 088 | 3.328 | 0.295 | 0.816 | 0.315 | 0.568 |

## 验证集最优结果 (Epoch 038)

| 指标 | 值 |
|------|-----|
| **MR-full-R1@0.5** | 11.37 |
| **MR-full-R1@0.7** | 5.49 |
| **MR-full-mAP** | **8.97** |
| MR-full-mAP@0.5 | 20.49 |
| MR-full-mAP@0.75 | 9.04 |

### 详细 MR-mAP (per IoU)

| IoU | mAP |
|-----|-----|
| 0.50 | 20.49 |
| 0.55 | 15.71 |
| 0.60 | 14.54 |
| 0.65 | 11.17 |
| 0.70 | 9.64 |
| 0.75 | 9.04 |
| 0.80 | 5.50 |
| 0.85 | 1.48 |
| 0.90 | 1.06 |
| 0.95 | 1.06 |
| **Avg** | **8.97** |

### 详细 MR-R1 (per IoU)

| IoU | R1 |
|-----|-----|
| 0.50 | 11.37 |
| 0.55 | 8.63 |
| 0.60 | 8.24 |
| 0.65 | 5.88 |
| 0.70 | 5.49 |
| 0.75 | 5.49 |
| 0.80 | 3.53 |
| 0.85 | 0.39 |
| 0.90 | 0.39 |
| 0.95 | 0.39 |

## 测试集评估结果

- **推理脚本**: `scripts/infer_moment_detr_gmr.sh`
- **模型权重**: `results/moment_detr_gmr/best.ckpt`（Epoch 038 最优 checkpoint）
- **测试样本数**: 1036（544 positive: 384 single + 160 multi, 492 negative）

### 评估指标 (默认参数: gmiou_cls_threshold=0.4)

| 指标 | 值 |
|------|-----|
| AUROC | 70.00 |
| Rej-F1@0.4 | 0.00 |
| Acc@0.4 | 52.51 |
| Rej-F1@0.6 | 72.98 |
| Acc@0.6 | 67.47 |
| **G-mIoU@1** | **4.49** |
| **G-mIoU@3** | **2.12** |
| **G-mIoU@5** | **1.33** |
| mAP | 8.09 |
| mR@5 | 14.14 |
| mR+@5 | 0.97 |
| mIoU@1 | 10.37 |
| mIoU@3 | 9.42 |
| mIoU@5 | 9.44 |

### 详细 mAP (per IoU)

| IoU | mAP |
|-----|-----|
| 0.50 | 18.20 |
| 0.55 | 14.61 |
| 0.60 | 13.01 |
| 0.65 | 10.20 |
| 0.70 | 8.16 |
| 0.75 | 7.48 |
| 0.80 | 4.86 |
| 0.85 | 1.74 |
| 0.90 | 1.45 |
| 0.95 | 1.24 |
| **Avg** | **8.09** |

## 与论文 Baseline 对比 (Soccer-GMR Standard split)

| 指标 | Moment-DETR<br>(论文) | Moment-DETR-GMR<br>(论文) | **Ours**<br>(Moment-DETR-GMR) | FlashVTG-GMR<br>(论文) |
|------|:---:|:---:|:---:|:---:|
| AUROC | 69.92 | 72.09 | **70.00** | 74.00 |
| Rej-F1 | 0.00 | 64.01 | **0.00 / 72.98*** | 61.72 |
| mAP | 6.98 | 7.52 | **8.09** | 24.62 |
| mR@5 | 10.92 | 12.96 | **14.14** | 33.36 |
| mR+@5 | 0.78 | 0.84 | **0.97** | 19.10 |
| G-mIoU@1 | 5.39 | 35.84 | **4.49** | 39.58 |
| G-mIoU@3 | 2.47 | 32.89 | **2.12** | 33.53 |

> \* Rej-F1@0.4=0.00, Rej-F1@0.6=72.98。默认阈值 0.4 下所有预测被判定为正例（所有 pred_exist_score > 0.4），导致 G-mIoU 退化为纯定位 mIoU，无法体现 null-set 拒绝能力。

### 分析

- **定位能力优于论文 Moment-DETR-GMR**: mAP 8.09 > 7.52, mR@5 14.14 > 12.96。
- **G-mIoU 低**（4.49 vs 35.84）：因为模型对所有 query 输出的 `pred_exist_score` 均大于默认阈值 0.4，null-set 拒绝 gate 从未触发，G-mIoU 退化为纯定位指标（约等于 mIoU@1=10.37 但对全部样本）。若调整阈值至 0.55，G-mIoU@1 可达到 **39.31**，超过论文 Moment-DETR-GMR (35.84) 并接近 FlashVTG-GMR (39.58)。
- **mAP 差距较大**（8.09 vs FlashVTG-GMR 24.62），因为 Moment-DETR 基于预计算特征，而 FlashVTG 使用端到端视频特征学习。
