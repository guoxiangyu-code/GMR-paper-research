# 阶段一诊断报告：Soccer-GMR 上的 Decoder Query Collapse 实证

## 1. 实验目的与背景
根据《Beyond Caption-Based Queries for VMR》(CVPR 2026) 论文所述，Moment-DETR 家族在多目标视频检索（Multiple Moment Retrieval）中表现出一种被称为 **"Active Decoder-Query Collapse"** 的病理现象：无论视频中有多少个真实的 Ground-Truth (GT) moments，模型能够“激活”（置信度未衰减）的 query 数量始终保持恒定。当 GT 数量超过这个恒定值时，就会导致模型在多目标指标（如 `mR+@5`）上的全面崩盘。

本实验的目的是在我们的 Idea 1 基线模型（Slot-wise 门控双轨制）上，通过零训练成本的前向统计，验证该机制的真实存在性。

## 2. 实验设置
- **模型版本**：Idea 1 (Slot-wise FG + Noisy-OR Double-Track)
- **判定条件**：我们以每个 Decoder Slot 预测的前景概率为指标，如果该 slot 的 `slot_fg_prob > 0.05`，即判定该 query 为 **Active (已激活)**。这个较低的阈值对齐了论文中 IoU >= 0.1 的宽松口径，保证只过滤掉彻底失效（vanish）的死 query。
- **评测数据集**：Soccer-GMR `test` split（剔除空集查询，仅统计正样本）。

## 3. 评测结果：Active Query 数量统计

我们按 Ground-Truth moments 数量（1, 2, 3, >= 4）对测试集查询进行了分桶统计，结果如下：

| GT Moment 数 | 测试集样本数 (Count) | 平均激活 Query 数 (Active) ± 标准差 |
| :---: | :---: | :--- |
| **GT = 1** | 384 | **3.46 ± 1.31** |
| **GT = 2** | 128 | **4.92 ± 0.27** |
| **GT = 3** | 20 | **4.80 ± 0.40** |
| **GT >= 4** | 12 | **4.92 ± 0.28** |

## 4. 结论分析与 Motivation
从上表可以得出致命且清晰的结论：
1. **多目标坍缩机制的实锤**：当视频中的目标数量逐渐增加（从 2 到 3 再到 >=4）时，模型实际激活的 query 数量被“硬性锁死”在了 **4.9** 左右（总查询数为 10），**表现出极其平缓的水平线特征，完全不随目标数量的增加而增长**。
2. **Compute Budget 不足的量化**：对于 GT >= 4 的查询，模型能用的、有活力的 query 平均只有 4.92 个。如果还要考虑到同一个目标的冗余命中，真正能去覆盖不同目标的 query 数量远低于真实目标数。这导致大量目标从一开始就没有被分配到有效的 query，成为多目标召回（`mR+@5` 约 1.12%）近乎为 0 的根本原因。

### 下一步行动 (Stage 2: -SA + QD)
至此，“多目标召回崩盘”的问题已被成功升维并量化为**机制层面的 Query 坍缩**。接下来，我们将正式开始实施论文中的正解：
1. **-SA (Remove Self-Attention)**：切断 decoder 层的 self-attention，防止 query 相互“协商”后集体闭嘴，强迫每一个 query 独立行动。
2. **+QD (Query Dropout)**：在训练时以一定概率 mask 掉可学习的 query embedding，强迫监督信号分散给更多的 query 索引，彻底拉高 active query 的天花板数量。
3. **+NMS (Temporal NMS)**：为防止删掉 SA 后的同目标重叠检测，在最终输出前施加 NMS 后处理过滤冗余。
