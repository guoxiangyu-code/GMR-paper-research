# GMR Moment-DETR 系统性改进工作计划

> **项目**: Generalized Moment Retrieval (Soccer-GMR)
> **基线模型**: Moment-DETR-GMR (已复现, results/moment_detr_gmr/)
> **核心原则**: 不做大而全的pipeline; 系统 error analysis → 定位主要瓶颈 → 小而有效的模块

---

## 当前基线关键指标 (测试集, 1036 样本)

| 指标 | 值 (默认阈值0.4) | 值 (阈值0.55) | 论文 Moment-DETR-GMR | 论文 FlashVTG-GMR |
|------|:-:|:-:|:-:|:-:|
| AUROC | 70.00 | 70.00 | 72.09 | 74.00 |
| Rej-F1 | 0.00 | 67.15 | 64.01 | 61.72 |
| mAP | 8.09 | 8.09 | 7.52 | 24.62 |
| mR@5 | 14.14 | 14.14 | 12.96 | 33.36 |
| mR+@5 | 0.97 | 0.97 | 0.84 | 19.10 |
| G-mIoU@1 | 4.49 | **39.31** | 35.84 | 39.58 |

---

## Phase 1: 基线复现与对齐 ✅
**Status**: `complete`
**产出**: results/moment_detr_gmr/

### 已完成工作:
- [x] 训练 Moment-DETR-GMR (89 epoch, 早停于 Epoch 38)
- [x] 验证集评估: mAP=8.97, R1@0.5=11.37
- [x] 测试集推理: 生成 moment_detr_gmr_test_submission.jsonl
- [x] 官方评测: mAP=8.09, G-mIoU@1=4.49 (默认阈值), G-mIoU@1=39.31 (阈值0.55)
- [x] 与论文 baseline 对比分析 → 记录在 REPORT.md

### 关键发现:
- 定位能力略优于论文 Moment-DETR-GMR (mAP 8.09 > 7.52)
- 默认阈值0.4下 G-mIoU 严重退化 (所有 pred_exist_score > 0.4, 拒识门从未触发)
- 调整阈值到0.55后 G-mIoU@1=39.31, 超过论文的35.84
- mR+@5 极低 (0.97), 说明多时刻检索几乎完全失败

---

## Phase 2: 系统 Error Analysis
**Status**: `complete`
**目标**: 全面诊断模型的5类错误模式, 产出汇总表+关键分布图, 一句话指出最主要瓶颈
**预计产出**: `pipeline/error_analysis.py`, `results/moment_detr_gmr/error_analysis/`

### Task 2.1: 数据准备与基础设施搭建
**Status**: `not_started`
- [ ] **2.1.1** 编写分析脚本框架 `pipeline/error_analysis.py`
  - 加载 test.jsonl (GT) 和 moment_detr_gmr_test_submission.jsonl (预测)
  - 按样本类型分组: positive-single (384), positive-multi (160), negative/null-set (492)
  - 提取每个 query 的 pred_exist_score, pred_relevant_windows, GT relevant_windows
- [ ] **2.1.2** 实现 IoU 计算工具函数 (与 eval/metrics.py 对齐)
  - 单窗口 tIoU(pred, gt) 计算
  - 最优匹配 (Hungarian/greedy) 实现, 确保与官方评测一致
- [ ] **2.1.3** 设计输出目录结构
  ```
  results/moment_detr_gmr/error_analysis/
  ├── stats/               # 数值统计 JSON
  ├── figures/             # 可视化图表
  ├── cases/               # 典型案例 
  └── summary.md           # 汇总分析报告
  ```

### Task 2.2: 拒识-误报分析 (null-set 被接受)
**Status**: `not_started`
- [ ] **2.2.1** 统计 pred_exist_score 的整体分布
  - 分别画 positive 和 negative 样本的 pred_exist_score 直方图 (重叠)
  - 计算二者的均值、中位数、标准差
- [ ] **2.2.2** FP 曲线: 随 pred_exist_score 阈值 (0.0~1.0, 步长0.01) 变化的 FP (false positive) 数量曲线
  - 同时画 FN 曲线和 Rej-F1 曲线
  - 找到最优 Rej-F1 对应的阈值
- [ ] **2.2.3** 误判案例分析: 找出 pred_exist_score 最高的 Top-20 负样本
  - 记录 query 文本、vid、pred_exist_score
  - 与同 vid 中的正样本 query 对比, 挑出语义相近的负正对 (如 "a shot" vs "a missed shot")
  - 产出: stats/fp_analysis.json, figures/exist_score_distribution.png, figures/fp_fn_curve.png

### Task 2.3: 拒识-漏报分析 (positive 被错拒)
**Status**: `not_started`
- [ ] **2.3.1** 在不同阈值下统计 FN (positive 被判为 null) 的数量和比例
- [ ] **2.3.2** 分析 FN 对 mAP 的拖累
  - 计算: 仅在正确接受的正样本上的 mAP vs 全体正样本的 mAP
  - 量化 FN 导致的 mAP 损失
- [ ] **2.3.3** FN 案例分析: 找出 pred_exist_score 最低的 Top-20 正样本
  - 记录 query 文本、GT 时刻窗口、pred_exist_score
  - 分析是否存在模式 (如特定动作类型、短时刻等)
  - 产出: stats/fn_analysis.json, figures/fn_ratio_curve.png

### Task 2.4: 多时刻漏检分析 ("只命中第一个")
**Status**: `not_started`
- [ ] **2.4.1** 在 |GT|>=2 的 160 个多时刻 query 上:
  - 统计"第一个 moment 命中率" (Top-K 预测中至少一个与第一个 GT IoU >= 0.5)
  - 统计"后续 moments 命中率" (对应 mR+)
  - 量化"只命中第一个"的样本占比 (第一个命中但第二个及以后全部未命中)
- [ ] **2.4.2** 按 GT moment 数量 (2, 3, 4+) 细分统计
  - 各组的 mR@K 和 mR+@K
  - 各组的"首个命中率"和"后续命中率"
- [ ] **2.4.3** 分析预测数量 vs GT 数量的关系
  - 模型是否倾向于只输出少量预测 (pred 数量分布)
  - 多时刻 query 中预测数量中位数 vs GT 数量中位数
- [ ] **2.4.4** 多时刻漏检案例分析
  - 选取 5 个典型 "只命中第一个" 的样本
  - 记录所有 GT 窗口、预测窗口及 IoU
  - 产出: stats/multi_moment_analysis.json, figures/multi_moment_hit_rate.png

### Task 2.5: 多检分析 (|pred| > |GT|)
**Status**: `not_started`
- [ ] **2.5.1** 统计 |pred| > |GT| 的样本数量和比例
  - 按样本类型 (single, multi, null) 分类统计
- [ ] **2.5.2** 分析多余预测框的 confidence score 分布
  - 多余框的得分 vs 命中框的得分对比
  - 是否存在可通过分数阈值过滤的简单策略
- [ ] **2.5.3** 多余框的时间位置分析
  - 多余框与 GT 框的时间距离分布
  - 是否集中在 GT 附近 (近邻误检) 还是随机分布
  - 产出: stats/over_detection_analysis.json, figures/extra_pred_score_dist.png

### Task 2.6: 边界不准分析
**Status**: `not_started`
- [ ] **2.6.1** 对所有匹配对 (pred, GT) 计算 IoU, 绘制 IoU 直方图
  - 使用贪心匹配 (按 IoU 从高到低匹配)
  - 统计 IoU 分布: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
- [ ] **2.6.2** 统计"差一点"样本 (0.3 <= IoU < 0.5) 的占比
  - 这些是潜在可通过 boundary refinement 挽救的样本
- [ ] **2.6.3** 边界偏差方向分析
  - 计算 start 偏移 (pred_start - gt_start) 和 end 偏移 (pred_end - gt_end) 分布
  - 判断是否存在系统性偏移 (如总是预测过长/过短)
- [ ] **2.6.4** 按视频时长和 GT 时刻时长分组分析
  - 短时刻 (<5s) vs 中时刻 (5-20s) vs 长时刻 (>20s) 的 IoU 分布差异
  - 产出: stats/boundary_analysis.json, figures/iou_histogram.png, figures/boundary_offset.png

### Task 2.7: 汇总报告
**Status**: `not_started`
- [ ] **2.7.1** 生成错误类型汇总表
  ```
  | 错误类型 | 影响样本数 | 影响比例 | 对 G-mIoU@1 的影响 (估算) |
  |----------|-----------|---------|-------------------------|
  | 拒识-误报 | ? | ? | ? |
  | 拒识-漏报 | ? | ? | ? |
  | 多时刻漏检 | ? | ? | ? |
  | 多检 | ? | ? | ? |
  | 边界不准 | ? | ? | ? |
  ```
- [ ] **2.7.2** 生成关键分布图 (合并为一张多子图 figure)
- [ ] **2.7.3** 一句话结论: 指出当前最主要的瓶颈
- [ ] **2.7.4** 产出完整的 error_analysis/summary.md

---

## Phase 3: 反事实 (Oracle) 修复排序
**Status**: `complete`
**目标**: 依次只修复某一类错误, 其余不变, 重跑 eval, 按指标增益排序
**产出**: `pipeline/oracle_fix.py`, `results/moment_detr_gmr/oracle_analysis/`

### Phase 3 核心结论

| Fix Type | G-mIoU@1 | Gain | mAP | Gain | Rej-F1 | Gain |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 39.31 | — | 8.09 | — | 67.15 | — |
| **Fix Over-detection** 🏆 | **58.23** | **+18.92** | 20.64 | +12.55 | 67.15 | 0 |
| Fix FP | 51.18 | +11.87 | 8.09 | 0 | 80.52 | +13.37 |
| Fix Boundary | 44.43 | +5.12 | **38.83** | **+30.74** | 67.15 | 0 |
| Fix Multi-miss | 39.31 | 0 | 8.09 | 0 | 67.15 | 0 |
| Fix All (上界) | 91.66 | +52.35 | 100.0 | +91.91 | 100.0 | +32.85 |

**关键洞察**:
- **过检 (Over-detection)** 是 G-mIoU 增益最大的单类修复 (+18.92)
- **边界不准** 对 mAP 影响最大 (+30.74), 说明已命中的 pred 边界质量差
- **Fix Multi-miss = 0 增益**: oracle 添加 GT 时刻没有提升 G-mIoU! 原因: G-mIoU 使用 set-level IoU, 而 multi-miss 的主因是 **pred 窗口多而乱** 导致 set IoU 分母膨胀, 单纯加 GT 不够——需要同时改善 pred 质量
- **Fix FP** 提升显著, 说明 null-set 拒识仍有空间改进

### Task 3.1: 构建 Oracle 修复框架
**Status**: `not_started`
- [ ] **3.1.1** 编写 `pipeline/oracle_fix.py`
  - 读取原始预测文件 moment_detr_gmr_test_submission.jsonl
  - 读取 GT 文件 data/label/Standard/test.jsonl
  - 对预测文件进行指定类型的 oracle 修改
  - 保存修改后的预测文件到临时目录
  - 调用 eval/eval_main.py 重跑评测
  - 记录 G-mIoU@1, mAP, Rej-F1 三个关键指标的变化

### Task 3.2: Oracle 修复 —— 修复误报 (Fix FP)
**Status**: `not_started`
- [ ] **3.2.1** 实现: 对所有真正的 null-set query, 将 pred_exist_score 设为 0 (或清空 pred_relevant_windows)
  - 即: 假设 oracle 完美拒识所有负样本
- [ ] **3.2.2** 重跑 eval, 记录 G-mIoU@1 / mAP / Rej-F1 变化
- [ ] **3.2.3** 分析: 拒识完美化对综合指标的提升幅度

### Task 3.3: Oracle 修复 —— 修复多时刻漏检 (Fix Multi-moment Miss)
**Status**: `not_started`
- [ ] **3.3.1** 实现: 对 |GT|>=2 的 query, 把漏掉的后续 GT moment 作为额外预测添加到 pred_relevant_windows 中
  - 具体: 对每个未被任何 pred 命中 (IoU<0.5) 的 GT moment, 直接加入预测列表
- [ ] **3.3.2** 重跑 eval, 记录指标变化
- [ ] **3.3.3** 分析: 多时刻完美召回对 mR+, mAP, G-mIoU 的提升

### Task 3.4: Oracle 修复 —— 修复边界 (Fix Boundary)
**Status**: `not_started`
- [ ] **3.4.1** 实现: 对所有已匹配的 (pred, GT) 对, 将 pred 的边界替换为 GT 的精确边界
  - 即: 把所有匹配对的 IoU 拉到 1.0
  - 未匹配的 pred 保持不变
- [ ] **3.4.2** 重跑 eval, 记录指标变化
- [ ] **3.4.3** 分析: 精确边界对 mAP 和 mIoU 的提升

### Task 3.5: Oracle 修复 —— 修复多检 (Fix Over-detection)
**Status**: `not_started`
- [ ] **3.5.1** 实现: 对所有样本, 删除未与任何 GT 匹配的多余预测框
  - 仅保留与 GT 匹配的预测 (IoU >= 某阈值, 如 0.3)
  - null-set 样本: 清空所有预测
- [ ] **3.5.2** 重跑 eval, 记录指标变化
- [ ] **3.5.3** 分析: 消除多余预测对 mAP 的影响

### Task 3.6: Oracle 组合修复 (Optional)
**Status**: `not_started`
- [ ] **3.6.1** 同时修复收益最大的两类错误, 观察是否有叠加效果
- [ ] **3.6.2** 全部修复 (upper bound 分析)

### Task 3.7: 指标增益排序与方向选择
**Status**: `not_started`
- [ ] **3.7.1** 生成排序表
  ```
  | Oracle 修复类型 | G-mIoU@1 增益 | mAP 增益 | Rej-F1 增益 | 排名 |
  |----------------|:------------:|:-------:|:---------:|:----:|
  | 修复误报 | ? | ? | ? | ? |
  | 修复多时刻漏检 | ? | ? | ? | ? |
  | 修复边界 | ? | ? | ? | ? |
  | 修复多检 | ? | ? | ? | ? |
  ```
- [ ] **3.7.2** 按增益选出排名第一的切入点
- [ ] **3.7.3** 确定改进方向并说明理由:
  - 多时刻漏检 → coverage-aware retrieval
  - 误报为主 → candidate verification
  - 边界偏移 → boundary refinement
- [ ] **3.7.4** 产出完整的 oracle_analysis/summary.md

---

## Phase 4: 方案设计与实施
**Status**: `in_progress`
**目标**: 基于 Phase 3 结论, 实现「NMS + 置信度过滤」后处理模块, 减少过检和噪声预测
**改进方向**: Over-detection 抑制 → NMS / score thresholding post-processing

### 背景分析

Phase 3 显示 Fix Over-detection 的 G-mIoU@1 增益最大 (+18.92):
- 原因: 模型对每个 query 固定输出 10 个预测框, 正样本 100% 过检
- 多余框的平均分数 (0.447) 显著低于命中框 (0.556), 可通过分数阈值过滤
- G-mIoU 的 set-level IoU 公式: `sum(matched_IoU) / (|pred|+|GT|-|matches|)` — 分母中 |pred| 越大, 得分越低
- Fix Boundary 对 mAP 贡献最大 (+30.74), 说明边界回归也需要改进

### Task 4.1: 预测后处理 — 置信度过滤 + soft-NMS
**Status**: `not_started`
- [ ] **4.1.1** 分析 pred_windows 的置信度分数分布
  - 在 positive 样本上: 画「命中框分数」vs「未命中框分数」的 CDF
  - 找到最优过滤阈值 (在 val 集上 sweep, 目标: G-mIoU@1)
- [ ] **4.1.2** 实现 `pipeline/postprocess.py`
  - 基于置信度阈值过滤低分预测框
  - 实现时序 Soft-NMS (抑制高度重叠的冗余框)
  - 参数: `score_threshold`, `nms_threshold`, `max_preds_per_query`
- [ ] **4.1.3** 在测试集上评估并对比
  - 记录 G-mIoU@1, mAP, mR@5, mR+@5 变化
  - 绘制 score_threshold vs G-mIoU@1 曲线 (类似 exist_score 阈值曲线)

### Task 4.2: 存在性得分校准 (Calibration)
**Status**: `not_started`
- [ ] **4.2.1** 分析 pred_exist_score 的校准问题
  - 画 calibration curve (reliability diagram): predicted score vs actual positive rate
  - 正样本均值=0.70, 负样本均值=0.52, 差距不够大
- [ ] **4.2.2** 温度缩放 (Temperature Scaling)
  - 在 val 集上搜索最优温度 T, 使 calibrated_score = sigmoid(logit/T) 分布更好分离
  - 评估校准后 AUROC 和 Rej-F1 的变化
- [ ] **4.2.3** 组合校准 + 过滤的整体后处理流水线
  - 先校准 exist_score, 再用 score_threshold 过滤预测框
  - 整体评估 G-mIoU@1, mAP 等指标

### Task 4.3: (可选) 边界精化
**Status**: `not_started`
- [ ] **4.3.1** 分析边界偏移模式
  - 平均 Δstart=+7.42s, Δend=+8.16s — 预测整体比 GT 靠后
  - 按视频类型/动作类型分组分析是否有规律
- [ ] **4.3.2** 如果规律明显, 实现简单的偏移校正 (加减常数)
  - 在 val 集上搜索最优偏移量
  - 这不是主要瓶颈, 仅作为辅助提升

### Task 4.4: 综合评估与论文写作准备
**Status**: `not_started`
- [ ] **4.4.1** 综合所有后处理改进, 得到最终系统
- [ ] **4.4.2** 与所有 baseline 对比 (含论文 FlashVTG-GMR)
- [ ] **4.4.3** 撰写改进方案说明

---

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| (暂无) | - | - |

---

## 关键文件索引
| 文件/目录 | 用途 |
|-----------|------|
| think.md | 总原则和工作规划清单 |
| eval/eval_main.py | 官方评测脚本 |
| eval/metrics.py | 评测指标实现 |
| data/label/Standard/test.jsonl | 测试集 GT |
| results/moment_detr_gmr/test/moment_detr_gmr_test_submission.jsonl | 测试集预测 |
| results/moment_detr_gmr/test/test_results.json | 默认阈值评测结果 |
| results/moment_detr_gmr/test/test_results_opt2.json | 阈值0.55评测结果 |
| results/moment_detr_gmr/REPORT.md | 基线复现报告 |
| scripts/infer_moment_detr_gmr.sh | 推理脚本 |
| scripts/train_moment_detr_gmr.sh | 训练脚本 |
