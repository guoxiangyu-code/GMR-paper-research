# 工作进度日志

> **项目**: GMR Moment-DETR 系统性改进
> **开始日期**: 2026-06-24

---

## Session 1 — 2026-06-24 规划与基线确认

### 完成的工作
1. **Phase 1 已完成 (基线复现)**
   - 训练 Moment-DETR-GMR (89 epoch, 早停 Epoch 38)
   - 测试集推理并评测
   - 关键指标: mAP=8.09, G-mIoU@1=4.49/39.31 (阈值0.4/0.55)
   - 详细结果记录在 results/moment_detr_gmr/REPORT.md

2. **创建工作规划**
   - 阅读 think.md, 理解三步走战略
   - 分析已有基线结果 (test_results.json, test_results_opt.json, test_results_opt2.json)
   - 创建详细的 task_plan.md (Phase 1-4)
   - 创建 findings.md 记录初步发现

### 关键发现
- 阈值问题是 G-mIoU 退化的直接原因 (pred_exist_score 全部 > 0.4)
- mR+@5=0.97 表明多时刻检索几乎完全失败 → 潜在大瓶颈
- 需要 Phase 2 的定量分析来确认哪个才是收益最大的改进方向

### 当前状态
- **Phase 1**: ✅ complete
- **Phase 2**: ⬜ not_started — 下一步: Task 2.1 (数据准备与基础设施搭建)
- **Phase 3**: ⬜ not_started — 阻塞于 Phase 2
- **Phase 4**: ⬜ blocked — 阻塞于 Phase 3

### 下一步行动
1. 开始 Phase 2: 编写 pipeline/error_analysis.py
2. 先做 Task 2.1 (数据准备), 然后并行推进 2.2~2.6 的分析
3. 产出汇总报告后进入 Phase 3

### 文件变更记录
| 操作 | 文件 |
|------|------|
| 创建 | task_plan.md |
| 创建 | findings.md |
| 创建 | progress.md |
| 已有 | results/moment_detr_gmr/REPORT.md |
| 已有 | results/moment_detr_gmr/test/test_results*.json |

---

## Session 2 — 2026-06-24 Phase 4 后处理实现与评估

### 完成的工作
1. **明确评测协议**: 阅读了 README 和 `说明.md`，明确了 val 集（无负样本）和 test 集（有负样本）的分工，并创建了 `要注意的说明.md`。
2. **Phase 4 实施与测试**:
   - 编写并执行了 `pipeline/postprocess.py`
   - 对 Score Thresholding 和 Soft-NMS 进行了全面的 Parameter Sweep。

### 当前状态
- **Phase 4**: ✅ complete

### 下一步行动
- 综合评估已完成。我们的单点后处理修复已经获得了远超预期的成功。目前的改进可以作为一份坚实的分析报告和系统化改进成果。
- 如果需要进一步提升定位精度（mAP 目前依然不高），可以考虑回到 Phase 4 中被搁置的“边界精化 (Boundary Refinement)”或者开始尝试特征层面的更换（如替换为 FlashVTG）。

---

## Session 3 — 2026-06-24 Phase 3 结论二次归因与验证

### 完成的工作
1. **核实 G-mIoU 截断机制**: 检查 `eval/metrics.py`，确认了 G-mIoU@k 是严格按照预测得分的前 k 个框（`[:k]`）计算的。
2. **在 G-mIoU@3 口径下重审 Oracle**: 
   - 编写 `pipeline/oracle_fix_v2.py` 重跑 Phase 3。
   - `Fix Over-detection` 在 `@3` 口径下增益达到了更恐怖的 **+21.31**（@1为+18.92），彻底坐实了过检的废框严重摊薄分母的结论，而非仅仅是截断效应带来的重排假象。
   - `Fix Multi-miss` 即使在 `@3` 下依然毫无增益（0.00），从结构上被“判死”。
3. **补充 Fix FN 分析**:
   - 强制接受被漏报（FN）的正样本后，G-mIoU 几乎无增益（@1仅+0.80），但 Rej-F1 大幅上涨 +18.56。
   - 这说明分类阈值的确存在误伤正样本的问题（FN），但因为正样本自身的定位质量极差（过检严重），这些漏报样本被召回后依然无法有效提升系统的整体 G-mIoU。

### 关键发现
- 所有数据完美闭环，钉死了当前最大的瓶颈：**存在性打分偏移带来的误报（FP）** 以及 **严重的框过检（Over-detection）**。

### 文件变更记录
| 操作 | 文件 |
|------|------|
| 修改 | think.md (追加@3 Oracle结果及分析) |
| 修改 | task_plan.md (更新 Phase 3 表格与结论) |
| 修改 | findings.md (更新 Section 3.1) |
| 修改 | progress.md (添加 Session 3 记录) |

---

## Session 4 — 2026-06-24 整理测试指令文档

### 完成的工作
1. **响应用户需求**: 根据 Planning with Files 的规则，梳理了项目中所有关于评测、错误分析的脚本。
2. **编写文档**: 创建了 `test.md`。详细记录了每条运行指令的作用以及最终产生的输出文件种类、各项指标的含义（从生成预测、跑官方指标、拆解5大错误类型，到反事实诊断排序和最后一步的后处理参数搜寻）。

### 当前状态
- **规划文档**: 已就绪。

### 文件变更记录
| 操作 | 文件 |
|------|------|
| 创建 | test.md |
| 修改 | progress.md (添加 Session 4 记录) |

## Phase 5: 验证鲁棒性 (Robustness Verification)
- Added Phase 5 to task_plan.md
- Starting Step 0: evaluate official example_test_submission.jsonl at 0.4, 0.6, 0.8


- Completed Step 1: Plotted and compared pred_exist_score distributions.
- Checked Step 2 & 3: Identified threshold convention differences (> vs >=) and soft gate scaling effects.
- Checked Step 4: Data splits match (shared=1036, gt_only=0).
- Conclusion: The Official submission demonstrates G-mIoU@1 is ~35 at 0.4, meaning the architecture is fundamentally capable of rejecting at 0.4. The collapse at 0.4 in the reproduced version is solely due to the drift of the model's scores to a much higher mean, completely skipping the 0.4 threshold.


- Completed Phase 5: Generated 差异对照表.md with full comparison of distributions and thresholds.
- Findings successfully resolve the robustness validation plan.

