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
