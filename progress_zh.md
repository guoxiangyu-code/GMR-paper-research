# 进度日志 — 广义时刻检索错误分析

## 会话 1 — 2026-06-23：初始仓库探索与规划

### 已完成操作
- [x] 阅读 `todo.md` — 理解 5 提示工作流（复现 → 错误分析 → 量化 → 反事实 → 实现）
- [x] 通过 3 个并发研究子代理探索完整仓库结构
- [x] 映射所有目录：`models/`，`training/`，`eval/`，`data/`，`features/`，`results/`，`scripts/`，`configs/`，`docs/`，`soccer_gmr_dataset/`
- [x] 验证 JSONL 标签格式（正样本/负样本/多时刻样本）
- [x] 统计数据划分：标准集 训练=4,138，验证=465，测试=1,036；全集 训练=16,898，验证=2,235，测试=2,986
- [x] 验证特征数量：2,288 个 CLIP 视觉，6,952 个 CLIP 文本，2,288 个 SlowFast
- [x] 阅读模型架构：Moment-DETR + GMR 适配器（通过最大池化解码器查询 → 2 层 MLP 的存在性预测头）
- [x] 阅读评估代码：`eval_main.py` + `metrics.py`，含贪心匹配，8 种指标类型
- [x] 阅读训练/推理脚本及配置层级
- [x] 发现两个结果目录：`moment_detr_gmr/`（本地训练）和 `moment_detr_gmr_official/`（论文结果）
- [x] 找到现有测试预测（1,035 行）和验证预测（464 行）
- [x] 回顾验证集指标：显著低于论文测试集指标（mAP 8.14 vs 18.69）
- [x] 创建 `task_plan.md`，`findings.md`，`progress.md`

### 关键发现
1. **存在两个检查点** — `results/moment_detr_gmr/best.ckpt`（本地）vs `results/moment_detr_gmr_official/model_best.ckpt`（官方）。需要验证哪个产生了论文报告的测试指标。
2. **验证集 mR+@5 = 0.5**（接近零）— 多时刻检索在验证集上严重不足；这与论文认定多时刻为核心挑战的结论一致。
3. **门控阈值不一致** — 模型配置使用 0.3（`exist_gate_thd`），评估默认使用 0.4（`exist_thres`/`cls_thresholds`）。
4. **全集划分存在大量缺失特征** — 训练集 16,898 中 14,672 个、验证集 2,235 中 1,126 个样本缺少特征；标准集划分是实际可行的选择。
5. **脚本中的特征路径**：`features/soccer_gmr/{clip,clip_text,slowfast}` — 确认存在且文件数量符合预期。

### 下一步
- **阶段 0**：在现有测试预测上运行 `eval/eval_main.py` 以复现基线指标
- **阶段 0**：澄清哪个检查点产生表 2 的结果
- **阶段 1**：构建 `error_analysis.py`，复用 `metrics.py` 中的 `greedy_match`
