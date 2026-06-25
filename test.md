# GMR 测试指令与结果说明

本文档记录了在 GMR (Generalized Moment Retrieval) 项目中，用于测试、评测以及错误诊断分析的各个指令，并详细说明了其运行结果的含义。

## 1. 推理生成预测结果 (Inference)

**指令:**
```bash
bash scripts/infer_moment_detr_gmr.sh
```

**测试目的:**
加载训练好的模型权重，对测试集的所有查询进行推理预测。

**会产生什么种类的结果:**
在 `results/moment_detr_gmr/test/` 生成 `moment_detr_gmr_test_submission.jsonl`。这是后续所有分析的基础数据源。
每行包含一个查询的预测结果，核心字段是 `pred_exist_score` (有相关片段的置信度) 和 `pred_relevant_windows` (预测的时序窗口及其分数，格式如 `[[start, end, score], ...]`)。

---

## 2. 官方基线评测 (Official Evaluation)

**指令:**
```bash
python eval/eval_main.py \
  --submission_path results/moment_detr_gmr/test/moment_detr_gmr_test_submission.jsonl \
  --gt_path data/label/Standard/test.jsonl \
  --save_path results/moment_detr_gmr/test/test_results.json
```
*(可通过 `--gmiou_cls_threshold` 改变默认的分类阈值，例如测试 `0.55`)*

**测试目的:**
严格比对模型预测结果与人工标注 (Ground Truth)，得出国际基准性能，从而横向对比原论文。

**会产生什么种类的结果:**
输出一个 `.json` 文件（例如 `test_results.json`），包含以下关键数值指标：
- **`AUROC`**: 判别 query 是否属于 null-set（无合理片段）的整体分类能力。
- **`Rej-F1`**: 拒识 F1 分数（衡量准确识别无片段 query，避免误报的能力）。
- **`mAP`**: 时序定位平均精度（衡量框得准不准）。
- **`mR@5`**: 在 Top-5 预测中找对第一个目标时刻的比例。
- **`mR+@5`**: 针对包含多个有效片段的 query，其后续时刻被找出的比例。
- **`G-mIoU@1`**: 兼顾“存在性判别（错拒/误报）”和“时序定位偏差”的终极融合分数。

---

## 3. 细粒度系统错误分析 (Error Analysis)

**指令:**
```bash
python pipeline/error_analysis.py
```

**测试目的:**
将模型在基线测试中丢失的分数，具象化地拆分为 5 种具体模式进行诊断量化：拒识误报 (FP)、拒识漏报 (FN)、边界不准 (Boundary inaccuracy)、多时刻漏检 (Multi-miss)、多检废框 (Over-detection)。

**会产生什么种类的结果:**
在 `results/moment_detr_gmr/error_analysis/` 目录下生成：
1. **`stats/*.json`**: 包含具体错误样本数和比例的纯数据分析文件。
2. **`figures/*.png`**: 可视化图表，例如：
   - 得分直方图（观察正负样本的置信度分离度）
   - IoU 直方图（观察时序边界准度）
   - FP/FN 阈值曲线
3. **`summary.md`**: 各类错误对样本的影响规模汇总表。

---

## 4. 反事实上限分析 (Oracle Analysis)

**指令:**
```bash
python pipeline/oracle_fix.py
```
*(改进版包含 G-mIoU@3 指标: `python pipeline/oracle_fix_v2.py`)*

**测试目的:**
利用 Ground Truth（上帝视角），分别针对上述 5 种错误，强行去进行“完美修复”（例如强行抹掉废框、强行拉齐边界），然后重跑评测，以衡量“如果未来把这个错误彻底解决了，总分能涨多少”。用来排定下一步优化的最高优先级。

**会产生什么种类的结果:**
在 `results/moment_detr_gmr/oracle_analysis/` 目录下生成：
- **修改后的临时预测文件并评测产生的 JSON**: 如 `fix_fp_eval.json`。
- **核心产物 `summary.md` 和终端增益排行表**: 输出一个带 `Gain`（涨幅）的排序清单。例如，终端会打印出 `Fix Over-detection` 能带来 `+18.92` 的最高提升，并自动打上 `🏆` 标签，明确告诉你这是最大的瓶颈。

---

## 5. 后处理策略调优测试 (Post-process Validation)

**指令:**
```bash
python pipeline/postprocess.py
```

**测试目的:**
基于 Oracle 分析找出的最大瓶颈（例如过检/误报严重），通过代码模拟实现不同的后处理防守策略（如分数截断 Score Thresholding、Soft-NMS 去重），快速验证方案是否有效。

**会产生什么种类的结果:**
在终端打印出“参数扫描（Sweep）”的比对结果：
- 比如会打印出 `Threshold=0.1` 时的 mAP，`Threshold=0.2` 时的 mAP，证明简单阈值过滤无效。
- 最后会得出最优组合方案及收益结果。
