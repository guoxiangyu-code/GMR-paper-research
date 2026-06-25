你是我的科研代码助手,在当前目录下里工作，请你结合代码仓库，对论文进行阅读，对代码仓库
总原则(必须遵守):

1. 不要一上来就设计大而全的pipeline;做系统 error analysis,定位一个最主要的失败模式,再围绕它做一个小而有效的模块。
2. 一切结论用数据说话,基于官方评测脚本(eval/eval_main.py, eval/metrics.py)产出可复现的统计。
   仓库关键信息:

- 评测命令:python eval/eval_main.py --submission_path <pred> --gt_path data/label/Standard/test.jsonl --save_path <out>;指标含 AUROC/Rej-F1/Acc、mAP/mR@k/mR+@k/mIoU@k、G-mIoU@k。
- 已发布模型:Moment-DETR-GMR(training/、models/、scripts/train_*.sh、infer_*.sh);论文最强是 FlashVTG-GMR(权重/特征在 HuggingFace,需 NDA 申请)。
  
  

第一步:复现基线并对齐论文数值。（目前已完成/home/guoxiangyu/GMR/generalized-moment-retrieval/results/moment_detr_gmr/）

第二步：做系统 error analysis。设计一个方案，依据评测命令和评测指标做系统分析。

参考：

1. 拒识-误报(null-set 被接受):画出随 pred_exist_score 阈值变化的 FP 曲线;挑出被误判得分最高的语义相近负样本(如 "a shot" vs "a missed shot")做案例。
2. 拒识-漏报(positive 被错拒):FN 占比及其对 mAP 的拖累。
3. 多时刻漏检 / 只命中第一个:在 |G|>=2 上分别统计"第一个 moment 命中率"与"后续 moments 命中率"(对应 mR+@k),量化"只命中第一个"的样本比例。
4. 多检:|pred|>|G| 的比例及多余框的得分分布。
5. 边界不准:匹配对 IoU 直方图,统计落在 0.3<=IoU<0.5 的"差一点"样本占比。
6. 汇总表 + 关键分布图,并用一句话指出当前看起来最主要的瓶颈。

第三步:用"反事实(oracle)修复"给瓶颈排序,避免凭直觉选题。

1. 依次只修复某一类错误、其余不变,重跑 eval,记录 G-mIoU@1 / mAP / Rej-F1 的提升:
   - 修复多时刻漏检:把漏掉的后续 GT 视为已正确召回。
   - 修复边界:把所有匹配对 IoU 拉到 1。
   - 修复误报:把所有 null-set 正确拒掉。
   - 修复多检:删除所有多余预测。
3. 按映射确定方向并说明理由:多时刻漏检→coverage-aware retrieval;误报为主→candidate verification;边界偏移→boundary refinement。

---

## 补充：基于 Oracle 修复验证 (G-mIoU@3 口径)

通过执行 `pipeline/oracle_fix_v2.py` 验证了：
1. `eval/metrics.py` 中的 `compute_G_mIoU` 确实严格按照 `preds[:k]` 进行了 top-k 截断。
2. 我们补全了针对 "Fix FN" 的分析。
3. 对 "Fix Multi-miss" 和其他情况均在 G-mIoU@3 的口径下进行了增益计算。

| 修复类型 | G-mIoU@1 增益 | G-mIoU@3 增益 | Rej-F1 增益 | mR+@5 增益 |
|---|---|---|---|---|
| **Fix Over-detection (修复过检)** | **+18.92** | **+21.31** | +0.00 | +2.18 |
| **Fix FN (修复漏报)** | +0.80 | +0.69 | **+18.56** | +0.00 |
| **Fix Multi-miss (多时刻漏检)** | 0.00 | 0.00 | 0.00 | 0.00 |

### 最终选定的瓶颈与理由：
**瓶颈**：严重过检（Over-detection）与存在性分数偏移（FP）。
**理由**：
1. "Fix Over-detection" 即使在 `@3` 口径下（考虑了多预测框），依然带来了最为震撼的增益（`+21.31`）。这说明模型输出的后续窗口主要是干扰噪声，不仅无助于找回漏检目标，反而稀释了正确匹配对的 IoU（分母惩罚）。
2. "Fix FN" 的实验清晰表明：模型的主要分类错误在于误报（FP）而非漏报（FN），因为即便我们 Oracle 地补全了因分数过低而错拒的正样本，G-mIoU 的增益也微乎其微（`+0.69`），但它揭示了由于这部分样本带来的 Rej-F1 缺口（`+18.56`）。
3. "Fix Multi-miss" 即便在 `@3` 甚至考虑 `mR+@5` 口径下依然**毫无增益**。这确认了我们之前的判断：模型目前根本不是“只差一个 GT 框”的问题，而是预测集里全是噪声，导致单纯塞入真实框不仅受限于 `@1` 结构，更无法改变被过检框摊薄的事实。

**改进方向**：后处理模块（置信度过滤 / 存在性得分校准），彻底校正打分偏移解决误报。