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
2. 按"指标增益"排序,选出收益最大的那一个作为切入点。
3. 按映射确定方向并说明理由:多时刻漏检→coverage-aware retrieval;误报为主→candidate verification;边界偏移→boundary refinement。