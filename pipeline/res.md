
指标	值 (默认阈值0.4)	值 (阈值0.55)	论文 Moment-DETR-GMR	论文 FlashVTG-GMR
AUROC	70.00	70.00	72.09	74.00
Rej-F1	0.00	67.15	64.01	61.72
mAP	8.09	8.09	7.52	24.62
mR@5	14.14	14.14	12.96	33.36
mR+@5	0.97	0.97	0.84	19.10
G-mIoU@1	4.49	39.31	35.84	39.58
Fix Type	G-mIoU@1	Gain	mAP	Gain	Rej-F1	Gain
Baseline	39.31	—	8.09	—	67.15	—
Fix Over-detection 🏆	58.23	+18.92	20.64	+12.55	67.15	0
Fix FP	51.18	+11.87	8.09	0	80.52	+13.37
Fix Boundary	44.43	+5.12	38.83	+30.74	67.15	0
Fix Multi-miss	39.31	0	8.09	0	67.15	0
Fix All (上界)	91.66	+52.35	100.0	+91.91	100.0	+32.85
Error	Attempt	Resolution
(暂无)	-	-
文件/目录	用途
think.md	总原则和工作规划清单
eval/eval_main.py	官方评测脚本
eval/metrics.py	评测指标实现
data/label/Standard/test.jsonl	测试集 GT
results/moment_detr_gmr/test/moment_detr_gmr_test_submission.jsonl	测试集预测
results/moment_detr_gmr/test/test_results.json	默认阈值评测结果
results/moment_detr_gmr/test/test_results_opt2.json	阈值0.55评测结果
results/moment_detr_gmr/REPORT.md	基线复现报告
scripts/infer_moment_detr_gmr.sh	推理脚本
scripts/train_moment_detr_gmr.sh	训练脚本