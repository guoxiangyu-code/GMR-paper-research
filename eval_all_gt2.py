import os
import json

base_exp_dir = "/home/guoxiangyu/Base_code/generalized-moment-retrieval/experiments"
eval_script = "/home/guoxiangyu/GMR-Idea1-DualTrack/eval/eval_main.py"
gt2_path = "/home/guoxiangyu/GMR-Idea1-DualTrack/data/label/Standard/test_gt2.jsonl"

exps = [
    ("Idea 1 (Dual-Track 纯净基座)", "/home/guoxiangyu/GMR-Idea1-DualTrack/results/moment_detr_gmr"),
    ("Probe #2 (+SA+QD)", f"{base_exp_dir}/20260627_stage2_run1/results"),
    ("Probe #3 (-SA+QD 坍缩)", f"{base_exp_dir}/20260627_stage2_run2/results"),
    ("Div Loss (coef=0.1)", f"{base_exp_dir}/20260627_stage3_coef0.1/results"),
    ("Div Loss (coef=0.3)", f"{base_exp_dir}/20260627_stage3_run1_coef0.3/results"),
    ("Div Loss (coef=0.5)", f"{base_exp_dir}/20260627_stage3_coef0.5/results"),
    ("Div Loss (coef=1.0)", f"{base_exp_dir}/20260627_stage3_coef1.0/results"),
]

print("| 实验配置 | mR@5 (找到第1个目标的概率) | mR+@5 (找到第2个目标的概率) |")
print("| :--- | :--- | :--- |")

for name, res_dir in exps:
    sub_path = f"{res_dir}/moment_detr_gmr_test_submission.jsonl"
    save_path = f"{res_dir}/test_metrics_gt2.json"
    
    if not os.path.exists(sub_path):
        print(f"| {name} | Missing submission file | - |")
        continue

    # Run eval if not already run
    if not os.path.exists(save_path):
        cmd = f"python {eval_script} --submission_path {sub_path} --gt_path {gt2_path} --save_path {save_path} --gmiou_cls_threshold 0.6 > /dev/null 2>&1"
        os.system(cmd)
        
    # Read metrics
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            metrics = json.load(f)
            mr5 = metrics['brief'].get('mR@5', 0)
            mrp5 = metrics['brief'].get('mR+@5', 0)
            print(f"| {name} | {mr5:.2f}% | {mrp5:.2f}% |")
    else:
        print(f"| {name} | Eval failed | - |")
