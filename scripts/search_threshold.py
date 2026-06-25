import subprocess
import json
import numpy as np

submission_path = "results/moment_detr_gmr_fixed_val/best_soccer_gmr_val_preds.jsonl"
gt_path = "data/label/Standard/val.jsonl"
save_prefix = "results/moment_detr_gmr_fixed_val/val_eval_thd_"

thresholds = np.arange(0.4, 0.81, 0.05)
best_thd = 0.4
best_score = -1

for thd in thresholds:
    thd_str = f"{thd:.2f}"
    save_path = f"{save_prefix}{thd_str}.json"
    
    cmd = [
        "python", "eval/eval_main.py",
        "--submission_path", submission_path,
        "--gt_path", gt_path,
        "--save_path", save_path,
        "--gmiou_cls_threshold", thd_str
    ]
    
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    
    with open(save_path, "r") as f:
        metrics = json.load(f)
        
    g_miou_1 = metrics["G-mIoU_detail"]["G-mIoU@1"]
    print(f"Threshold: {thd_str}, G-mIoU@1: {g_miou_1:.2f}")
    
    if g_miou_1 > best_score:
        best_score = g_miou_1
        best_thd = thd

print(f"Best Threshold: {best_thd:.2f} with G-mIoU@1: {best_score:.2f}")
