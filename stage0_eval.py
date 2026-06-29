import os
import json

eval_script = "/home/guoxiangyu/GMR-Idea1-DualTrack/eval/eval_main.py"
gt_path = "/home/guoxiangyu/GMR-Idea1-DualTrack/data/label/Standard/test_gt2.jsonl"
sub_path = "/home/guoxiangyu/GMR-Idea1-DualTrack/results/moment_detr_gmr/moment_detr_gmr_test_submission.jsonl"

taus = [0.3, 0.4, 0.5, 0.6, 0.7]
mr5_sum = 0
gmiou1_sum = 0
valid = 0

for tau in taus:
    save_path = f"/tmp/eval_baseline_tau{tau}.json"
    cmd = f"python {eval_script} --submission_path '{sub_path}' --gt_path '{gt_path}' --save_path '{save_path}' --gmiou_cls_threshold {tau} > /dev/null 2>&1"
    os.system(cmd)
    
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            metrics = json.load(f)
            # mR+@5 is in metrics['brief']
            mr5 = metrics['brief'].get('mR+@5', 0)
            gmiou1 = metrics['brief'].get('G-mIoU@1', 0)
            mr5_sum += mr5
            gmiou1_sum += gmiou1
            valid += 1

if valid == len(taus):
    avg_mr5 = mr5_sum / valid
    avg_gmiou1 = gmiou1_sum / valid
    print(f"Baseline Avg mR+@5: {avg_mr5:.2f}%")
    print(f"Baseline Avg G-mIoU@1: {avg_gmiou1:.2f}%")
    
    # Check assertions
    assert abs(avg_mr5 - 1.17) < 0.3, f"mR+@5 mismatch: {avg_mr5}"
    
    os.makedirs("/home/guoxiangyu/GMR-Idea1-DualTrack/results", exist_ok=True)
    with open("/home/guoxiangyu/GMR-Idea1-DualTrack/results/anchor.json", "w") as f:
        json.dump({
            "baseline_mRplus@5": avg_mr5,
            "baseline_G_mIoU@1": avg_gmiou1,
            "oracle_ceiling": 14.06
        }, f, indent=2)
else:
    print("Failed to compute for all taus.")
