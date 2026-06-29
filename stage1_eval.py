import os
import json
import subprocess

def evaluate_rescore(rescore_type):
    print(f"\n--- Running inference with RESCORE_TYPE={rescore_type} ---")
    env = os.environ.copy()
    env["RESCORE_TYPE"] = rescore_type
    
    # Run inference
    subprocess.run(["bash", "scripts/infer_moment_detr_gmr.sh"], env=env, check=True, stdout=subprocess.DEVNULL)
    
    # Evaluate
    eval_script = "/home/guoxiangyu/GMR-Idea1-DualTrack/eval/eval_main.py"
    gt_path = "/home/guoxiangyu/GMR-Idea1-DualTrack/data/label/Standard/test_gt2.jsonl"
    sub_path = "/home/guoxiangyu/GMR-Idea1-DualTrack/results/moment_detr_gmr/test/moment_detr_gmr_test_submission.jsonl"
    
    taus = [0.3, 0.4, 0.5, 0.6, 0.7]
    mr5_sum = 0
    valid = 0
    
    for tau in taus:
        save_path = f"/tmp/eval_{rescore_type}_tau{tau}.json"
        cmd = f"python {eval_script} --submission_path '{sub_path}' --gt_path '{gt_path}' --save_path '{save_path}' --gmiou_cls_threshold {tau} > /dev/null 2>&1"
        os.system(cmd)
        
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                metrics = json.load(f)
                mr5 = metrics['brief'].get('mR+@5', 0)
                mr5_sum += mr5
                valid += 1
                
    if valid == len(taus):
        avg_mr5 = mr5_sum / valid
        print(f"[{rescore_type}] Avg mR+@5: {avg_mr5:.2f}%")
        
        # Save results
        out_dir = "/home/guoxiangyu/GMR-Idea1-DualTrack/results/stage1_saliency_contrast"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/{rescore_type}.json", "w") as f:
            json.dump({"mRplus@5": avg_mr5}, f, indent=2)
    else:
        print(f"Failed to evaluate {rescore_type}")

if __name__ == "__main__":
    evaluate_rescore("sharpA")
    evaluate_rescore("sharpB")
