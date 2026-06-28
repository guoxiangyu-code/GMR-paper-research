import os
import json

eval_script = "/home/guoxiangyu/GMR-Idea1-DualTrack/eval/eval_main.py"
gt_path = "/home/guoxiangyu/GMR-Idea1-DualTrack/data/label/Standard/test.jsonl"
official_sub = "/home/guoxiangyu/Base_code/generalized-moment-retrieval/eval/example/example_test_submission.jsonl"

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print(f"{'Thresh':<8} | {'G-mIoU@1':<10} | {'Rej-F1':<10} | {'mAP':<8} | {'mR@5':<8} | {'mR+@5':<8}")
print("-" * 65)

for th in thresholds:
    save_path = f"/tmp/eval_official_{th}.json"
    cmd = f"python {eval_script} --submission_path {official_sub} --gt_path {gt_path} --save_path {save_path} --gmiou_cls_threshold {th} --cls_thresholds {th} > /dev/null 2>&1"
    os.system(cmd)
    
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            metrics = json.load(f)
            gmiou = metrics['brief'].get('G-mIoU@1', 0)
            rej_f1 = metrics['brief'].get(f'Rej-F1@{th}', 0)
            # if format differs, try GMR-CLS
            if rej_f1 == 0 and "GMR-CLS" in metrics:
                rej_f1 = metrics["GMR-CLS"]["per_threshold"].get(str(th), {}).get("Rej-F1", 0)
                
            map_score = metrics['brief'].get('mAP', 0)
            mr5 = metrics['brief'].get('mR@5', 0)
            mrp5 = metrics['brief'].get('mR+@5', 0)
            
            print(f"{th:<8.1f} | {gmiou:<10.2f} | {rej_f1:<10.2f} | {map_score:<8.2f} | {mr5:<8.2f} | {mrp5:<8.2f}")
    else:
        print(f"{th:<8.1f} | FAILED")

