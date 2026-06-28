import json

def calculate_iou(window1, window2):
    start1, end1 = window1[:2]
    start2, end2 = window2[:2]
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection_duration = max(0, intersection_end - intersection_start)
    union_duration = (end1 - start1) + (end2 - start2) - intersection_duration
    if union_duration == 0:
        return 0.0
    return intersection_duration / union_duration

gt_path = "/home/guoxiangyu/GMR-Idea1-DualTrack/data/label/Standard/test.jsonl"
probe_pred_path = "/home/guoxiangyu/Base_code/generalized-moment-retrieval/experiments/20260627_stage2_run2/results/moment_detr_gmr_test_submission.jsonl"
idea1_pred_path = "/home/guoxiangyu/GMR-Idea1-DualTrack/results/moment_detr_gmr/moment_detr_gmr_test_submission.jsonl"

# Load GT=2 subset
gt_dict = {}
with open(gt_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        gts = data.get('relevant_windows', [])
        if len(gts) == 2:
            gt_dict[data['qid']] = gts

def evaluate_physical_coverage(pred_path, model_name):
    preds = {}
    with open(pred_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            preds[data['qid']] = data['pred_relevant_windows']

    hit_primary_count = 0
    hit_secondary_count = 0
    total = len(gt_dict)
    
    for qid, gts in gt_dict.items():
        if qid not in preds:
            continue
        pred_windows = preds[qid][:10]  # Take all 10 queries
        gt1, gt2 = gts
        
        # Max IoU for each GT across all 10 queries
        max_iou_gt1 = max([calculate_iou(pw, gt1) for pw in pred_windows]) if pred_windows else 0
        max_iou_gt2 = max([calculate_iou(pw, gt2) for pw in pred_windows]) if pred_windows else 0
        
        # Determine primary and secondary based on IoU
        iou_primary = max(max_iou_gt1, max_iou_gt2)
        iou_secondary = min(max_iou_gt1, max_iou_gt2)
        
        if iou_primary >= 0.5:
            hit_primary_count += 1
        if iou_secondary >= 0.5:
            hit_secondary_count += 1
            
    print(f"--- {model_name} ---")
    print(f"GT=2 Samples: {total}")
    print(f"Primary Target Hit (Max IoU >= 0.5): {hit_primary_count} / {total} ({hit_primary_count/total*100:.2f}%)")
    print(f"Secondary Target Hit (Max IoU >= 0.5): {hit_secondary_count} / {total} ({hit_secondary_count/total*100:.2f}%)\n")

evaluate_physical_coverage(idea1_pred_path, "Idea 1 (Double-Track)")
evaluate_physical_coverage(probe_pred_path, "Probe #3 (-SA + QD)")
