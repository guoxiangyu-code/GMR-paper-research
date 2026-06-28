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

gt_dict = {}
with open(gt_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        gts = data.get('relevant_windows', [])
        if len(gts) == 2:
            gt_dict[data['qid']] = gts

probe_preds = {}
with open(probe_pred_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        probe_preds[data['qid']] = data['pred_relevant_windows']

for qid, gts in gt_dict.items():
    if qid not in probe_preds: continue
    
    probe_windows = probe_preds[qid]
    gt1, gt2 = gts
    
    # We want a case where GT1 is hit MULTIPLE TIMES in top 5, and GT2 is hit in rank > 5
    hit_gt1_count_top5 = sum(1 for pw in probe_windows[:5] if calculate_iou(pw, gt1) >= 0.5)
    
    iou_gt2_top10 = max([calculate_iou(pw, gt2) for pw in probe_windows[:10]]) if probe_windows else 0
    iou_gt2_top5 = max([calculate_iou(pw, gt2) for pw in probe_windows[:5]]) if probe_windows else 0
    
    if iou_gt2_top10 >= 0.5 and iou_gt2_top5 < 0.5 and hit_gt1_count_top5 >= 2:
        print(f"--- Extreme Crowding Example QID: {qid} ---")
        print(f"Ground Truth 1: {gt1}")
        print(f"Ground Truth 2: {gt2}")
        print("\n[Probe #3 (-SA + QD)] Top 10 Predictions:")
        for i, pw in enumerate(probe_windows[:10]):
            hit_gt1 = calculate_iou(pw, gt1) >= 0.5
            hit_gt2 = calculate_iou(pw, gt2) >= 0.5
            marker = "<- HITS GT1 (DUPLICATE)!" if hit_gt1 else ("<- HITS GT2 (TRUNCATED)!" if hit_gt2 else "")
            print(f"  Rank {i+1}: Span [{pw[0]:.1f}, {pw[1]:.1f}], Score: {pw[2]:.4f}  {marker}")
        break
