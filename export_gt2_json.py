import json
import os

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
idea1_pred_path = "/home/guoxiangyu/GMR-Idea1-DualTrack/results/moment_detr_gmr/moment_detr_gmr_test_submission.jsonl"
probe_pred_path = "/home/guoxiangyu/Base_code/generalized-moment-retrieval/experiments/20260627_stage2_run2/results/moment_detr_gmr_test_submission.jsonl"
output_json_path = "/home/guoxiangyu/GMR-Idea1-DualTrack/diagnostic_gt2_analysis.json"

gt_dict = {}
with open(gt_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        gts = data.get('relevant_windows', [])
        if len(gts) == 2:
            gt_dict[data['qid']] = gts

def analyze_model(pred_path):
    preds = {}
    with open(pred_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            preds[data['qid']] = data['pred_relevant_windows']

    results = []
    
    for qid, gts in gt_dict.items():
        if qid not in preds: continue
        pred_windows = preds[qid][:10]
        gt_A, gt_B = gts
        
        hit_A_queries = [(i+1, pw, calculate_iou(pw, gt_A)) for i, pw in enumerate(pred_windows) if calculate_iou(pw, gt_A) >= 0.5]
        hit_B_queries = [(i+1, pw, calculate_iou(pw, gt_B)) for i, pw in enumerate(pred_windows) if calculate_iou(pw, gt_B) >= 0.5]
        
        max_sA = max([pw[2] for _, pw, _ in hit_A_queries]) if hit_A_queries else 0
        max_sB = max([pw[2] for _, pw, _ in hit_B_queries]) if hit_B_queries else 0
        
        if max_sA >= max_sB:
            primary_gt = gt_A
            secondary_gt = gt_B
            primary_hits = hit_A_queries
            secondary_hits = hit_B_queries
        else:
            primary_gt = gt_B
            secondary_gt = gt_A
            primary_hits = hit_B_queries
            secondary_hits = hit_A_queries
            
        prim_hit_data = None
        if primary_hits:
            best_p_rank, best_p_pw, best_p_iou = sorted(primary_hits, key=lambda x: -x[1][2])[0]
            prim_hit_data = {
                "rank": best_p_rank,
                "score": round(best_p_pw[2], 4),
                "iou": round(best_p_iou, 4),
                "is_top5": best_p_rank <= 5
            }
            
        sec_hit_data = None
        if secondary_hits:
            best_s_rank, best_s_pw, best_s_iou = sorted(secondary_hits, key=lambda x: -x[1][2])[0]
            sec_hit_data = {
                "rank": best_s_rank,
                "score": round(best_s_pw[2], 4),
                "iou": round(best_s_iou, 4),
                "is_top5": best_s_rank <= 5
            }
            
        top10_details = []
        for i, pw in enumerate(pred_windows):
            iou_p = calculate_iou(pw, primary_gt)
            iou_s = calculate_iou(pw, secondary_gt)
            if iou_p >= 0.5:
                target = "Primary GT"
                iou = iou_p
            elif iou_s >= 0.5:
                target = "Secondary GT"
                iou = iou_s
            else:
                target = "Background"
                iou = max(iou_p, iou_s)
                
            top10_details.append({
                "rank": i + 1,
                "window": [round(pw[0], 1), round(pw[1], 1)],
                "score": round(pw[2], 4),
                "hit_type": target,
                "max_iou": round(iou, 4)
            })
            
        results.append({
            "qid": qid,
            "primary_gt": primary_gt,
            "secondary_gt": secondary_gt,
            "primary_hit": prim_hit_data,
            "secondary_hit": sec_hit_data,
            "is_truncated_case": (sec_hit_data is not None and not sec_hit_data["is_top5"]),
            "top10_predictions": top10_details
        })
        
    return results

output_data = {
    "Idea_1_DualTrack": analyze_model(idea1_pred_path),
    "Probe_3_MinusSA_QD": analyze_model(probe_pred_path)
}

with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"Data exported successfully to {output_json_path}")
