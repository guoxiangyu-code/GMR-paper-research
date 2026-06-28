import json
import numpy as np

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

gt_dict = {}
with open(gt_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        gts = data.get('relevant_windows', [])
        if len(gts) == 2:
            gt_dict[data['qid']] = gts

def analyze_score_distributions(pred_path, model_name):
    preds = {}
    with open(pred_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            preds[data['qid']] = data['pred_relevant_windows']

    gt1_scores = []
    gt2_scores = []
    gt2_ranks = []
    
    for qid, gts in gt_dict.items():
        if qid not in preds: continue
        pred_windows = preds[qid][:10]  # top 10
        gt1, gt2 = gts
        
        # Find best query hitting GT1
        hit_gt1_queries = [(i+1, pw[2]) for i, pw in enumerate(pred_windows) if calculate_iou(pw, gt1) >= 0.5]
        hit_gt2_queries = [(i+1, pw[2]) for i, pw in enumerate(pred_windows) if calculate_iou(pw, gt2) >= 0.5]
        
        # We need to correctly identify which GT is the "Primary" (the one with the higher max score)
        max_s1 = max([s for r, s in hit_gt1_queries]) if hit_gt1_queries else 0
        max_s2 = max([s for r, s in hit_gt2_queries]) if hit_gt2_queries else 0
        
        if max_s1 == 0 and max_s2 == 0: continue
        
        if max_s1 >= max_s2:
            primary_hits = hit_gt1_queries
            secondary_hits = hit_gt2_queries
        else:
            primary_hits = hit_gt2_queries
            secondary_hits = hit_gt1_queries
            
        if primary_hits:
            gt1_scores.append(max([s for r, s in primary_hits]))
        if secondary_hits:
            best_sec_rank, best_sec_score = sorted(secondary_hits, key=lambda x: -x[1])[0]
            gt2_scores.append(best_sec_score)
            gt2_ranks.append(best_sec_rank)

    print(f"========== {model_name} 分数与排序分布 ==========")
    print(f"主目标 (Primary GT) 平均置信度分数: {np.mean(gt1_scores):.4f} (样本数: {len(gt1_scores)})")
    if gt2_scores:
        print(f"次要目标 (Secondary GT) 平均置信度分数: {np.mean(gt2_scores):.4f} (样本数: {len(gt2_scores)})")
        print(f"置信度鸿沟 (主目标比次要目标高出): {np.mean(gt1_scores) - np.mean(gt2_scores):.4f}")
        print(f"次要目标 (Secondary GT) 在候选列表中的平均排名: 第 {np.mean(gt2_ranks):.1f} 名")
        # 分布区间
        rank_in_top5 = sum(1 for r in gt2_ranks if r <= 5)
        rank_out_top5 = sum(1 for r in gt2_ranks if r > 5)
        print(f" -> 挤进 Top-5 的次数: {rank_in_top5}")
        print(f" -> 跌出 Top-5 被截断的次数: {rank_out_top5}")
    print("\n")

analyze_score_distributions(idea1_pred_path, "Idea 1 (双轨制)")
analyze_score_distributions(probe_pred_path, "探针 #3 (-SA + QD)")
