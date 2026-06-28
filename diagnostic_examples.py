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
idea1_pred_path = "/home/guoxiangyu/GMR-Idea1-DualTrack/results/moment_detr_gmr/moment_detr_gmr_test_submission.jsonl"
probe_pred_path = "/home/guoxiangyu/Base_code/generalized-moment-retrieval/experiments/20260627_stage2_run2/results/moment_detr_gmr_test_submission.jsonl"

gt_dict = {}
with open(gt_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        gts = data.get('relevant_windows', [])
        if len(gts) == 2:
            gt_dict[data['qid']] = gts

def get_examples(pred_path, model_name, limit=3):
    preds = {}
    with open(pred_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            preds[data['qid']] = data['pred_relevant_windows']

    count = 0
    print(f"============== {model_name} 的典型被截断案例 ==============")
    for qid, gts in gt_dict.items():
        if qid not in preds: continue
        
        pred_windows = preds[qid][:10]
        gt1, gt2 = gts
        
        hit_gt1_top5 = any(calculate_iou(pw, gt1) >= 0.5 for pw in pred_windows[:5])
        hit_gt2_top5 = any(calculate_iou(pw, gt2) >= 0.5 for pw in pred_windows[:5])
        
        hit_gt2_top10 = any(calculate_iou(pw, gt2) >= 0.5 for pw in pred_windows[:10])
        hit_gt1_top10 = any(calculate_iou(pw, gt1) >= 0.5 for pw in pred_windows[:10])
        
        # We want: one GT is hit in Top 5 (Primary), the other is hit in Rank 6-10 (Secondary)
        primary_gt, secondary_gt = None, None
        
        if hit_gt1_top5 and not hit_gt2_top5 and hit_gt2_top10:
            primary_gt = gt1
            secondary_gt = gt2
        elif hit_gt2_top5 and not hit_gt1_top5 and hit_gt1_top10:
            primary_gt = gt2
            secondary_gt = gt1
            
        if primary_gt and secondary_gt:
            print(f"\n[案例 {count+1}] QID: {qid}")
            print(f"主事件 (Primary GT): {primary_gt}")
            print(f"次要事件 (被截断的 GT): {secondary_gt}")
            print("预测排序列表 (Top 10):")
            
            for i, pw in enumerate(pred_windows):
                iou_p = calculate_iou(pw, primary_gt)
                iou_s = calculate_iou(pw, secondary_gt)
                
                if iou_p >= 0.5:
                    marker = f"🔴 命中主事件! (IoU: {iou_p:.2f})"
                elif iou_s >= 0.5:
                    status = "✅ 但惨遭截断" if i > 4 else "✅ 成功存活"
                    marker = f"🟢 命中次要事件! (IoU: {iou_s:.2f}) {status}"
                else:
                    marker = "⚪ 假阳性干扰/背景"
                    
                rank_str = f"Rank {i+1:2d}"
                print(f"  {rank_str} | 得分: {pw[2]:.4f} | 预测框: [{pw[0]:5.1f}, {pw[1]:5.1f}] | {marker}")
                
            count += 1
            if count >= limit:
                break
    print("\n")

get_examples(idea1_pred_path, "Idea 1 (双轨制)")
get_examples(probe_pred_path, "探针 #3 (-SA + QD)")
