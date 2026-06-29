import torch
import json
import os
import subprocess
from stage2_train import RerankHead, FEATS, stack_feats
from collections import defaultdict
import numpy as np

def tiou(w1,w2):
    s1,e1=w1; s2,e2=w2
    inter=max(0,min(e1,e2)-max(s1,s2)); uni=(e1-s1)+(e2-s2)-inter
    return inter/uni if uni>0 else 0

def temporal_nms(preds):
    chosen=[]
    pool=list(preds)
    while pool and len(chosen)<len(preds):
        best=None; best_v=-1
        for p in pool:
            pen=max([tiou(p[0:2],c[0:2]) for c in chosen], default=0)
            v=p[2]*(1-pen)
            if v>best_v: best_v=v; best=p
        chosen.append(best); pool.remove(best)
    return chosen

def evaluate_fusion():
    os.makedirs("results/stage2_main", exist_ok=True)
    
    taus = [0.3, 0.4, 0.5, 0.6, 0.7]
    eval_script = "/home/guoxiangyu/GMR-Idea1-DualTrack/eval/eval_main.py"
    gt_path_all = "/home/guoxiangyu/GMR-Idea1-DualTrack/data/label/Standard/test.jsonl"
    gt_path_gt2 = "/home/guoxiangyu/GMR-Idea1-DualTrack/data/label/Standard/test_gt2.jsonl"
    # Actually evaluate.py defaults to reporting everything but we need subset
    # I'll evaluate on test_gt2 to see the critical mR+@5
    
    seeds = [0, 1, 2]
    alphas = [1.0, 0.3, 0.5, 0.7]
    betas = [0.0, 0.3, 0.5, 0.7]
    
    results = defaultdict(list)
    cache_path = "results/rerank_cache_test.pt"
    cache = torch.load(cache_path)
    
    # Group by qid
    grouped = defaultdict(list)
    for f in cache:
        grouped[f["qid"]].append(f)
        
    baseline_path = "results/moment_detr_gmr/moment_detr_gmr_test_submission.jsonl"
    exist_scores_dict = {}
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            for line in f:
                data = json.loads(line)
                exist_scores_dict[data["qid"]] = data.get("pred_exist_score", 1.0)
    
    best_overall_a = 0
    best_overall_b = 0
    best_mr5 = 0
        
    subsets = {
        "all": "/home/guoxiangyu/GMR-Idea1-DualTrack/data/label/Standard/test.jsonl",
        "gt1": "/home/guoxiangyu/GMR-Idea1-DualTrack/data/label/Standard/test_gt1.jsonl",
        "gt2": "/home/guoxiangyu/GMR-Idea1-DualTrack/data/label/Standard/test_gt2.jsonl"
    }
    
    for seed in seeds:
        head_path = f"results/rerank_head_seed{seed}.pt"
        head = RerankHead().cuda()            # 构造时 buffer 是占位, load 后被覆盖
        head.load_state_dict(torch.load(head_path))
        head.eval()
        with torch.no_grad():
            feat_mat = stack_feats(cache).cuda()          # (N,4), 顺序与训练一致
            mlp_scores = head.forward_feats(feat_mat).cpu().numpy()
        for f, sc in zip(cache, mlp_scores):
            f["mlp_score"] = float(sc)
                
        for a in alphas:
            for b in betas:
                if a == 1.0 and b != 0.0: continue
                if a != 1.0 and b == 0.0: continue
                
                submission = []
                for qid, queries in grouped.items():
                    cur_query_pred = {"qid": qid, "query": "dummy", "vid": "dummy"}
                    if qid in exist_scores_dict:
                        cur_query_pred["pred_exist_score"] = exist_scores_dict[qid]
                    ranked_preds = []
                    for q in queries:
                        normalized_mlp = 1.0 / (1.0 + np.exp(-q["mlp_score"]))
                        score = (a * q["exist"] + b * normalized_mlp) / (a + b)
                        ranked_preds.append([float(q["s"]) * q["duration"], float(q["e"]) * q["duration"], float(score)])
                        
                    ranked_preds = sorted(ranked_preds, key=lambda x: x[2], reverse=True)
                    ranked_preds = temporal_nms(ranked_preds)
                    cur_query_pred["pred_relevant_windows"] = ranked_preds
                    submission.append(cur_query_pred)
                    
                sub_path = f"/tmp/sub_s{seed}_a{a}_b{b}.jsonl"
                with open(sub_path, "w") as f:
                    for row in submission:
                        f.write(json.dumps(row) + "\n")
                        
                for subset_name, subset_path in subsets.items():
                    if not os.path.exists(subset_path): continue
                    
                    mr5_sum = 0
                    gmiou1_sum = 0
                    rej_f1_sum = 0
                    valid = 0
                    for tau in taus:
                        save_path = f"/tmp/eval_s{seed}_a{a}_b{b}_{subset_name}_t{tau}.json"
                        cmd = f"python {eval_script} --submission_path '{sub_path}' --gt_path '{subset_path}' --save_path '{save_path}' --gmiou_cls_threshold {tau} > /dev/null 2>&1"
                        os.system(cmd)
                        if os.path.exists(save_path):
                            with open(save_path, "r") as f:
                                metrics = json.load(f)
                                mr5_sum += metrics['brief'].get('mR+@5', 0)
                                gmiou1_sum += metrics['brief'].get('G-mIoU@1', 0)
                                rej_f1_sum += metrics['brief'].get('Rej-F1', 0)
                                valid += 1
                    
                    if valid > 0:
                        avg_mr5 = mr5_sum / valid
                        avg_gmiou1 = gmiou1_sum / valid
                        avg_rej_f1 = rej_f1_sum / valid
                        results[f"{a}_{b}_{subset_name}_mr5"].append(avg_mr5)
                        results[f"{a}_{b}_{subset_name}_gmiou1"].append(avg_gmiou1)
                        results[f"{a}_{b}_{subset_name}_rejf1"].append(avg_rej_f1)
                        if subset_name == "gt2":
                            print(f"Seed {seed} a={a} b={b} -> GT=2 mR+@5: {avg_mr5:.2f}% | G-mIoU@1: {avg_gmiou1:.2f}%")
                    
    # Find best alpha beta
    for k, v in results.items():
        mean_val = np.mean(v)
        std_val = np.std(v)
        print(f"Fuse {k}: mean = {mean_val:.2f}%, std = {std_val:.2f}%")
        
        # Save results
        with open(f"results/stage2_main/fuse_{k}.json", "w") as f:
            json.dump({"mean": mean_val, "std": std_val, "raw": v}, f)

if __name__ == "__main__":
    evaluate_fusion()
