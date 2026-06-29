# -*- coding: utf-8 -*-
"""GT=2 重打分离线模拟:用 diagnostic_gt2_analysis.json 现有字段,
判定"置信度-位置解耦重打分"能把 mR+@5 撬起多少。
用法: python rescore_sim.py diagnostic_gt2_analysis.json
"""
import json, sys

IOU_THR = 0.5   # 与你诊断里 19.53%/1.17% 同口径
TOPK = 5

def target_recalled(preds, target_key, k):
    """前 k 个预测里,是否有命中指定目标(hit_type 命中且 IoU>=阈值)。"""
    tag = "Primary GT" if target_key == "primary" else "Secondary GT"
    for p in preds[:k]:
        if p["hit_type"] == tag and p["max_iou"] >= IOU_THR:
            return True
    return False

def mrplus_at5(samples, order_fn):
    """order_fn(sample)->重排后的预测列表。GT=2 时 mR+@5=两目标都进top5 才算1。"""
    total = 0.0
    for s in samples:
        preds = order_fn(s)
        p_hit = target_recalled(preds, "primary", TOPK)
        s_hit = target_recalled(preds, "secondary", TOPK)
        # GT=2: 分母=1, 命中数-1; 两个都中=1, 否则0
        matched = int(p_hit) + int(s_hit)
        total += max(0, matched - 1) / 1.0
    return 100.0 * total / len(samples)

# ---- 三种排序策略 ----
def order_baseline(s):
    """现状:按 score 降序(JSON 已是此序)。"""
    return s["top10_predictions"]

def order_oracle(s):
    """上界:把任何命中 GT 的预测排到最前(模拟完美重打分器)。"""
    preds = s["top10_predictions"]
    def key(p):
        is_gt = p["hit_type"] in ("Primary GT", "Secondary GT") and p["max_iou"] >= IOU_THR
        return (0 if is_gt else 1, -p["score"])   # GT 命中优先,其余按原分
    return sorted(preds, key=key)

def order_diversity(s):
    """GT-free 代理:贪心选分最高,后续对"与已选时序重叠"的候选降权,
    逼断崖下、位置不同的 query 上位。不使用任何 GT 信息。"""
    preds = sorted(s["top10_predictions"], key=lambda p: -p["score"])
    def tiou(a, b):
        s1,e1=a; s2,e2=b
        inter=max(0,min(e1,e2)-max(s1,s2)); uni=(e1-s1)+(e2-s2)-inter
        return inter/uni if uni>0 else 0
    chosen=[]
    pool=list(preds)
    while pool and len(chosen)<len(preds):
        best=None; best_v=-1
        for p in pool:
            pen=max([tiou(p["window"],c["window"]) for c in chosen], default=0)
            v=p["score"]*(1-pen)          # 与已选重叠越多,有效分越低
            if v>best_v: best_v=v; best=p
        chosen.append(best); pool.remove(best)
    return chosen

def main():
    data=json.load(open(sys.argv[1], encoding="utf-8"))
    for model_key in data:
        S=data[model_key]
        print(f"\n=== {model_key}  (n={len(S)}) ===")
        print(f"  基线 (按 score top5)      mR+@5 = {mrplus_at5(S, order_baseline):.2f}%")
        print(f"  GT-free 代理 (多样性重排) mR+@5 = {mrplus_at5(S, order_diversity):.2f}%")
        print(f"  Oracle 上界 (完美重打分)  mR+@5 = {mrplus_at5(S, order_oracle):.2f}%")

if __name__=="__main__":
    main()
