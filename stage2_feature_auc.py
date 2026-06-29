# stage2_feature_auc.py
# 用法: python stage2_feature_auc.py
# 目的: 在训练任何 MLP 之前, 判定 perception 特征是否真的零判别力。
import torch
import numpy as np
from collections import defaultdict

FEATS = ["xattn_entropy", "sal_sharp", "width", "xmodal_align"]
CACHE_TRAIN = "results/rerank_cache_train.pt"

def auc_score(values, labels):
    """无 sklearn 依赖的 AUC = P(score_pos > score_neg), 用秩和(Mann-Whitney U)算。"""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = values[labels == 1]
    neg = values[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n_pos, n_neg
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    # 处理并列: 同值取平均秩
    _, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg_rank = (start + cum + 1) / 2.0  # 每个唯一值的平均秩
    ranks = avg_rank[inv]
    rank_sum_pos = ranks[labels == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc), n_pos, n_neg

def ks_distance(values, labels):
    """正负两类经验CDF的最大差距(KS), 0=无区分, 1=完全可分。"""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = np.sort(values[labels == 1])
    neg = np.sort(values[labels == 0])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    grid = np.sort(np.unique(values))
    cdf_pos = np.searchsorted(pos, grid, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, grid, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))

def describe(name, values, labels):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    auc, n_pos, n_neg = auc_score(values, labels)
    # AUC<0.5 说明该特征方向与label相反, 取 max(auc, 1-auc) 看"绝对判别力"
    auc_abs = max(auc, 1 - auc) if not np.isnan(auc) else float("nan")
    ks = ks_distance(values, labels)
    mu_p = values[labels == 1].mean() if n_pos else float("nan")
    mu_n = values[labels == 0].mean() if n_neg else float("nan")
    sd = values.std() + 1e-9
    cohen_d = (mu_p - mu_n) / sd  # 标准化均值差
    return {
        "feat": name, "auc": auc, "auc_abs": auc_abs, "ks": ks,
        "mu_pos": mu_p, "mu_neg": mu_n, "cohen_d": cohen_d,
        "n_pos": n_pos, "n_neg": n_neg,
    }

def main():
    cache = torch.load(CACHE_TRAIN)
    labels = [int(f["label"]) for f in cache]
    print(f"总窗口数={len(cache)}, 正例(命中GT)={sum(labels)}, 负例={len(labels)-sum(labels)}, "
          f"正例占比={100*sum(labels)/len(labels):.2f}%\n")

    rows = []
    for name in FEATS:
        vals = [float(f[name]) for f in cache]
        rows.append(describe(name, vals, labels))
    # 对照基准: 现有 existence 分数本身的判别力(看 MLP 至少要超过它)
    rows.append(describe("exist(baseline)", [float(f["exist"]) for f in cache], labels))

    print(f"{'feature':<18}{'AUC':>8}{'|AUC|':>8}{'KS':>8}{'mu_pos':>10}{'mu_neg':>10}{'cohen_d':>9}")
    for r in rows:
        print(f"{r['feat']:<18}{r['auc']:>8.3f}{r['auc_abs']:>8.3f}{r['ks']:>8.3f}"
              f"{r['mu_pos']:>10.4f}{r['mu_neg']:>10.4f}{r['cohen_d']:>9.3f}")

    # ---- 自动判读 ----
    perc = [r for r in rows if r["feat"] in FEATS]
    best = max(perc, key=lambda r: r["auc_abs"])
    print("\n" + "=" * 60)
    print(f"最强 perception 特征: {best['feat']}  |AUC|={best['auc_abs']:.3f}  KS={best['ks']:.3f}")
    if best["auc_abs"] >= 0.60:
        print(">> 结论: perception 含判别信号。MLP 之前失败=实现bug(尺度/优化),")
        print(">>       请运行 stage2_train.py(带标准化版)重训, 不要去改回归头。")
    elif best["auc_abs"] >= 0.55:
        print(">> 结论: 弱信号。单特征不够, 但组合+标准化可能有效, 值得跑标准化MLP。")
    else:
        print(">> 结论: perception 近乎零判别力(全部~0.5)。此时'特征不够'成立,")
        print(">>       但这仍不能推出'解锁位置能成功'——下一步应在 EaTR-GMR 上复测诊断。")

    # ---- 额外: 分层看 gt_cnt==2 的样本(第二目标问题核心) ----
    cache2 = [f for f in cache if int(f.get("gt_cnt", 0)) == 2]
    if cache2:
        lab2 = [int(f["label"]) for f in cache2]
        print(f"\n[GT=2 子集] 窗口数={len(cache2)}, 命中={sum(lab2)} ({100*sum(lab2)/len(lab2):.2f}%)")
        for name in FEATS:
            vals = [float(f[name]) for f in cache2]
            r = describe(name, vals, lab2)
            print(f"  {name:<16} |AUC|={r['auc_abs']:.3f}  KS={r['ks']:.3f}  cohen_d={r['cohen_d']:.3f}")

if __name__ == "__main__":
    main()
