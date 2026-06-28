# -*- coding: utf-8 -*-
"""第二刀:GT>=2 样本的 active query span vs GT span 散布图。
直接肉眼判定 query 是铺在多个 moment 上,还是塌缩到主 moment。
依赖:推理时 dump 一份 per-sample 预测到 npz/json,字段见下。
用法:
  python experiments/20260627_stage2_run1/plot_query_span.py \
      --dump experiments/20260627_stage2_run1/test_pred_dump.pt \
      --out_dir experiments/20260627_stage2_run1/span_viz \
      --tau 0.05 --max_samples 24
"""
import argparse, os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def cxw_to_stae(spans):
    """spans: [N,2] (cx,w) 归一化 → (st,ed)。"""
    st = spans[:, 0] - spans[:, 1] / 2.0
    ed = spans[:, 0] + spans[:, 1] / 2.0
    return np.stack([st, ed], axis=1)

def plot_one(ax, gt_spans, q_spans, q_prob, title):
    # GT 底色带
    for g in gt_spans:
        ax.axvspan(g[0], g[1], ymin=0, ymax=1, color="#9ecae1", alpha=0.45, zorder=0)
    # 每个 active query 一条横条,y 错位排列,颜色=置信度
    cmap = get_cmap("autumn_r")
    order = np.argsort(-q_prob)
    for row, qi in enumerate(order):
        s, e = q_spans[qi]
        ax.barh(y=row, width=max(e - s, 1e-3), left=s, height=0.7,
                color=cmap(float(q_prob[qi])), edgecolor="k", linewidth=0.4, zorder=2)
        ax.text(e + 0.005, row, f"{q_prob[qi]:.2f}", va="center", fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, max(len(q_spans), 1))
    ax.set_yticks([])
    ax.set_xlabel("归一化时间轴")
    ax.set_title(title, fontsize=8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True,
                    help="推理 dump:list of dict,每条含 gt_spans[Ng,2], "
                         "pred_spans[N,2](cx,w), slot_fg_prob[N]")
    ap.add_argument("--out_dir", default="span_viz")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--max_samples", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = torch.load(args.dump)
    samples = [d for d in data if len(d["gt_spans"]) >= 2]   # 只看 GT>=2
    print(f"GT>=2 样本数: {len(samples)}")

    # 统计:每条样本里 active query 命中了几个不同 GT(tIoU>0.3 算命中)
    covered_ratio = []
    for d in samples:
        duration = float(d.get("duration", 150.0))
        gt_orig = np.asarray(d["gt_spans"], dtype=float)
        # gt_orig is absolute [st, ed]. We normalize to [0, 1]
        gt = gt_orig / duration if gt_orig.size > 0 else gt_orig
        prob = np.asarray(d["slot_fg_prob"], dtype=float)
        pred = cxw_to_stae(np.asarray(d["pred_spans"], dtype=float))
        act = pred[prob > args.tau]
        hit = set()
        for gi, g in enumerate(gt):
            for p in act:
                inter = max(0, min(g[1], p[1]) - max(g[0], p[0]))
                union = (g[1]-g[0]) + (p[1]-p[0]) - inter
                if union > 0 and inter/union > 0.3:
                    hit.add(gi); break
        if len(gt) > 0:
            covered_ratio.append(len(hit) / len(gt))
    cr = np.array(covered_ratio)
    print(f"[关键统计] GT>=2 样本平均 GT 覆盖率 = {cr.mean():.3f} "
          f"(=1 表示每个 GT 都有 query 覆盖;<<1 表示 query 塌缩漏掉次要 moment)")
    print(f"  覆盖率=1.0 的样本占比: {(cr>=0.999).mean():.3f}")
    print(f"  覆盖率<=0.5 的样本占比: {(cr<=0.5).mean():.3f}")

    # 画前 max_samples 张
    n = min(args.max_samples, len(samples))
    cols, rows = 4, int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 1.6*rows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        d = samples[i]
        duration = float(d.get("duration", 150.0))
        gt_orig = np.asarray(d["gt_spans"], dtype=float)
        gt = gt_orig / duration if gt_orig.size > 0 else gt_orig
        prob = np.asarray(d["slot_fg_prob"], float)
        pred = cxw_to_stae(np.asarray(d["pred_spans"], float))
        m = prob > args.tau
        plot_one(axes[i], gt, pred[m], prob[m],
                 f"#{i} GT={len(gt)} active={int(m.sum())} cov={covered_ratio[i]:.2f}")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("GT>=2: active query span (色=置信) vs GT moment (蓝带)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.out_dir, "query_span_grid.png")
    fig.savefig(out, dpi=150)
    print(f"[已保存] {out}")

if __name__ == "__main__":
    main()
