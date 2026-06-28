# -*- coding: utf-8 -*-
"""阶段二 #3 vs 基线 active-vs-moment 对比绘图。
复用 diagnose.py 的产物(每行 (n_gt_moment, n_active),active 阈值 0.05)。
用法:
  python plot_compare.py \
      --baseline experiments/diag/active_vs_moment.npy \
      --variant  experiments/20260627_stage2_run1/active_vs_moment.npy \
      --out      experiments/20260627_stage2_run1/compare_active.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUCKETS = [("GT=1", 1, 1), ("GT=2", 2, 2), ("GT=3", 3, 3), ("GT>=4", 4, 10**9)]


def bucket_stats(rows):
    """rows: [N,2] (n_gt, n_active) → list of (label, count, mean, std)."""
    out = []
    g = rows[:, 0]
    a = rows[:, 1]
    for label, lo, hi in BUCKETS:
        m = a[(g >= lo) & (g <= hi)]
        if len(m):
            out.append((label, len(m), float(m.mean()), float(m.std())))
        else:
            out.append((label, 0, np.nan, np.nan))
    return out


def print_table(name, stats):
    print(f"\n[{name}]")
    print(f"{'GT 桶':<8}{'样本数':>8}{'active 均值':>14}{'std':>10}")
    for label, n, mean, std in stats:
        mean_s = "--" if np.isnan(mean) else f"{mean:.2f}"
        std_s = "--" if np.isnan(std) else f"{std:.2f}"
        print(f"{label:<8}{n:>8}{mean_s:>14}{std_s:>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="基线 .npy 路径")
    ap.add_argument("--variant", required=True, help="#3 .npy 路径")
    ap.add_argument("--out", default="compare_active.png")
    ap.add_argument("--baseline_name", default="Baseline (Idea 1)")
    ap.add_argument("--variant_name", default="#3 (-SA+QD+NMS)")
    args = ap.parse_args()

    base = np.load(args.baseline)
    var = np.load(args.variant)
    base_stats = bucket_stats(base)
    var_stats = bucket_stats(var)

    print_table(args.baseline_name, base_stats)
    print_table(args.variant_name, var_stats)

    # 关键判定:GT>=2 各桶 active 是否抬升(治住坍缩的签名)
    print("\n[判定] GT>=2 active 抬升幅度(>0 即坍缩缓解):")
    for i, (label, _, _) in enumerate([(b[0], b[1], b[2]) for b in BUCKETS]):
        if BUCKETS[i][1] >= 2:
            d = var_stats[i][2] - base_stats[i][2]
            d_s = "--" if np.isnan(d) else f"{d:+.2f}"
            print(f"  {label}: {d_s}")

    # 画图
    x = np.arange(len(BUCKETS))
    labels = [b[0] for b in BUCKETS]
    bm = [s[2] for s in base_stats]
    be = [0 if np.isnan(s[3]) else s[3] for s in base_stats]
    vm = [s[2] for s in var_stats]
    ve = [0 if np.isnan(s[3]) else s[3] for s in var_stats]

    plt.figure(figsize=(7, 5))
    plt.errorbar(x, bm, yerr=be, marker="o", capsize=4,
                 label=args.baseline_name, linewidth=2)
    plt.errorbar(x, vm, yerr=ve, marker="s", capsize=4,
                 label=args.variant_name, linewidth=2)
    plt.xticks(x, labels)
    plt.xlabel("GT Moment 数")
    plt.ylabel("平均激活 Query 数 (active, prob>0.05)")
    plt.title("Active Decoder-Query vs GT Moments")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\n[已保存] {args.out}")


if __name__ == "__main__":
    main()
