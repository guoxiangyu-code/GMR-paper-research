#!/usr/bin/env python3
"""
Phase 2: Systematic Error Analysis for Moment-DETR-GMR on Soccer-GMR test set.

Analyzes 5 error types:
  2.2  Rejection FP  — null-set queries accepted (false alarms)
  2.3  Rejection FN  — positive queries rejected (misses)
  2.4  Multi-moment miss — only first moment retrieved
  2.5  Over-detection — |pred| > |GT|
  2.6  Boundary inaccuracy — matched pairs with low IoU

Usage:
  cd /home/guoxiangyu/GMR/generalized-moment-retrieval
  python pipeline/error_analysis.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
GT_PATH = ROOT / "data/label/Standard/test.jsonl"
PRED_PATH = ROOT / "results/moment_detr_gmr/test/moment_detr_gmr_test_submission.jsonl"
OUT_DIR = ROOT / "results/moment_detr_gmr/error_analysis"
STATS_DIR = OUT_DIR / "stats"
FIG_DIR = OUT_DIR / "figures"
CASE_DIR = OUT_DIR / "cases"

for d in [STATS_DIR, FIG_DIR, CASE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── matplotlib (non-interactive) ────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})

# ─── I/O helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_json(obj: Any, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  [saved] {path.relative_to(ROOT)}")


# ─── IoU helpers (mirrors eval/metrics.py) ────────────────────────────────────

def temporal_iou(pred: List[float], gt: List[float]) -> float:
    """Compute temporal IoU between two windows [st, ed]."""
    inter_st = max(pred[0], gt[0])
    inter_ed = min(pred[1], gt[1])
    inter = max(0.0, inter_ed - inter_st)
    if inter == 0:
        return 0.0
    union = (pred[1] - pred[0]) + (gt[1] - gt[0]) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(
    preds: List[List[float]],
    gts: List[List[float]],
    iou_thd: float = -1.0,
) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching of preds to gts (mirrors eval/metrics.py)."""
    if not preds or not gts:
        return []
    iou_matrix = np.array(
        [[temporal_iou(p, g) for g in gts] for p in preds], dtype=np.float64
    )
    matched_gt: set = set()
    matches: List[Tuple[int, int, float]] = []
    for i in range(iou_matrix.shape[0]):
        best_iou, best_j = -1.0, None
        for j in range(iou_matrix.shape[1]):
            if j not in matched_gt and iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j
        if best_j is not None and best_iou >= iou_thd:
            matched_gt.add(best_j)
            matches.append((i, best_j, float(best_iou)))
    return matches


# ─── Load data ────────────────────────────────────────────────────────────────

def load_data():
    gt_list = load_jsonl(GT_PATH)
    pred_list = load_jsonl(PRED_PATH)

    pred_map: Dict[Any, Dict] = {d["qid"]: d for d in pred_list}

    records = []
    for gt in gt_list:
        qid = gt["qid"]
        gt_windows = gt.get("relevant_windows", [])
        pred = pred_map.get(qid, {})
        pred_windows_raw = pred.get("pred_relevant_windows", [])
        # strip confidence score → [[st, ed], ...]
        pred_windows = [[w[0], w[1]] for w in pred_windows_raw]
        exist_score = float(pred.get("pred_exist_score", 0.0))

        is_positive = len(gt_windows) > 0
        is_multi = len(gt_windows) >= 2

        records.append({
            "qid": qid,
            "vid": gt.get("vid", ""),
            "query": gt.get("query", ""),
            "gt_windows": gt_windows,
            "pred_windows": pred_windows,   # [[st, ed], ...]
            "pred_windows_with_score": pred_windows_raw,  # [[st, ed, score], ...]
            "exist_score": exist_score,
            "is_positive": is_positive,
            "is_multi": is_multi,
            "n_gt": len(gt_windows),
            "n_pred": len(pred_windows),
        })

    print(f"Loaded {len(records)} samples: "
          f"{sum(r['is_positive'] for r in records)} positive "
          f"({sum(r['is_multi'] for r in records)} multi), "
          f"{sum(not r['is_positive'] for r in records)} negative")
    return records


# ─── Task 2.2: Rejection FP analysis ─────────────────────────────────────────

def analyze_rejection_fp(records: List[Dict]) -> Dict:
    """Null-set queries accepted as positive (false alarms)."""
    print("\n=== Task 2.2: Rejection FP (null-set accepted) ===")

    pos_scores = [r["exist_score"] for r in records if r["is_positive"]]
    neg_scores = [r["exist_score"] for r in records if not r["is_positive"]]

    # ── Score distribution stats ──
    def stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
        }

    score_stats = {
        "positive": stats(pos_scores),
        "negative": stats(neg_scores),
    }

    # ── Threshold sweep ──
    thresholds = np.arange(0.0, 1.01, 0.01)
    sweep = []
    best_rej_f1, best_thd = -1.0, 0.0
    for thd in thresholds:
        TP = sum(1 for r in records if r["is_positive"] and r["exist_score"] > thd)
        TN = sum(1 for r in records if not r["is_positive"] and r["exist_score"] <= thd)
        FP = sum(1 for r in records if not r["is_positive"] and r["exist_score"] > thd)
        FN = sum(1 for r in records if r["is_positive"] and r["exist_score"] <= thd)
        rej_p = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        rej_r = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        rej_f1 = 2 * rej_p * rej_r / (rej_p + rej_r) if (rej_p + rej_r) > 0 else 0.0
        sweep.append({"thd": round(float(thd), 2), "TP": TP, "TN": TN, "FP": FP, "FN": FN,
                      "Rej-F1": round(100 * rej_f1, 2)})
        if rej_f1 > best_rej_f1:
            best_rej_f1 = rej_f1
            best_thd = float(thd)

    print(f"  Best Rej-F1 = {100*best_rej_f1:.2f} at threshold = {best_thd:.2f}")

    # ── Top-20 FP cases (neg samples with highest exist_score) ──
    neg_sorted = sorted(
        [r for r in records if not r["is_positive"]],
        key=lambda r: r["exist_score"], reverse=True
    )
    top_fp_cases = [
        {"qid": r["qid"], "vid": r["vid"], "query": r["query"],
         "exist_score": r["exist_score"]}
        for r in neg_sorted[:20]
    ]

    result = {
        "score_stats": score_stats,
        "best_threshold": best_thd,
        "best_rej_f1": round(100 * best_rej_f1, 2),
        "threshold_sweep": sweep,
        "top20_fp_cases": top_fp_cases,
    }
    save_json(result, STATS_DIR / "fp_analysis.json")

    # ── Figures ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: score distribution
    ax = axes[0]
    bins = np.linspace(0, 1, 51)
    ax.hist(neg_scores, bins=bins, alpha=0.65, color="#e74c3c", label=f"Negative (null, n={len(neg_scores)})", density=True)
    ax.hist(pos_scores, bins=bins, alpha=0.55, color="#3498db", label=f"Positive (n={len(pos_scores)})", density=True)
    ax.axvline(0.4, color="gray", ls="--", lw=1.2, label="Default τ=0.4")
    ax.axvline(0.55, color="orange", ls="--", lw=1.2, label="Opt τ=0.55")
    ax.set_xlabel("pred_exist_score")
    ax.set_ylabel("Density")
    ax.set_title("Existence Score Distribution\n(positive vs. negative samples)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: FP / FN / Rej-F1 vs threshold
    ax = axes[1]
    thds_arr = np.array([s["thd"] for s in sweep])
    fp_arr = np.array([s["FP"] for s in sweep])
    fn_arr = np.array([s["FN"] for s in sweep])
    rej_f1_arr = np.array([s["Rej-F1"] for s in sweep])
    ax2 = ax.twinx()
    ax.plot(thds_arr, fp_arr, color="#e74c3c", lw=1.8, label="FP count")
    ax.plot(thds_arr, fn_arr, color="#3498db", lw=1.8, label="FN count")
    ax2.plot(thds_arr, rej_f1_arr, color="#2ecc71", lw=2.2, ls="-.", label="Rej-F1 (%)")
    ax.axvline(best_thd, color="orange", ls="--", lw=1.2, label=f"Best τ={best_thd:.2f}")
    ax.set_xlabel("Existence Score Threshold τ")
    ax.set_ylabel("Count (FP / FN)")
    ax2.set_ylabel("Rej-F1 (%)")
    ax.set_title("FP / FN / Rej-F1 vs. Threshold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "exist_score_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] figures/exist_score_distribution.png")

    return result


# ─── Task 2.3: Rejection FN analysis ─────────────────────────────────────────

def analyze_rejection_fn(records: List[Dict], thd: float = 0.55) -> Dict:
    """Positive queries rejected (false rejections) at the optimal threshold."""
    print(f"\n=== Task 2.3: Rejection FN (positive rejected, τ={thd}) ===")

    fn_records = [r for r in records if r["is_positive"] and r["exist_score"] <= thd]
    pos_records = [r for r in records if r["is_positive"]]

    fn_ratio = len(fn_records) / len(pos_records) if pos_records else 0.0
    print(f"  FN count = {len(fn_records)} / {len(pos_records)} positives "
          f"= {100*fn_ratio:.1f}%")

    # Analyze mAP impact: subset of correctly accepted positives
    tp_records = [r for r in records if r["is_positive"] and r["exist_score"] > thd]

    # Top-20 FN cases (positive with lowest exist_score)
    fn_sorted = sorted(fn_records, key=lambda r: r["exist_score"])
    top_fn_cases = [
        {"qid": r["qid"], "vid": r["vid"], "query": r["query"],
         "exist_score": r["exist_score"],
         "gt_windows": r["gt_windows"],
         "n_gt": r["n_gt"],
         "is_multi": r["is_multi"]}
        for r in fn_sorted[:20]
    ]

    # FN breakdown by multi vs single
    fn_single = [r for r in fn_records if not r["is_multi"]]
    fn_multi = [r for r in fn_records if r["is_multi"]]

    result = {
        "threshold": thd,
        "fn_count": len(fn_records),
        "positive_count": len(pos_records),
        "fn_ratio": round(100 * fn_ratio, 2),
        "fn_single": len(fn_single),
        "fn_multi": len(fn_multi),
        "tp_count": len(tp_records),
        "fn_exist_score_stats": {
            "mean": float(np.mean([r["exist_score"] for r in fn_records])) if fn_records else 0.0,
            "median": float(np.median([r["exist_score"] for r in fn_records])) if fn_records else 0.0,
            "max": float(max([r["exist_score"] for r in fn_records])) if fn_records else 0.0,
        },
        "top20_fn_cases": top_fn_cases,
    }
    save_json(result, STATS_DIR / "fn_analysis.json")

    # Figure: FN ratio across thresholds
    thds_list = np.arange(0.3, 0.85, 0.025)
    fn_ratios = []
    fp_ratios = []
    for t in thds_list:
        fn_c = sum(1 for r in records if r["is_positive"] and r["exist_score"] <= t)
        fp_c = sum(1 for r in records if not r["is_positive"] and r["exist_score"] > t)
        fn_ratios.append(100 * fn_c / len(pos_records) if pos_records else 0)
        fp_ratios.append(100 * fp_c / sum(1 for r in records if not r["is_positive"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thds_list, fn_ratios, color="#3498db", lw=2, label="FN ratio (% of positives rejected)")
    ax.plot(thds_list, fp_ratios, color="#e74c3c", lw=2, label="FP ratio (% of negatives accepted)")
    ax.axvline(thd, color="orange", ls="--", lw=1.5, label=f"Opt τ={thd}")
    ax.set_xlabel("Threshold τ")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("FN / FP Rates vs. Threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fn_ratio_curve.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] figures/fn_ratio_curve.png")

    return result


# ─── Task 2.4: Multi-moment miss analysis ────────────────────────────────────

def analyze_multi_moment(records: List[Dict], iou_thd: float = 0.5) -> Dict:
    """Only first moment retrieved for multi-moment queries."""
    print(f"\n=== Task 2.4: Multi-moment miss (IoU threshold={iou_thd}) ===")

    multi_records = [r for r in records if r["is_multi"]]
    print(f"  Multi-moment queries: {len(multi_records)}")

    per_query = []
    for r in multi_records:
        gts = r["gt_windows"]
        preds = r["pred_windows"]  # already top-10

        # Use all 10 preds (top-5 for k=5)
        matches_k5 = greedy_match(preds[:5], gts, iou_thd=iou_thd)
        matched_gt_indices_k5 = {m[1] for m in matches_k5}

        # First moment hit: any match hits GT index 0 (chronologically first)
        gts_sorted = sorted(enumerate(gts), key=lambda x: x[1][0])
        first_gt_idx = gts_sorted[0][0]  # original index of earliest GT

        first_hit_k5 = first_gt_idx in matched_gt_indices_k5
        subsequent_hits_k5 = len(matched_gt_indices_k5 - {first_gt_idx})
        total_gts = len(gts)
        any_hit_k5 = len(matched_gt_indices_k5) > 0

        per_query.append({
            "qid": r["qid"],
            "n_gt": total_gts,
            "n_pred": len(preds),
            "n_matched_k5": len(matches_k5),
            "first_hit_k5": first_hit_k5,
            "subsequent_hits_k5": subsequent_hits_k5,
            "any_hit_k5": any_hit_k5,
            "only_first_hit": first_hit_k5 and subsequent_hits_k5 == 0,
        })

    # Summary stats
    n = len(per_query)
    any_hit = sum(p["any_hit_k5"] for p in per_query)
    first_hit = sum(p["first_hit_k5"] for p in per_query)
    only_first = sum(p["only_first_hit"] for p in per_query)
    any_subsequent = sum(1 for p in per_query if p["subsequent_hits_k5"] > 0)

    print(f"  Any hit @k=5:         {any_hit}/{n} = {100*any_hit/n:.1f}%")
    print(f"  First moment hit @k=5:{first_hit}/{n} = {100*first_hit/n:.1f}%")
    print(f"  Only first hit @k=5:  {only_first}/{n} = {100*only_first/n:.1f}%")
    print(f"  Has subsequent hits:  {any_subsequent}/{n} = {100*any_subsequent/n:.1f}%")

    # Breakdown by n_gt
    by_n_gt = defaultdict(list)
    for p in per_query:
        key = p["n_gt"] if p["n_gt"] <= 4 else "5+"
        by_n_gt[key].append(p)

    n_gt_breakdown = {}
    for key in sorted(by_n_gt.keys(), key=lambda x: int(str(x).replace("+", "99"))):
        grp = by_n_gt[key]
        n_gt_breakdown[str(key)] = {
            "count": len(grp),
            "any_hit_rate": round(100 * sum(p["any_hit_k5"] for p in grp) / len(grp), 1),
            "first_hit_rate": round(100 * sum(p["first_hit_k5"] for p in grp) / len(grp), 1),
            "subsequent_hit_rate": round(100 * sum(1 for p in grp if p["subsequent_hits_k5"] > 0) / len(grp), 1),
            "only_first_hit_rate": round(100 * sum(p["only_first_hit"] for p in grp) / len(grp), 1),
        }

    # n_pred distribution for multi-moment queries
    n_pred_dist = defaultdict(int)
    for p in per_query:
        n_pred_dist[p["n_pred"]] += 1

    # Top cases: only_first_hit (most impactful misses)
    top_miss_cases = [
        {"qid": r["qid"], "vid": r["vid"], "query": r["query"],
         "gt_windows": r["gt_windows"], "pred_windows": r["pred_windows"][:5],
         "n_gt": r["n_gt"]}
        for r in multi_records
        if any(p["qid"] == r["qid"] and p["only_first_hit"] for p in per_query)
    ][:10]

    result = {
        "n_multi_queries": n,
        "iou_threshold": iou_thd,
        "summary": {
            "any_hit_k5": {"count": any_hit, "rate": round(100*any_hit/n, 1)},
            "first_moment_hit_k5": {"count": first_hit, "rate": round(100*first_hit/n, 1)},
            "only_first_hit_k5": {"count": only_first, "rate": round(100*only_first/n, 1)},
            "has_subsequent_hits_k5": {"count": any_subsequent, "rate": round(100*any_subsequent/n, 1)},
        },
        "n_gt_breakdown": n_gt_breakdown,
        "n_pred_distribution": {str(k): v for k, v in sorted(n_pred_dist.items())},
        "top_miss_cases": top_miss_cases,
    }
    save_json(result, STATS_DIR / "multi_moment_analysis.json")

    # Figure: hit rates by n_gt
    keys = list(n_gt_breakdown.keys())
    any_rates = [n_gt_breakdown[k]["any_hit_rate"] for k in keys]
    first_rates = [n_gt_breakdown[k]["first_hit_rate"] for k in keys]
    sub_rates = [n_gt_breakdown[k]["subsequent_hit_rate"] for k in keys]
    only_first_rates = [n_gt_breakdown[k]["only_first_hit_rate"] for k in keys]
    counts = [n_gt_breakdown[k]["count"] for k in keys]

    x = np.arange(len(keys))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - 1.5*width, any_rates, width, label="Any hit @k=5", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x - 0.5*width, first_rates, width, label="First moment hit @k=5", color="#2ecc71", alpha=0.85)
    bars3 = ax.bar(x + 0.5*width, sub_rates, width, label="Has subsequent hit @k=5", color="#e67e22", alpha=0.85)
    bars4 = ax.bar(x + 1.5*width, only_first_rates, width, label="ONLY first hit (miss all others)", color="#e74c3c", alpha=0.85)

    ax.set_xlabel("Number of GT Moments per Query")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Multi-moment Hit Rates by GT Count\n(k=5 predictions, IoU≥0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"|GT|={k}\n(n={c})" for k, c in zip(keys, counts)])
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    for bar in [*bars1, *bars2, *bars3, *bars4]:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%",
                    ha="center", va="bottom", fontsize=7.5)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "multi_moment_hit_rate.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] figures/multi_moment_hit_rate.png")

    return result


# ─── Task 2.5: Over-detection analysis ───────────────────────────────────────

def analyze_over_detection(records: List[Dict], iou_thd: float = 0.5) -> Dict:
    """Extra predictions beyond matched GTs."""
    print(f"\n=== Task 2.5: Over-detection (IoU threshold={iou_thd}) ===")

    # For positive samples: count extra preds beyond matched GT
    pos_records = [r for r in records if r["is_positive"]]
    neg_records = [r for r in records if not r["is_positive"]]

    over_det_stats = []
    extra_scores_all = []
    hit_scores_all = []

    for r in pos_records:
        preds = r["pred_windows"][:10]
        preds_with_score = r["pred_windows_with_score"][:10]
        gts = r["gt_windows"]

        matches = greedy_match(preds, gts, iou_thd=iou_thd)
        matched_pred_idx = {m[0] for m in matches}

        n_matched = len(matches)
        n_extra = len(preds) - n_matched
        is_over_det = n_extra > 0

        # Scores of matched vs extra preds
        for i, pw in enumerate(preds_with_score):
            score = pw[2] if len(pw) > 2 else 1.0
            if i in matched_pred_idx:
                hit_scores_all.append(score)
            else:
                extra_scores_all.append(score)

        over_det_stats.append({
            "qid": r["qid"],
            "n_gt": len(gts),
            "n_pred": len(preds),
            "n_matched": n_matched,
            "n_extra": n_extra,
            "is_over_det": is_over_det,
        })

    n_pos = len(pos_records)
    n_over = sum(1 for s in over_det_stats if s["is_over_det"])
    n_neg_nonzero = sum(1 for r in neg_records if r["n_pred"] > 0)

    print(f"  Over-detected positives: {n_over}/{n_pos} = {100*n_over/n_pos:.1f}%")
    print(f"  Negatives with preds: {n_neg_nonzero}/{len(neg_records)} = {100*n_neg_nonzero/len(neg_records):.1f}%")
    print(f"  Hit score mean: {np.mean(hit_scores_all):.3f}, Extra score mean: {np.mean(extra_scores_all):.3f}")

    result = {
        "n_positive": n_pos,
        "n_over_detected": n_over,
        "over_det_rate_positive": round(100*n_over/n_pos, 1) if n_pos else 0,
        "n_negative": len(neg_records),
        "n_negative_with_preds": n_neg_nonzero,
        "neg_false_window_rate": round(100*n_neg_nonzero/len(neg_records), 1) if neg_records else 0,
        "hit_scores": {
            "mean": round(float(np.mean(hit_scores_all)), 4) if hit_scores_all else 0,
            "median": round(float(np.median(hit_scores_all)), 4) if hit_scores_all else 0,
        },
        "extra_scores": {
            "mean": round(float(np.mean(extra_scores_all)), 4) if extra_scores_all else 0,
            "median": round(float(np.median(extra_scores_all)), 4) if extra_scores_all else 0,
        },
        "n_extra_distribution": {
            str(k): sum(1 for s in over_det_stats if s["n_extra"] == k)
            for k in range(0, 11)
        },
    }
    save_json(result, STATS_DIR / "over_detection_analysis.json")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: extra pred score distribution vs hit pred scores
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    ax.hist(hit_scores_all, bins=bins, alpha=0.65, color="#2ecc71", density=True, label=f"Matched preds (n={len(hit_scores_all)})")
    ax.hist(extra_scores_all, bins=bins, alpha=0.65, color="#e74c3c", density=True, label=f"Extra preds (n={len(extra_scores_all)})")
    ax.set_xlabel("Prediction Confidence Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution:\nMatched vs. Extra Predictions")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: n_extra distribution
    ax = axes[1]
    n_extras = [s["n_extra"] for s in over_det_stats]
    bins2 = np.arange(-0.5, max(n_extras)+1.5, 1)
    ax.hist(n_extras, bins=bins2, color="#9b59b6", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Number of Extra (Unmatched) Predictions per Query")
    ax.set_ylabel("Count")
    ax.set_title("Over-detection: Extra Preds Distribution\n(positive queries only)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "extra_pred_score_dist.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] figures/extra_pred_score_dist.png")

    return result


# ─── Task 2.6: Boundary inaccuracy analysis ──────────────────────────────────

def analyze_boundary(records: List[Dict]) -> Dict:
    """IoU distribution and boundary offset for matched prediction-GT pairs."""
    print(f"\n=== Task 2.6: Boundary inaccuracy ===")

    all_matched_ious = []
    start_offsets = []   # pred_start - gt_start
    end_offsets = []     # pred_end - gt_end
    gt_durations = []
    near_miss = []       # 0.3 <= IoU < 0.5

    pos_records = [r for r in records if r["is_positive"]]

    for r in pos_records:
        preds = r["pred_windows"][:5]   # top-5
        gts = r["gt_windows"]
        if not preds or not gts:
            continue

        matches = greedy_match(preds, gts, iou_thd=-1.0)  # force-match for boundary analysis
        for (i_pred, j_gt, iou) in matches:
            p = preds[i_pred]
            g = gts[j_gt]
            all_matched_ious.append(iou)
            start_offsets.append(p[0] - g[0])
            end_offsets.append(p[1] - g[1])
            gt_dur = g[1] - g[0]
            gt_durations.append(gt_dur)
            if 0.3 <= iou < 0.5:
                near_miss.append(iou)

    all_ious_arr = np.array(all_matched_ious)
    start_arr = np.array(start_offsets)
    end_arr = np.array(end_offsets)

    n_total = len(all_matched_ious)
    n_near_miss = len(near_miss)

    print(f"  Total matched pairs: {n_total}")
    print(f"  Mean IoU: {np.mean(all_ious_arr):.3f}")
    print(f"  Near-miss (0.3≤IoU<0.5): {n_near_miss} = {100*n_near_miss/n_total:.1f}%")
    print(f"  Start offset: mean={np.mean(start_arr):.2f}s, std={np.std(start_arr):.2f}s")
    print(f"  End offset:   mean={np.mean(end_arr):.2f}s, std={np.std(end_arr):.2f}s")

    # IoU bins
    iou_bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    iou_bin_counts = {}
    for lo, hi in iou_bins:
        cnt = int(np.sum((all_ious_arr >= lo) & (all_ious_arr < hi)))
        iou_bin_counts[f"[{lo},{hi})"] = cnt

    # By GT duration group
    dur_groups = {"<5s": [], "5-20s": [], ">20s": []}
    for iou, dur in zip(all_matched_ious, gt_durations):
        if dur < 5:
            dur_groups["<5s"].append(iou)
        elif dur <= 20:
            dur_groups["5-20s"].append(iou)
        else:
            dur_groups[">20s"].append(iou)

    duration_iou_stats = {}
    for grp, ious in dur_groups.items():
        if ious:
            duration_iou_stats[grp] = {
                "count": len(ious),
                "mean_iou": round(float(np.mean(ious)), 4),
                "median_iou": round(float(np.median(ious)), 4),
            }

    result = {
        "n_matched_pairs": n_total,
        "mean_iou": round(float(np.mean(all_ious_arr)), 4),
        "median_iou": round(float(np.median(all_ious_arr)), 4),
        "near_miss_count": n_near_miss,
        "near_miss_rate": round(100*n_near_miss/n_total, 1) if n_total else 0,
        "start_offset_mean": round(float(np.mean(start_arr)), 3),
        "start_offset_std": round(float(np.std(start_arr)), 3),
        "end_offset_mean": round(float(np.mean(end_arr)), 3),
        "end_offset_std": round(float(np.std(end_arr)), 3),
        "iou_bin_counts": iou_bin_counts,
        "duration_iou_stats": duration_iou_stats,
    }
    save_json(result, STATS_DIR / "boundary_analysis.json")

    # Figures
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Left: IoU histogram
    ax = axes[0]
    ax.hist(all_matched_ious, bins=np.linspace(0, 1, 31), color="#3498db", alpha=0.85, edgecolor="white")
    ax.axvspan(0.3, 0.5, alpha=0.15, color="#e74c3c", label="Near-miss (0.3-0.5)")
    ax.axvline(0.5, color="gray", ls="--", lw=1.2, label="IoU=0.5")
    ax.set_xlabel("IoU")
    ax.set_ylabel("Count")
    ax.set_title(f"Matched Pair IoU Distribution\n(top-5 preds, force-match, n={n_total})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Middle: Start offset distribution
    ax = axes[1]
    clip = np.percentile(np.abs(start_arr), 95)
    ax.hist(start_arr, bins=60, range=(-clip, clip), color="#2ecc71", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="gray", ls="--", lw=1.5)
    ax.axvline(np.mean(start_arr), color="#e74c3c", ls="-", lw=1.8,
               label=f"Mean={np.mean(start_arr):.2f}s")
    ax.set_xlabel("pred_start − gt_start (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Start Boundary Offset")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: End offset distribution
    ax = axes[2]
    clip = np.percentile(np.abs(end_arr), 95)
    ax.hist(end_arr, bins=60, range=(-clip, clip), color="#e67e22", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="gray", ls="--", lw=1.5)
    ax.axvline(np.mean(end_arr), color="#e74c3c", ls="-", lw=1.8,
               label=f"Mean={np.mean(end_arr):.2f}s")
    ax.set_xlabel("pred_end − gt_end (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("End Boundary Offset")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "iou_histogram.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] figures/iou_histogram.png")

    # Additional: boundary offset scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    clip_s = np.percentile(np.abs(start_arr), 90)
    clip_e = np.percentile(np.abs(end_arr), 90)
    mask = (np.abs(start_arr) <= clip_s) & (np.abs(end_arr) <= clip_e)
    ax.scatter(start_arr[mask], end_arr[mask], alpha=0.3, s=10,
               c=all_ious_arr[np.array(range(len(all_ious_arr)))[mask]], cmap="RdYlGn",
               vmin=0, vmax=1)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Start Offset (pred_start − gt_start, s)")
    ax.set_ylabel("End Offset (pred_end − gt_end, s)")
    ax.set_title("Boundary Offset Scatter\n(color = IoU, green=high, red=low)")
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="IoU")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "boundary_offset.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] figures/boundary_offset.png")

    return result


# ─── Task 2.7: Summary ────────────────────────────────────────────────────────

def write_summary(records, fp_res, fn_res, multi_res, over_res, bnd_res):
    """Generate summary.md with the error type table and main bottleneck."""
    print(f"\n=== Task 2.7: Writing summary ===")

    n_total = len(records)
    n_pos = sum(1 for r in records if r["is_positive"])
    n_neg = sum(1 for r in records if not r["is_positive"])
    n_multi = sum(1 for r in records if r["is_multi"])

    # Estimate G-mIoU@1 impact (rough): each error type affects some fraction of total queries
    # FP: 492 neg accepted → each contributes 0 instead of 1 → loss ≈ 492/1036 * 100 = 47.5
    # (already quantified by oracle runs in Phase 3)

    lines = [
        "# Error Analysis Summary — Moment-DETR-GMR Test Set",
        "",
        f"> **Dataset**: Soccer-GMR Standard Test | {n_total} queries "
        f"({n_pos} positive: {n_multi} multi + {n_pos-n_multi} single | {n_neg} negative)",
        "",
        "---",
        "",
        "## Error Type Breakdown",
        "",
        "| # | Error Type | Affected Samples | Rate | Key Stat |",
        "|---|-----------|:----------------:|:----:|---------|",
    ]

    # 2.2 FP
    fp_at_04 = 492  # from test_results.json
    fp_at_55 = 123
    lines.append(
        f"| 2.2 | **Rejection FP** (null-set accepted) "
        f"| {fp_at_04} (τ=0.4) / {fp_at_55} (τ=0.55) "
        f"| {100*fp_at_04/n_neg:.0f}% / {100*fp_at_55/n_neg:.0f}% (of negatives) "
        f"| Best Rej-F1={fp_res['best_rej_f1']:.1f}% at τ={fp_res['best_threshold']:.2f} |"
    )

    # 2.3 FN
    lines.append(
        f"| 2.3 | **Rejection FN** (positive rejected) "
        f"| {fn_res['fn_count']} (τ=0.55) "
        f"| {fn_res['fn_ratio']:.1f}% (of positives) "
        f"| {fn_res['fn_single']} single + {fn_res['fn_multi']} multi |"
    )

    # 2.4 Multi-moment miss
    multi_only_first = multi_res["summary"]["only_first_hit_k5"]["count"]
    multi_rate = multi_res["summary"]["only_first_hit_k5"]["rate"]
    lines.append(
        f"| 2.4 | **Multi-moment miss** (only 1st retrieved) "
        f"| {multi_only_first}/{n_multi} multi queries "
        f"| {multi_rate:.1f}% "
        f"| mR+@5={0.97}% vs mR@5={14.14}% |"
    )

    # 2.5 Over-detection
    lines.append(
        f"| 2.5 | **Over-detection** (excess preds) "
        f"| {over_res['n_over_detected']}/{n_pos} positives "
        f"| {over_res['over_det_rate_positive']:.1f}% "
        f"| Extra pred score mean={over_res['extra_scores']['mean']:.3f} |"
    )

    # 2.6 Boundary
    lines.append(
        f"| 2.6 | **Boundary inaccuracy** (near-miss 0.3≤IoU<0.5) "
        f"| {bnd_res['near_miss_count']} matched pairs "
        f"| {bnd_res['near_miss_rate']:.1f}% "
        f"| Mean IoU={bnd_res['mean_iou']:.3f}, "
        f"Δstart={bnd_res['start_offset_mean']:+.2f}s, "
        f"Δend={bnd_res['end_offset_mean']:+.2f}s |"
    )

    lines += [
        "",
        "---",
        "",
        "## Score Distribution (pred_exist_score)",
        "",
        f"| | Mean | Median | Std |",
        f"|---|---:|---:|---:|",
        f"| Positive queries | {fp_res['score_stats']['positive']['mean']:.4f} | "
        f"{fp_res['score_stats']['positive']['median']:.4f} | "
        f"{fp_res['score_stats']['positive']['std']:.4f} |",
        f"| Negative queries | {fp_res['score_stats']['negative']['mean']:.4f} | "
        f"{fp_res['score_stats']['negative']['median']:.4f} | "
        f"{fp_res['score_stats']['negative']['std']:.4f} |",
        "",
        "---",
        "",
        "## Multi-moment Analysis (160 queries with |GT|≥2)",
        "",
        f"| Metric | Count | Rate |",
        f"|-------|------:|-----:|",
        f"| Any moment hit @k=5 | {multi_res['summary']['any_hit_k5']['count']} | {multi_res['summary']['any_hit_k5']['rate']}% |",
        f"| First moment hit @k=5 | {multi_res['summary']['first_moment_hit_k5']['count']} | {multi_res['summary']['first_moment_hit_k5']['rate']}% |",
        f"| ONLY first hit (miss rest) | {multi_res['summary']['only_first_hit_k5']['count']} | {multi_res['summary']['only_first_hit_k5']['rate']}% |",
        f"| Has subsequent hits | {multi_res['summary']['has_subsequent_hits_k5']['count']} | {multi_res['summary']['has_subsequent_hits_k5']['rate']}% |",
        "",
        "---",
        "",
        "## 🔍 主要瓶颈分析 (待 Phase 3 Oracle 修复后确认)",
        "",
        "基于当前 error analysis 数据的初步判断:",
        "",
        "1. **拒识误报 (FP)** — 所有492个负样本在默认阈值下被误接受, "
        "直接导致 G-mIoU@1 从39.31降至4.49 (35分差距)。"
        "这是 G-mIoU 退化的**直接原因**, 但通过调阈值即可部分修复, 属于 calibration 问题。",
        "",
        "2. **多时刻漏检 (Multi-miss)** — mR+@5仅0.97%, 与mR@5=14.14的巨大差距"
        "说明模型几乎完全无法检索多时刻。"
        f"在{n_multi}个多时刻查询中, 仅{multi_res['summary']['has_subsequent_hits_k5']['count']}个"
        f"({multi_res['summary']['has_subsequent_hits_k5']['rate']}%)命中了后续时刻。"
        "这是**定位层面**的核心瓶颈, 且与 FlashVTG-GMR 的差距最大 (mR+@5: 0.97 vs 19.10)。",
        "",
        "3. **边界不准** — 平均 IoU 较低, 存在一定比例的近失样本可通过 refinement 提升。",
        "",
        "**初步结论**: "
        "多时刻漏检是需要模型层面改进的最大瓶颈 (mR+差距18倍); "
        "拒识误报虽然对 G-mIoU 数值影响最大, 但阈值调整即可修复。"
        "需等待 Phase 3 Oracle 排序来量化各类错误对指标的贡献。",
        "",
        "---",
        "",
        "## 图表索引",
        "",
        "| 图表 | 路径 |",
        "|------|------|",
        "| Existence score分布 + FP/FN曲线 | figures/exist_score_distribution.png |",
        "| FN/FP率随阈值变化 | figures/fn_ratio_curve.png |",
        "| 多时刻命中率 | figures/multi_moment_hit_rate.png |",
        "| 多余预测分数分布 | figures/extra_pred_score_dist.png |",
        "| IoU直方图 + 边界偏移 | figures/iou_histogram.png |",
        "| 边界偏移散点图 | figures/boundary_offset.png |",
    ]

    summary_path = OUT_DIR / "summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [saved] error_analysis/summary.md")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 2: Systematic Error Analysis")
    print("=" * 60)

    records = load_data()

    # Run all analyses
    fp_res = analyze_rejection_fp(records)
    fn_res = analyze_rejection_fn(records, thd=0.55)
    multi_res = analyze_multi_moment(records, iou_thd=0.5)
    over_res = analyze_over_detection(records, iou_thd=0.5)
    bnd_res = analyze_boundary(records)

    # Summary
    write_summary(records, fp_res, fn_res, multi_res, over_res, bnd_res)

    print("\n" + "=" * 60)
    print("Phase 2 COMPLETE. Results in results/moment_detr_gmr/error_analysis/")
    print("=" * 60)


if __name__ == "__main__":
    main()
