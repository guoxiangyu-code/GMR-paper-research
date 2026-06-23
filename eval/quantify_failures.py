# -*- coding: utf-8 -*-
"""
Phase 2: Quantify six failure modes from the error detail table.

Produces summary statistics, distribution data, and plots for:
1. Rejection FP (null-set accepted)
2. Rejection FN (positive incorrectly rejected)
3. Multi-moment miss (only first hit)
4. Multiple detections (|pred| > |G|)
5. Inaccurate boundaries (IoU histogram, near-miss)
6. Summary bottleneck identification
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_detail(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    return rows


def quantify_rejection_fp(rows: List[Dict], threshold: float = 0.4) -> Dict[str, Any]:
    null_rows = [r for r in rows if r["scenario"] == "null_set"]
    if not null_rows:
        return {"mode": "rejection_FP", "count": 0}

    scores = np.array([r["pred_exist_score"] for r in null_rows])
    accepted = scores > threshold
    fp_count = int(accepted.sum())
    tn_count = int((~accepted).sum())

    thresholds = np.arange(0.2, 0.8, 0.01)
    fp_curve = []
    for t in thresholds:
        fp_curve.append({"threshold": round(float(t), 2), "FP_count": int((scores > t).sum())})

    top_fp = sorted(null_rows, key=lambda r: r["pred_exist_score"], reverse=True)[:10]
    top_fp_summary = [{"qid": r["qid"], "exist_score": r["pred_exist_score"]} for r in top_fp]

    return {
        "mode": "rejection_FP",
        "total_null_set": len(null_rows),
        "FP_count_at_0.4": fp_count,
        "TN_count_at_0.4": tn_count,
        "FP_rate_at_0.4": round(fp_count / len(null_rows), 4),
        "score_distribution": {
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
            "median": round(float(np.median(scores)), 4),
        },
        "FP_curve": fp_curve,
        "top_10_FP_qids": top_fp_summary,
    }


def quantify_rejection_fn(rows: List[Dict], threshold: float = 0.4) -> Dict[str, Any]:
    pos_rows = [r for r in rows if r["scenario"] != "null_set"]
    if not pos_rows:
        return {"mode": "rejection_FN", "count": 0}

    scores = np.array([r["pred_exist_score"] for r in pos_rows])
    rejected = scores <= threshold
    fn_count = int(rejected.sum())
    tp_count = int((~rejected).sum())

    fn_rows = [r for r in pos_rows if r["pred_exist_score"] <= threshold]
    fn_by_scenario = defaultdict(int)
    for r in fn_rows:
        fn_by_scenario[r["scenario"]] += 1

    return {
        "mode": "rejection_FN",
        "total_positive": len(pos_rows),
        "FN_count_at_0.4": fn_count,
        "TP_count_at_0.4": tp_count,
        "FN_rate_at_0.4": round(fn_count / len(pos_rows), 4),
        "FN_by_scenario": dict(fn_by_scenario),
        "score_distribution": {
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
        },
    }


def quantify_multi_moment_miss(rows: List[Dict]) -> Dict[str, Any]:
    multi_rows = [r for r in rows if r["scenario"] == "multi"]
    if not multi_rows:
        return {"mode": "multi_moment_miss", "count": 0}

    total_gt_moments = sum(r["n_gt"] for r in multi_rows)
    total_matched_iou50 = sum(r["match_count_iou50"] for r in multi_rows)
    total_missed_iou50 = sum(r["miss_count_iou50"] for r in multi_rows)

    hit_first = 0
    hit_rest = 0
    total_first = 0
    total_rest = 0
    only_first_hit = 0

    for r in multi_rows:
        n_gt = r["n_gt"]
        total_first += 1
        total_rest += (n_gt - 1)
        if r.get("best_iou_first_gt", 0) >= 0.5:
            hit_first += 1
        if r["match_count_iou50"] >= 2:
            hit_rest += min(r["match_count_iou50"] - 1, n_gt - 1)
        if r["match_count_iou50"] == 1:
            only_first_hit += 1

    first_hit_rate = round(hit_first / total_first, 4) if total_first > 0 else 0.0
    rest_hit_rate = round(hit_rest / total_rest, 4) if total_rest > 0 else 0.0
    only_first_proportion = round(only_first_hit / len(multi_rows), 4)

    error_breakdown = defaultdict(int)
    for r in multi_rows:
        error_breakdown[r["error_type_iou50"]] += 1

    return {
        "mode": "multi_moment_miss",
        "num_multi_queries": len(multi_rows),
        "total_gt_moments": total_gt_moments,
        "total_matched_iou50": total_matched_iou50,
        "total_missed_iou50": total_missed_iou50,
        "first_moment_hit_rate_iou50": first_hit_rate,
        "rest_moments_hit_rate_iou50": rest_hit_rate,
        "only_first_hit_proportion": only_first_proportion,
        "multi_recall_iou50": round(total_matched_iou50 / total_gt_moments, 4),
        "error_breakdown": dict(error_breakdown),
    }


def quantify_multiple_detections(rows: List[Dict]) -> Dict[str, Any]:
    pos_rows = [r for r in rows if r["scenario"] != "null_set"]

    extra_count = 0
    extra_pred_counts = []
    for r in pos_rows:
        extra = r["extra_count_iou50"]
        if extra > 0:
            extra_count += 1
            extra_pred_counts.append(extra)

    multi_det_rate = round(extra_count / len(pos_rows), 4) if pos_rows else 0.0

    return {
        "mode": "multiple_detections",
        "num_positive": len(pos_rows),
        "queries_with_extra_preds_iou50": extra_count,
        "multi_det_rate_iou50": multi_det_rate,
        "extra_pred_distribution": {
            "mean": round(float(np.mean(extra_pred_counts)), 2) if extra_pred_counts else 0,
            "max": int(np.max(extra_pred_counts)) if extra_pred_counts else 0,
            "total_extra": int(sum(extra_pred_counts)),
        },
    }


def quantify_inaccurate_boundaries(rows: List[Dict]) -> Dict[str, Any]:
    pos_rows = [r for r in rows if r["scenario"] != "null_set"]

    all_ious = []
    for r in pos_rows:
        all_ious.extend(r["matched_ious_forced"])

    if not all_ious:
        return {"mode": "inaccurate_boundaries", "num_matched_pairs": 0}

    ious = np.array(all_ious)
    near_miss = int(((ious >= 0.3) & (ious < 0.5)).sum())
    good_match = int((ious >= 0.5).sum())
    poor_match = int((ious < 0.3).sum())

    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(ious, bins=bins)
    iou_histogram = {}
    for i in range(len(bins) - 1):
        key = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
        iou_histogram[key] = int(hist[i])

    return {
        "mode": "inaccurate_boundaries",
        "num_matched_pairs": len(all_ious),
        "iou_stats": {
            "mean": round(float(ious.mean()), 4),
            "median": round(float(np.median(ious)), 4),
            "std": round(float(ious.std()), 4),
        },
        "near_miss_0.3_0.5": near_miss,
        "near_miss_pct": round(near_miss / len(all_ious), 4),
        "good_match_ge_0.5": good_match,
        "good_match_pct": round(good_match / len(all_ious), 4),
        "poor_match_lt_0.3": poor_match,
        "poor_match_pct": round(poor_match / len(all_ious), 4),
        "iou_histogram": iou_histogram,
    }


def identify_bottleneck(results: List[Dict[str, Any]], total_queries: int) -> Dict[str, Any]:
    impacts = []

    for r in results:
        mode = r["mode"]
        if mode == "rejection_FP":
            fp = r.get("FP_count_at_0.4", 0)
            impacts.append({"mode": mode, "affected_queries": fp,
                            "pct_of_total": round(fp / total_queries, 4)})
        elif mode == "rejection_FN":
            fn = r.get("FN_count_at_0.4", 0)
            impacts.append({"mode": mode, "affected_queries": fn,
                            "pct_of_total": round(fn / total_queries, 4)})
        elif mode == "multi_moment_miss":
            missed = r.get("total_missed_iou50", 0)
            multi_q = r.get("num_multi_queries", 0)
            only_first = r.get("only_first_hit_proportion", 0)
            impacts.append({"mode": mode, "affected_queries": multi_q,
                            "missed_gt_moments": missed,
                            "only_first_hit_pct": only_first,
                            "pct_of_total": round(multi_q / total_queries, 4)})
        elif mode == "multiple_detections":
            extra_q = r.get("queries_with_extra_preds_iou50", 0)
            impacts.append({"mode": mode, "affected_queries": extra_q,
                            "pct_of_total": round(extra_q / total_queries, 4)})
        elif mode == "inaccurate_boundaries":
            near_miss = r.get("near_miss_0.3_0.5", 0)
            impacts.append({"mode": mode, "affected_queries": near_miss,
                            "near_miss_pct": r.get("near_miss_pct", 0),
                            "pct_of_total": round(near_miss / total_queries, 4)})

    impacts.sort(key=lambda x: x["pct_of_total"], reverse=True)

    primary = impacts[0]["mode"] if impacts else "unknown"
    return {
        "total_queries": total_queries,
        "ranked_bottlenecks": impacts,
        "primary_bottleneck": primary,
        "summary_sentence": f"Primary bottleneck: {primary} (affects {impacts[0]['pct_of_total']*100:.1f}% of queries)" if impacts else "No bottleneck identified",
    }


def plot_failure_distributions(results: List[Dict], output_dir: str) -> None:
    if not HAS_MPL:
        print("[quantify_failures] matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    for r in results:
        mode = r["mode"]

        if mode == "rejection_FP" and "FP_curve" in r:
            fig, ax = plt.subplots(figsize=(8, 4))
            curve = r["FP_curve"]
            thresholds = [c["threshold"] for c in curve]
            fp_counts = [c["FP_count"] for c in curve]
            ax.plot(thresholds, fp_counts, "b-", linewidth=2)
            ax.axvline(x=0.4, color="r", linestyle="--", label="tau=0.4")
            ax.set_xlabel("Existence Score Threshold")
            ax.set_ylabel("False Positive Count")
            ax.set_title("Rejection FP Curve: Null-Set Accepted vs Threshold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "rejection_fp_curve.png"), dpi=150)
            plt.close(fig)

        elif mode == "inaccurate_boundaries" and "iou_histogram" in r:
            fig, ax = plt.subplots(figsize=(8, 4))
            hist = r["iou_histogram"]
            labels = list(hist.keys())
            values = list(hist.values())
            ax.bar(labels, values, color="steelblue", edgecolor="black", alpha=0.8)
            ax.set_xlabel("IoU Range")
            ax.set_ylabel("Count")
            ax.set_title("IoU Histogram of Matched Prediction-GT Pairs")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "iou_histogram.png"), dpi=150)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    error_labels = []
    error_counts = []
    for r in results:
        mode = r["mode"]
        if mode == "rejection_FP":
            error_labels.append("Rej FP")
            error_counts.append(r.get("FP_count_at_0.4", 0))
        elif mode == "rejection_FN":
            error_labels.append("Rej FN")
            error_counts.append(r.get("FN_count_at_0.4", 0))
        elif mode == "multi_moment_miss":
            error_labels.append("Multi-miss")
            error_counts.append(r.get("total_missed_iou50", 0))
        elif mode == "multiple_detections":
            error_labels.append("Multi-det")
            error_counts.append(r.get("queries_with_extra_preds_iou50", 0))
        elif mode == "inaccurate_boundaries":
            error_labels.append("Near-miss\n(IoU 0.3-0.5)")
            error_counts.append(r.get("near_miss_0.3_0.5", 0))

    if error_labels:
        bars = ax.bar(error_labels, error_counts, color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71", "#9b59b6"][:len(error_labels)])
        ax.set_ylabel("Affected Queries / Pairs")
        ax.set_title("Failure Mode Impact Comparison")
        for bar, count in zip(bars, error_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    str(count), ha="center", va="bottom", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "failure_mode_comparison.png"), dpi=150)
        plt.close(fig)

    print(f"[quantify_failures] Plots saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Quantify failure modes")
    parser.add_argument("--detail_path", type=str, required=True, help="Path to error detail JSONL")
    parser.add_argument("--output_dir", type=str, default="results/error_analysis")
    parser.add_argument("--exist_threshold", type=float, default=0.4)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    rows = load_detail(args.detail_path)
    os.makedirs(args.output_dir, exist_ok=True)

    results = [
        quantify_rejection_fp(rows, threshold=args.exist_threshold),
        quantify_rejection_fn(rows, threshold=args.exist_threshold),
        quantify_multi_moment_miss(rows),
        quantify_multiple_detections(rows),
        quantify_inaccurate_boundaries(rows),
    ]

    bottleneck = identify_bottleneck(results, total_queries=len(rows))
    results.append(bottleneck)

    for r in results:
        mode = r.get("mode", "unknown")
        print(f"\n=== {mode} ===")
        print(json.dumps(r, indent=2, ensure_ascii=False))

    output_json = os.path.join(args.output_dir, f"{args.split}_failure_modes.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n[quantify_failures] Saved -> {output_json}")

    plot_dir = os.path.join(args.output_dir, "plots")
    plot_failure_distributions(results, plot_dir)


if __name__ == "__main__":
    main()
