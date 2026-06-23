# -*- coding: utf-8 -*-
"""
Per-query error analysis for Soccer-GMR.

Builds a detail table: per-qid scenario label, matched/missed/extra counts,
per-match IoU, and existence-score statistics. Reuses matching functions
from eval/metrics.py without modifying official eval logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from metrics import (
    DEFAULT_IOU_THRESHOLDS,
    greedy_match,
    get_existence_score,
)
from utils import load_jsonl


def assign_scenario(gt_windows: List) -> str:
    if len(gt_windows) == 0:
        return "null_set"
    if len(gt_windows) == 1:
        return "single"
    return "multi"


def classify_error(
    scenario: str,
    match_count: int,
    miss_count: int,
    extra_count: int,
    exist_score: float,
    threshold: float,
) -> str:
    if scenario == "null_set":
        if exist_score > threshold:
            return "rejection_FP"
        return "correct_reject"
    if scenario == "single":
        if match_count > 0:
            if extra_count > 0:
                return "single_hit_plus_extra"
            return "single_hit"
        return "single_miss"
    if scenario == "multi":
        if match_count == 0:
            return "multi_total_miss"
        if miss_count > 0:
            return "multi_partial_miss"
        if extra_count > 0:
            return "multi_hit_plus_extra"
        return "multi_full_hit"
    return "unknown"


def build_detail_table(
    submission: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_thresholds: np.ndarray = DEFAULT_IOU_THRESHOLDS,
    exist_threshold: float = 0.4,
) -> List[Dict[str, Any]]:
    pred_by_qid: Dict[Any, Dict] = {d["qid"]: d for d in submission}

    detail_rows: List[Dict[str, Any]] = []

    for gt_entry in ground_truth:
        qid = gt_entry["qid"]
        gt_windows = gt_entry.get("relevant_windows", [])
        scenario = assign_scenario(gt_windows)
        n_gt = len(gt_windows)

        pred_entry = pred_by_qid.get(qid, {})
        pred_windows_raw = pred_entry.get("pred_relevant_windows", [])
        pred_windows = [[w[0], w[1]] for w in pred_windows_raw if len(w) >= 2][:10]
        pred_scores = [w[2] if len(w) > 2 else 1.0 for w in pred_windows_raw[:10]]
        n_pred = len(pred_windows)

        exist_score, score_source = get_existence_score(pred_entry)

        match_results = {}
        for thd in iou_thresholds:
            thd_key = f"{thd:.2f}"
            matches = greedy_match(pred_windows, gt_windows, iou_thd=float(thd))
            match_results[thd_key] = matches

        thd_050_matches = match_results.get("0.50", [])
        match_count_050 = len(thd_050_matches)
        miss_count_050 = max(0, n_gt - match_count_050) if scenario != "null_set" else 0
        extra_count_050 = max(0, n_pred - match_count_050) if scenario != "null_set" else 0

        forced_matches = greedy_match(pred_windows, gt_windows, iou_thd=-1.0)
        matched_ious = [m[2] for m in forced_matches]

        avg_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
        max_iou = float(np.max(matched_ious)) if matched_ious else 0.0

        per_thd_summary = {}
        for thd in iou_thresholds:
            thd_key = f"{thd:.2f}"
            matches = match_results[thd_key]
            per_thd_summary[thd_key] = {
                "match_count": len(matches),
                "miss_count": max(0, n_gt - len(matches)) if scenario != "null_set" else 0,
                "extra_count": max(0, n_pred - len(matches)) if scenario != "null_set" else 0,
            }

        error_label = classify_error(
            scenario, match_count_050, miss_count_050, extra_count_050, exist_score, exist_threshold
        )

        row = {
            "qid": qid,
            "scenario": scenario,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "match_count_iou50": match_count_050,
            "miss_count_iou50": miss_count_050,
            "extra_count_iou50": extra_count_050,
            "avg_iou_forced": round(avg_iou, 4),
            "max_iou_forced": round(max_iou, 4),
            "pred_exist_score": round(exist_score, 6),
            "score_source": score_source,
            "error_type_iou50": error_label,
            "per_thd_summary": per_thd_summary,
            "matched_ious_forced": [round(v, 4) for v in matched_ious],
        }

        if scenario == "multi" and n_gt >= 2:
            first_gt = gt_windows[0]
            rest_gts = gt_windows[1:]
            from utils import compute_temporal_iou_batch_cross
            first_pred_arr = np.array(pred_windows[:1]) if pred_windows else np.empty((0, 2))
            first_gt_arr = np.array([first_gt])
            rest_gt_arr = np.array(rest_gts) if rest_gts else np.empty((0, 2))

            if len(first_pred_arr) > 0 and len(first_gt_arr) > 0:
                iou_first, _ = compute_temporal_iou_batch_cross(first_pred_arr, first_gt_arr)
                best_first_iou = float(iou_first.max()) if iou_first.size > 0 else 0.0
            else:
                best_first_iou = 0.0

            if len(first_pred_arr) > 0 and len(rest_gt_arr) > 0:
                iou_rest, _ = compute_temporal_iou_batch_cross(first_pred_arr, rest_gt_arr)
                best_rest_iou = float(iou_rest.max()) if iou_rest.size > 0 else 0.0
            else:
                best_rest_iou = 0.0

            row["best_iou_first_gt"] = round(best_first_iou, 4)
            row["best_iou_rest_gts"] = round(best_rest_iou, 4)

        detail_rows.append(row)

    return detail_rows


def summarize_error_types(detail_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = defaultdict(int)
    for row in detail_rows:
        counts[row["error_type_iou50"]] += 1

    scenario_counts = defaultdict(int)
    for row in detail_rows:
        scenario_counts[row["scenario"]] += 1

    iou_hist = {"0.0-0.1": 0, "0.1-0.2": 0, "0.2-0.3": 0, "0.3-0.4": 0,
                "0.4-0.5": 0, "0.5-0.6": 0, "0.6-0.7": 0, "0.7-0.8": 0,
                "0.8-0.9": 0, "0.9-1.0": 0}
    for row in detail_rows:
        for iou_val in row["matched_ious_forced"]:
            bucket = int(iou_val * 10)
            bucket = min(bucket, 9)
            key = f"{bucket/10:.1f}-{(bucket+1)/10:.1f}"
            if key in iou_hist:
                iou_hist[key] += 1

    null_set_scores = [row["pred_exist_score"] for row in detail_rows if row["scenario"] == "null_set"]
    pos_scores = [row["pred_exist_score"] for row in detail_rows if row["scenario"] != "null_set"]

    return {
        "error_type_counts": dict(counts),
        "scenario_counts": dict(scenario_counts),
        "iou_histogram_forced_matches": iou_hist,
        "null_set_exist_scores": {
            "count": len(null_set_scores),
            "mean": round(float(np.mean(null_set_scores)), 4) if null_set_scores else 0,
            "std": round(float(np.std(null_set_scores)), 4) if null_set_scores else 0,
            "min": round(float(np.min(null_set_scores)), 4) if null_set_scores else 0,
            "max": round(float(np.max(null_set_scores)), 4) if null_set_scores else 0,
        },
        "positive_exist_scores": {
            "count": len(pos_scores),
            "mean": round(float(np.mean(pos_scores)), 4) if pos_scores else 0,
            "std": round(float(np.std(pos_scores)), 4) if pos_scores else 0,
            "min": round(float(np.min(pos_scores)), 4) if pos_scores else 0,
            "max": round(float(np.max(pos_scores)), 4) if pos_scores else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-query error analysis for Soccer-GMR")
    parser.add_argument("--submission_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/error_analysis")
    parser.add_argument("--exist_threshold", type=float, default=0.4)
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    args = parser.parse_args()

    submission = load_jsonl(args.submission_path)
    gt = load_jsonl(args.gt_path)

    os.makedirs(args.output_dir, exist_ok=True)

    detail_rows = build_detail_table(
        submission, gt,
        iou_thresholds=DEFAULT_IOU_THRESHOLDS,
        exist_threshold=args.exist_threshold,
    )

    summary = summarize_error_types(detail_rows)

    detail_jsonl = os.path.join(args.output_dir, f"{args.split}_error_detail.jsonl")
    with open(detail_jsonl, "w", encoding="utf-8") as f:
        for row in detail_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    detail_csv = os.path.join(args.output_dir, f"{args.split}_error_detail.csv")
    csv_fields = [
        "qid", "scenario", "n_gt", "n_pred", "match_count_iou50",
        "miss_count_iou50", "extra_count_iou50", "avg_iou_forced",
        "max_iou_forced", "pred_exist_score", "score_source", "error_type_iou50",
    ]
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_json = os.path.join(args.output_dir, f"{args.split}_error_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"[error_analysis] {args.split}: {len(detail_rows)} queries analyzed")
    print(f"[error_analysis] Error type counts: {summary['error_type_counts']}")
    print(f"[error_analysis] Scenario counts: {summary['scenario_counts']}")
    print(f"[error_analysis] Detail -> {detail_jsonl}")
    print(f"[error_analysis] CSV   -> {detail_csv}")
    print(f"[error_analysis] Summary -> {summary_json}")


if __name__ == "__main__":
    main()
