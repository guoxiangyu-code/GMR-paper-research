# -*- coding: utf-8 -*-
"""
Phase 3: Counterfactual (Oracle) Fixes to Rank Bottlenecks.

Apply oracle fixes to prediction files and re-evaluate with official metrics.
Each fix isolates one failure mode, measuring its independent contribution.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from copy import deepcopy
from typing import Any, Dict, List

from metrics import DEFAULT_IOU_THRESHOLDS, greedy_match
from eval_main import evaluate_gmr
from utils import load_jsonl


def load_sub_gt(sub_path: str, gt_path: str):
    submission = load_jsonl(sub_path)
    gt = load_jsonl(gt_path)
    pred_qids = {e["qid"] for e in submission if isinstance(e, dict) and "qid" in e}
    gt_qids = {e["qid"] for e in gt}
    shared = pred_qids & gt_qids
    submission = [e for e in submission if e.get("qid") in shared]
    gt = [e for e in gt if e.get("qid") in shared]
    return submission, gt


def fix_multi_moment_miss(submission: List[Dict], gt: List[Dict]) -> List[Dict]:
    """Oracle: add missed GT moments as perfect predictions (IoU=1.0) with high score."""
    gt_by_qid = {d["qid"]: d for d in gt}
    fixed = deepcopy(submission)

    for pred in fixed:
        qid = pred["qid"]
        gt_entry = gt_by_qid.get(qid)
        if not gt_entry or len(gt_entry.get("relevant_windows", [])) < 2:
            continue

        gts = gt_entry["relevant_windows"]
        existing_preds = pred.get("pred_relevant_windows", [])
        pred_windows = [[w[0], w[1]] for w in existing_preds[:10]]

        matches = greedy_match(pred_windows, gts, iou_thd=0.5)
        matched_gt_indices = {m[1] for m in matches}

        new_preds = list(existing_preds[:10])
        max_score = max((w[2] for w in existing_preds[:10]), default=1.0)

        for i, gt_win in enumerate(gts):
            if i not in matched_gt_indices:
                new_preds.append([gt_win[0], gt_win[1], max_score * 0.99])

        pred["pred_relevant_windows"] = new_preds
    return fixed


def fix_boundaries(submission: List[Dict], gt: List[Dict]) -> List[Dict]:
    """Oracle: set matched predictions to exactly match GT (IoU=1.0)."""
    gt_by_qid = {d["qid"]: d for d in gt}
    fixed = deepcopy(submission)

    for pred in fixed:
        qid = pred["qid"]
        gt_entry = gt_by_qid.get(qid)
        if not gt_entry or len(gt_entry.get("relevant_windows", [])) == 0:
            continue

        gts = gt_entry["relevant_windows"]
        existing_preds = pred.get("pred_relevant_windows", [])
        pred_windows = [[w[0], w[1]] for w in existing_preds[:10]]

        matches = greedy_match(pred_windows, gts, iou_thd=-1.0)

        used_gt = set()
        new_preds = list(existing_preds)
        for pred_idx, gt_idx, iou in matches:
            if pred_idx < len(new_preds) and gt_idx not in used_gt:
                new_preds[pred_idx] = [gts[gt_idx][0], gts[gt_idx][1], new_preds[pred_idx][2]]
                used_gt.add(gt_idx)

        pred["pred_relevant_windows"] = new_preds
    return fixed


def fix_false_positives(submission: List[Dict], gt: List[Dict]) -> List[Dict]:
    """Oracle: set exist_score=0 for all null-set queries (correct rejection)."""
    gt_by_qid = {d["qid"]: d for d in gt}
    fixed = deepcopy(submission)

    for pred in fixed:
        qid = pred["qid"]
        gt_entry = gt_by_qid.get(qid)
        if gt_entry and len(gt_entry.get("relevant_windows", [])) == 0:
            pred["pred_exist_score"] = 0.0
    return fixed


def fix_multiple_detections(submission: List[Dict], gt: List[Dict]) -> List[Dict]:
    """Oracle: remove all predictions not matched to a GT at IoU>0."""
    gt_by_qid = {d["qid"]: d for d in gt}
    fixed = deepcopy(submission)

    for pred in fixed:
        qid = pred["qid"]
        gt_entry = gt_by_qid.get(qid)
        if not gt_entry or len(gt_entry.get("relevant_windows", [])) == 0:
            continue

        gts = gt_entry["relevant_windows"]
        existing_preds = pred.get("pred_relevant_windows", [])
        pred_windows = [[w[0], w[1]] for w in existing_preds[:10]]

        matches = greedy_match(pred_windows, gts, iou_thd=0.0)
        matched_pred_indices = {m[0] for m in matches}

        new_preds = [w for i, w in enumerate(existing_preds) if i in matched_pred_indices]
        pred["pred_relevant_windows"] = new_preds
    return fixed


def run_counterfactual(
    name: str,
    fixed_sub: List[Dict],
    gt: List[Dict],
    split: str,
) -> Dict[str, Any]:
    results = evaluate_gmr(fixed_sub, gt, verbose=False)
    brief = results["brief"]
    print(f"[counterfactual] {name}: {json.dumps(brief, ensure_ascii=False)}")
    return {"fix": name, "metrics": brief}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Counterfactual oracle fixes")
    parser.add_argument("--submission_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/error_analysis")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    submission, gt = load_sub_gt(args.submission_path, args.gt_path)

    print(f"[counterfactual] Baseline evaluation...")
    baseline = run_counterfactual("baseline", deepcopy(submission), gt, args.split)

    print(f"\n[counterfactual] Fix 1: Multi-moment misses (add missed GTs)...")
    fix1_sub = fix_multi_moment_miss(deepcopy(submission), gt)
    fix1 = run_counterfactual("fix_multi_moment", fix1_sub, gt, args.split)

    print(f"\n[counterfactual] Fix 2: Boundaries (set matched preds to GT)...")
    fix2_sub = fix_boundaries(deepcopy(submission), gt)
    fix2 = run_counterfactual("fix_boundaries", fix2_sub, gt, args.split)

    print(f"\n[counterfactual] Fix 3: False positives (correctly reject null-set)...")
    fix3_sub = fix_false_positives(deepcopy(submission), gt)
    fix3 = run_counterfactual("fix_false_positives", fix3_sub, gt, args.split)

    print(f"\n[counterfactual] Fix 4: Multiple detections (remove unmatched preds)...")
    fix4_sub = fix_multiple_detections(deepcopy(submission), gt)
    fix4 = run_counterfactual("fix_multiple_detections", fix4_sub, gt, args.split)

    all_results = [baseline, fix1, fix2, fix3, fix4]

    base_metrics = baseline["metrics"]
    comparison = []
    for r in all_results:
        entry = {"fix": r["fix"]}
        gains = {}
        for key in ["AUROC", "Rej-F1@0.4", "G-mIoU@1", "mAP", "mR@5", "mR+@5", "mIoU@1"]:
            base_val = base_metrics.get(key, 0)
            fix_val = r["metrics"].get(key, 0)
            gains[key] = round(fix_val - base_val, 2)
        entry["gains"] = gains
        entry["absolute"] = r["metrics"]
        comparison.append(entry)

    output_json = os.path.join(args.output_dir, f"{args.split}_counterfactual.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("COUNTERFACTUAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Fix':<25} {'AUROC':>8} {'Rej-F1':>8} {'G-mIoU':>8} {'mAP':>8} {'mR@5':>8} {'mR+@5':>8} {'mIoU@1':>8}")
    print("-" * 80)
    for entry in comparison:
        gains = entry["gains"]
        fix_name = entry["fix"]
        print(f"{fix_name:<25} {gains['AUROC']:>+8.2f} {gains['Rej-F1@0.4']:>+8.2f} "
              f"{gains['G-mIoU@1']:>+8.2f} {gains['mAP']:>+8.2f} {gains['mR@5']:>+8.2f} "
              f"{gains['mR+@5']:>+8.2f} {gains['mIoU@1']:>+8.2f}")

    ranked = sorted(comparison[1:], key=lambda x: sum(abs(v) for v in x["gains"].values()), reverse=True)
    print(f"\nRanked by total metric gain:")
    for i, entry in enumerate(ranked):
        total = round(sum(abs(v) for v in entry["gains"].values()), 2)
        print(f"  {i+1}. {entry['fix']}: total_gain={total}")

    print(f"\n[counterfactual] Saved -> {output_json}")


if __name__ == "__main__":
    main()
