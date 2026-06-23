# -*- coding: utf-8 -*-
"""
Phase 4: Comprehensive evaluation comparing baseline vs calibrated module.

Produces a full comparison table and ablation study.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MODELS_DIR = os.path.join(REPO_ROOT, "models", "moment_detr_gmr")
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from utils import load_jsonl
from eval_main import evaluate_gmr
from exist_calibrator import ExistenceCalibrator


def run_eval(name: str, sub: List[Dict], gt: List[Dict]) -> Dict[str, Any]:
    results = evaluate_gmr(sub, gt, verbose=False)
    brief = results["brief"]
    print(f"  {name}: {json.dumps(brief, ensure_ascii=False)}")
    return {"name": name, "metrics": brief}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Comparison and ablation")
    parser.add_argument("--val_sub_path", type=str,
                        default="results/moment_detr_gmr/val/moment_detr_gmr_val_submission.jsonl")
    parser.add_argument("--val_gt_path", type=str,
                        default="data/label/Standard/val.jsonl")
    parser.add_argument("--test_sub_path", type=str,
                        default="results/moment_detr_gmr/test/moment_detr_gmr_test_submission.jsonl")
    parser.add_argument("--test_gt_path", type=str,
                        default="data/label/Standard/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/error_analysis/phase4")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    val_sub = load_jsonl(args.val_sub_path)
    val_gt = load_jsonl(args.val_gt_path)
    test_sub = load_jsonl(args.test_sub_path)
    test_gt = load_jsonl(args.test_gt_path)

    # === TEST SET EVALUATION ===
    print("=" * 80)
    print("TEST SET COMPARISON")
    print("=" * 80)

    # 1. Baseline (no calibration)
    print("\n1. Baseline (threshold=0.4):")
    baseline_test = run_eval("baseline_thd04", deepcopy(test_sub), test_gt)

    # 2. Calibrated (full module)
    print("\n2. Calibrated (temp scaling + adaptive threshold):")
    calibrator = ExistenceCalibrator.fit(val_sub, val_gt, method="max_rej_f1")
    test_calibrated = calibrator.apply_to_predictions(test_sub)
    calibrated_test = run_eval("calibrated", test_calibrated, test_gt)

    # 3. Ablation: temperature only (keep threshold=0.4)
    print("\n3. Ablation: temp scaling only (thd=0.4):")
    calibrator_temp_only = ExistenceCalibrator(
        temperature=calibrator.temperature, threshold=0.4
    )
    test_temp_only = calibrator_temp_only.apply_to_predictions(test_sub)
    temp_only_test = run_eval("temp_only_thd04", test_temp_only, test_gt)

    # 4. Ablation: threshold only (temp=1.0, just change threshold)
    print("\n4. Ablation: threshold only (temp=1.0):")
    calibrator_thd_only = ExistenceCalibrator(temperature=1.0, threshold=0.6)
    test_thd_only = calibrator_thd_only.apply_to_predictions(test_sub)
    thd_only_test = run_eval("thd_only_0.6", test_thd_only, test_gt)

    # 5. Ablation: hard gate without score scaling
    print("\n5. Ablation: hard gate (no score scaling):")
    calibrator_hard = deepcopy(calibrator)
    test_hard = deepcopy(test_sub)
    from exist_calibrator import ExistenceCalibrator as EC
    for pred in test_hard:
        score = pred.get("pred_exist_score", 0.0)
        cal_score = calibrator_hard.calibrate_single(score)
        pred["pred_exist_score"] = round(cal_score, 6)
        if cal_score <= calibrator_hard.threshold:
            pred["pred_relevant_windows"] = [[w[0], w[1], 0.0] for w in pred.get("pred_relevant_windows", [])]
    hard_test = run_eval("hard_gate_no_scale", test_hard, test_gt)

    # === VAL SET EVALUATION ===
    print("\n" + "=" * 80)
    print("VAL SET COMPARISON")
    print("=" * 80)

    print("\n1. Baseline:")
    baseline_val = run_eval("baseline", deepcopy(val_sub), val_gt)

    val_calibrated = calibrator.apply_to_predictions(val_sub)
    calibrated_val = run_eval("calibrated", val_calibrated, val_gt)

    # === COMPARISON TABLE ===
    all_test = [baseline_test, calibrated_test, temp_only_test, thd_only_test, hard_test]
    all_val = [baseline_val, calibrated_val]

    key_metrics = ["AUROC", "Rej-F1@0.4", "Rej-F1@0.6", "G-mIoU@1", "mAP", "mR@5", "mR+@5", "mIoU@1"]

    print("\n" + "=" * 80)
    print("TEST SET COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Method':<30}" + "".join(f"{k:>12}" for k in key_metrics)
    print(header)
    print("-" * len(header))
    for entry in all_test:
        row = f"{entry['name']:<30}"
        for k in key_metrics:
            row += f"{entry['metrics'].get(k, 0):>12.2f}"
        print(row)

    print("\n" + "=" * 80)
    print("GAINS vs BASELINE (Test)")
    print("=" * 80)
    base = baseline_test["metrics"]
    header = f"{'Method':<30}" + "".join(f"{k:>12}" for k in key_metrics)
    print(header)
    print("-" * len(header))
    for entry in all_test[1:]:
        row = f"{entry['name']:<30}"
        for k in key_metrics:
            gain = entry["metrics"].get(k, 0) - base.get(k, 0)
            row += f"{gain:>+12.2f}"
        print(row)

    # Save results
    output = {
        "test_results": all_test,
        "val_results": all_val,
        "calibrator_params": {
            "temperature": calibrator.temperature,
            "threshold": calibrator.threshold,
        },
    }
    output_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"\n[phase4] Saved -> {output_path}")

    # Save calibrated test predictions
    calib_path = os.path.join(args.output_dir, "test_calibrated.jsonl")
    with open(calib_path, "w", encoding="utf-8") as f:
        for row in test_calibrated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[phase4] Calibrated test predictions -> {calib_path}")

    # Cross-reference with error analysis
    print("\n" + "=" * 80)
    print("CROSS-REFERENCE: Error reduction on target category")
    print("=" * 80)
    from error_analysis import build_detail_table
    from metrics import DEFAULT_IOU_THRESHOLDS

    baseline_detail = build_detail_table(test_sub, test_gt, iou_thresholds=DEFAULT_IOU_THRESHOLDS)
    calib_detail = build_detail_table(test_calibrated, test_gt, iou_thresholds=DEFAULT_IOU_THRESHOLDS)

    baseline_errors = {}
    calib_errors = {}
    for row in baseline_detail:
        e = row["error_type_iou50"]
        baseline_errors[e] = baseline_errors.get(e, 0) + 1
    for row in calib_detail:
        e = row["error_type_iou50"]
        calib_errors[e] = calib_errors.get(e, 0) + 1

    all_error_types = sorted(set(list(baseline_errors.keys()) + list(calib_errors.keys())))
    print(f"{'Error Type':<30} {'Baseline':>10} {'Calibrated':>10} {'Change':>10}")
    print("-" * 60)
    for e in all_error_types:
        b = baseline_errors.get(e, 0)
        c = calib_errors.get(e, 0)
        print(f"{e:<30} {b:>10} {c:>10} {c-b:>+10}")

    cross_ref = {
        "baseline_error_counts": baseline_errors,
        "calibrated_error_counts": calib_errors,
    }
    cross_ref_path = os.path.join(args.output_dir, "cross_reference.json")
    with open(cross_ref_path, "w", encoding="utf-8") as f:
        json.dump(cross_ref, f, indent=4, ensure_ascii=False)
    print(f"[phase4] Cross-reference -> {cross_ref_path}")


if __name__ == "__main__":
    main()
