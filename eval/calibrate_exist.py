# -*- coding: utf-8 -*-
"""
Existence Score Calibration Module for GMR.

Post-processing module that calibrates the existence head's output to improve
rejection accuracy. Two calibration strategies:

1. Temperature scaling (Platt scaling): learn a single temperature parameter
   on the validation set to sharpen or soften the sigmoid output.
2. Adaptive threshold: learn the optimal threshold on the validation set
   using the Rej-F1 objective.

This is a post-processing module, no model retraining needed.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np

from utils import load_jsonl
from metrics import get_existence_score


def calibrate_temperature(
    null_scores: np.ndarray,
    pos_scores: np.ndarray,
    method: str = "max_rej_f1",
) -> Tuple[float, float]:
    """
    Find optimal temperature and threshold on validation data.

    We model calibrated_score = sigmoid((logit(score) + bias) / temperature).
    Since we only have sigmoid outputs, we convert to logit space first.

    Args:
        null_scores: existence scores for null-set queries
        pos_scores: existence scores for positive queries
        method: "max_rej_f1" or "max_auroc"

    Returns:
        (optimal_threshold, optimal_temperature)
    """
    eps = 1e-7

    def to_logit(s):
        s = np.clip(s, eps, 1.0 - eps)
        return np.log(s / (1.0 - s))

    def to_sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    best_score = -1.0
    best_temp = 1.0
    best_thd = 0.4

    all_scores = np.concatenate([null_scores, pos_scores])
    all_labels = np.concatenate([np.zeros(len(null_scores)), np.ones(len(pos_scores))])
    all_logits = to_logit(all_scores)

    for temp in np.arange(0.5, 5.0, 0.1):
        calibrated_logits = all_logits / temp
        calibrated_scores = to_sigmoid(calibrated_logits)

        if method == "max_rej_f1":
            for thd in np.arange(0.3, 0.8, 0.01):
                pred_pos = calibrated_scores > thd
                tp = int(((all_labels == 1) & pred_pos).sum())
                fn = int(((all_labels == 1) & ~pred_pos).sum())
                tn = int(((all_labels == 0) & ~pred_pos).sum())
                fp = int(((all_labels == 0) & pred_pos).sum())
                rej_p = tn / (tn + fn) if (tn + fn) > 0 else 0
                rej_r = tn / (tn + fp) if (tn + fp) > 0 else 0
                rej_f1 = 2 * rej_p * rej_r / (rej_p + rej_r) if (rej_p + rej_r) > 0 else 0
                if rej_f1 > best_score:
                    best_score = rej_f1
                    best_temp = temp
                    best_thd = thd
        elif method == "max_gmiou":
            for thd in np.arange(0.3, 0.8, 0.01):
                pred_pos = calibrated_scores > thd
                tp = int(((all_labels == 1) & pred_pos).sum())
                fn = int(((all_labels == 1) & ~pred_pos).sum())
                tn = int(((all_labels == 0) & ~pred_pos).sum())
                fp = int(((all_labels == 0) & pred_pos).sum())
                acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                if acc > best_score:
                    best_score = acc
                    best_temp = temp
                    best_thd = thd

    return best_thd, best_temp


def apply_calibration(
    submission: List[Dict[str, Any]],
    temperature: float,
    threshold: float,
) -> List[Dict[str, Any]]:
    """
    Apply temperature scaling to existence scores and update predictions.

    If calibrated score <= threshold, zero out window scores (hard gate).
    """
    eps = 1e-7
    calibrated = deepcopy(submission)

    for pred in calibrated:
        score, _ = get_existence_score(pred)
        s = np.clip(score, eps, 1.0 - eps)
        logit = np.log(s / (1.0 - s))
        calibrated_score = 1.0 / (1.0 + np.exp(-logit / temperature))
        pred["pred_exist_score"] = round(float(calibrated_score), 6)

        windows = pred.get("pred_relevant_windows", [])
        if calibrated_score <= threshold:
            pred["pred_relevant_windows"] = [[w[0], w[1], 0.0] for w in windows]
        else:
            scale = calibrated_score
            pred["pred_relevant_windows"] = [
                [w[0], w[1], round(w[2] * scale, 6)] for w in windows
            ]

    return calibrated


def main() -> None:
    parser = argparse.ArgumentParser(description="Existence score calibration")
    parser.add_argument("--val_sub_path", type=str, required=True)
    parser.add_argument("--val_gt_path", type=str, required=True)
    parser.add_argument("--test_sub_path", type=str, required=True)
    parser.add_argument("--test_gt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/calibrated")
    parser.add_argument("--method", type=str, default="max_rej_f1",
                        choices=["max_rej_f1", "max_gmiou"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    val_sub = load_jsonl(args.val_sub_path)
    val_gt = load_jsonl(args.val_gt_path)
    test_sub = load_jsonl(args.test_sub_path)
    test_gt = load_jsonl(args.test_gt_path)

    val_gt_by_qid = {d["qid"]: d for d in val_gt}

    val_null_scores = []
    val_pos_scores = []
    for pred in val_sub:
        qid = pred["qid"]
        gt_entry = val_gt_by_qid.get(qid)
        if not gt_entry:
            continue
        is_positive = len(gt_entry.get("relevant_windows", [])) > 0
        score, _ = get_existence_score(pred)
        if is_positive:
            val_pos_scores.append(score)
        else:
            val_null_scores.append(score)

    val_null_scores = np.array(val_null_scores)
    val_pos_scores = np.array(val_pos_scores)

    print(f"[calibrate] Val: {len(val_null_scores)} null-set, {len(val_pos_scores)} positive")
    print(f"[calibrate] Val null-set mean={val_null_scores.mean():.4f}, pos mean={val_pos_scores.mean():.4f}")

    optimal_thd, optimal_temp = calibrate_temperature(
        val_null_scores, val_pos_scores, method=args.method
    )
    print(f"[calibrate] Optimal: temperature={optimal_temp:.2f}, threshold={optimal_thd:.2f}")

    calib_params = {
        "temperature": round(optimal_temp, 4),
        "threshold": round(optimal_thd, 4),
        "method": args.method,
        "val_null_mean": round(float(val_null_scores.mean()), 4),
        "val_pos_mean": round(float(val_pos_scores.mean()), 4),
    }
    params_path = os.path.join(args.output_dir, "calibration_params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(calib_params, f, indent=4)
    print(f"[calibrate] Params saved -> {params_path}")

    val_calibrated = apply_calibration(val_sub, optimal_temp, optimal_thd)
    test_calibrated = apply_calibration(test_sub, optimal_temp, optimal_thd)

    val_out_path = os.path.join(args.output_dir, "val_calibrated.jsonl")
    test_out_path = os.path.join(args.output_dir, "test_calibrated.jsonl")

    with open(val_out_path, "w", encoding="utf-8") as f:
        for row in val_calibrated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(test_out_path, "w", encoding="utf-8") as f:
        for row in test_calibrated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[calibrate] Calibrated predictions saved")
    print(f"  Val -> {val_out_path}")
    print(f"  Test -> {test_out_path}")

    from eval_main import evaluate_gmr

    print(f"\n[calibrate] Evaluating calibrated val predictions...")
    val_results = evaluate_gmr(val_calibrated, val_gt, verbose=True)

    print(f"\n[calibrate] Evaluating calibrated test predictions...")
    test_results = evaluate_gmr(test_calibrated, test_gt, verbose=True)

    val_metrics_path = os.path.join(args.output_dir, "val_calibrated_metrics.json")
    test_metrics_path = os.path.join(args.output_dir, "test_calibrated_metrics.json")

    with open(val_metrics_path, "w", encoding="utf-8") as f:
        json.dump(dict(val_results), f, indent=4, ensure_ascii=False)
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(dict(test_results), f, indent=4, ensure_ascii=False)

    print(f"\n[calibrate] Done!")


if __name__ == "__main__":
    main()
