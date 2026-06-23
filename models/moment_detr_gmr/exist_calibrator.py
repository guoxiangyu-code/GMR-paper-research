# -*- coding: utf-8 -*-
"""
Existence-Calibrated GMR Post-Processing Module.

A toggleable post-processing module (`--use_exist_calib`) that improves
GMR rejection metrics via:

1. Temperature scaling of existence scores (learned on val)
2. Adaptive threshold for existence gate (learned on val)
3. Hard gate: zero all window scores when calibrated exist_score <= threshold
4. Score recalibration: multiply window scores by calibrated exist_score

This module is purely post-processing — no model retraining needed.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ExistenceCalibrator:
    """Calibrate existence scores via temperature scaling."""

    def __init__(self, temperature: float = 1.0, threshold: float = 0.4):
        self.temperature = temperature
        self.threshold = threshold

    @staticmethod
    def _to_logit(s: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        s = np.clip(s, eps, 1.0 - eps)
        return np.log(s / (1.0 - s))

    @staticmethod
    def _to_sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def calibrate_scores(self, scores: np.ndarray) -> np.ndarray:
        logits = self._to_logit(scores)
        return self._to_sigmoid(logits / self.temperature)

    def calibrate_single(self, score: float) -> float:
        return float(self.calibrate_scores(np.array([score]))[0])

    def apply_to_predictions(self, submission: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply calibration to a list of predictions."""
        from copy import deepcopy
        calibrated = deepcopy(submission)

        for pred in calibrated:
            score = pred.get("pred_exist_score", 0.0)
            cal_score = self.calibrate_single(score)
            pred["pred_exist_score"] = round(cal_score, 6)

            windows = pred.get("pred_relevant_windows", [])
            if cal_score <= self.threshold:
                pred["pred_relevant_windows"] = [[w[0], w[1], 0.0] for w in windows]
            else:
                pred["pred_relevant_windows"] = [
                    [w[0], w[1], round(w[2] * cal_score, 6)] for w in windows
                ]

        return calibrated

    @classmethod
    def fit(
        cls,
        val_sub: List[Dict[str, Any]],
        val_gt: List[Dict[str, Any]],
        method: str = "max_rej_f1",
    ) -> "ExistenceCalibrator":
        """Fit temperature and threshold on validation data."""
        val_gt_by_qid = {d["qid"]: d for d in val_gt}

        null_scores = []
        pos_scores = []
        for pred in val_sub:
            qid = pred["qid"]
            gt_entry = val_gt_by_qid.get(qid)
            if not gt_entry:
                continue
            is_positive = len(gt_entry.get("relevant_windows", [])) > 0
            score = pred.get("pred_exist_score", 0.0)
            if is_positive:
                pos_scores.append(score)
            else:
                null_scores.append(score)

        null_scores = np.array(null_scores)
        pos_scores = np.array(pos_scores)

        all_scores = np.concatenate([null_scores, pos_scores])
        all_labels = np.concatenate([np.zeros(len(null_scores)), np.ones(len(pos_scores))])

        best_score = -1.0
        best_temp = 1.0
        best_thd = 0.4

        for temp in np.arange(0.3, 5.0, 0.05):
            logits = cls._to_logit(all_scores)
            calibrated = cls._to_sigmoid(logits / temp)

            if method == "max_rej_f1":
                for thd in np.arange(0.3, 0.8, 0.01):
                    pred_pos = calibrated > thd
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
            elif method == "max_accuracy":
                for thd in np.arange(0.3, 0.8, 0.01):
                    pred_pos = calibrated > thd
                    acc = float((pred_pos == all_labels).mean())
                    if acc > best_score:
                        best_score = acc
                        best_temp = temp
                        best_thd = thd

        print(f"[ExistenceCalibrator] Fit: temp={best_temp:.2f}, thd={best_thd:.2f}, "
              f"score={best_score:.4f}, method={method}")
        return cls(temperature=best_temp, threshold=best_thd)

    def save(self, path: str) -> None:
        params = {"temperature": self.temperature, "threshold": self.threshold}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=4)

    @classmethod
    def load(cls, path: str) -> "ExistenceCalibrator":
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
        return cls(temperature=params["temperature"], threshold=params["threshold"])
