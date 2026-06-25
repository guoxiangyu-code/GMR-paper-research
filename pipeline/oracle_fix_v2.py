#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import tempfile
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
GT_PATH = ROOT / "data/label/Standard/test.jsonl"
PRED_PATH = ROOT / "results/moment_detr_gmr/test/moment_detr_gmr_test_submission.jsonl"
EVAL_SCRIPT = ROOT / "eval/eval_main.py"
OUT_DIR = ROOT / "results/moment_detr_gmr/oracle_analysis_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GMIOU_THRESHOLD = 0.55

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def temporal_iou(pred: List[float], gt: List[float]) -> float:
    inter_st = max(pred[0], gt[0])
    inter_ed = min(pred[1], gt[1])
    inter = max(0.0, inter_ed - inter_st)
    if inter == 0:
        return 0.0
    union = (pred[1] - pred[0]) + (gt[1] - gt[0]) - inter
    return inter / union if union > 0 else 0.0

def greedy_match(preds: List[List[float]], gts: List[List[float]], iou_thd: float = 0.5) -> List[Tuple[int, int, float]]:
    if not preds or not gts:
        return []
    iou_matrix = np.array([[temporal_iou(p, g) for g in gts] for p in preds], dtype=np.float64)
    matched_gt: set = set()
    matches = []
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

def run_eval(pred_path: Path, save_path: Path) -> Dict:
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--submission_path", str(pred_path),
        "--gt_path", str(GT_PATH),
        "--save_path", str(save_path),
        "--gmiou_cls_threshold", str(GMIOU_THRESHOLD),
        "--cls_thresholds", "0.4", "0.55", "0.6",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  [ERROR] eval failed:\n{result.stderr[:500]}")
        return {}
    with open(save_path) as f:
        data = json.load(f)
    return data.get("brief", data)

def extract_metrics(brief: Dict) -> Dict:
    return {
        "G-mIoU@1": brief.get("G-mIoU@1", 0.0),
        "G-mIoU@3": brief.get("G-mIoU@3", 0.0),
        "mAP": brief.get("mAP", 0.0),
        "mR+@5": brief.get("mR+@5", 0.0),
        "Rej-F1": brief.get(f"Rej-F1@{GMIOU_THRESHOLD}", brief.get("Rej-F1@0.6", 0.0)),
    }

def process_eval(pred_list, label):
    print(f"\n=== {label} ===")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        for d in pred_list:
            f.write(json.dumps(d) + "\n")
        tmp_pred = Path(f.name)
    save_path = OUT_DIR / f"{label.replace(' ', '_').lower()}.json"
    brief = run_eval(tmp_pred, save_path)
    tmp_pred.unlink()
    metrics = extract_metrics(brief)
    print(f"  {label}: {metrics}")
    return metrics

def run_oracle_v2():
    gt_list = load_jsonl(GT_PATH)
    pred_list = load_jsonl(PRED_PATH)
    gt_map = {d["qid"]: d for d in gt_list}
    
    baseline = process_eval(pred_list, "Baseline")
    
    # 1. Fix Over-detection (Remove unmatched)
    fixed_od = []
    for d in pred_list:
        d2 = copy.deepcopy(d)
        gt = gt_map.get(d["qid"], {})
        gts = gt.get("relevant_windows", [])
        pred_windows_raw = d.get("pred_relevant_windows", [])
        if not gts:
            d2["pred_relevant_windows"] = []
        else:
            pred_windows = [[w[0], w[1]] for w in pred_windows_raw]
            matches = greedy_match(pred_windows, gts, iou_thd=0.5)
            matched_pred_idx = {m[0] for m in matches}
            d2["pred_relevant_windows"] = [pred_windows_raw[i] for i in range(len(pred_windows_raw)) if i in matched_pred_idx]
        fixed_od.append(d2)
    res_od = process_eval(fixed_od, "Fix Over-detection")
    
    # 2. Fix FN (Force accept falsely rejected positives)
    fixed_fn = []
    for d in pred_list:
        d2 = copy.deepcopy(d)
        gt = gt_map.get(d["qid"], {})
        is_positive = len(gt.get("relevant_windows", [])) > 0
        if is_positive:
            # Force score > threshold to prevent false rejection
            if d.get("pred_exist_score", 0.0) <= GMIOU_THRESHOLD:
                d2["pred_exist_score"] = 1.0
        fixed_fn.append(d2)
    res_fn = process_eval(fixed_fn, "Fix FN")
    
    # 3. Fix Multi-miss
    fixed_mm = []
    for d in pred_list:
        d2 = copy.deepcopy(d)
        gt = gt_map.get(d["qid"], {})
        gts = gt.get("relevant_windows", [])
        if len(gts) < 2:
            fixed_mm.append(d2)
            continue
        pred_windows_raw = d.get("pred_relevant_windows", [])
        pred_windows = [[w[0], w[1]] for w in pred_windows_raw]
        matches = greedy_match(pred_windows, gts, iou_thd=0.5)
        matched_gt_idx = {m[1] for m in matches}
        unmatched_gts = [gts[j] for j in range(len(gts)) if j not in matched_gt_idx]
        if unmatched_gts:
            extra = [[g[0], g[1], 1.0] for g in unmatched_gts]
            d2["pred_relevant_windows"] = list(pred_windows_raw) + extra
        fixed_mm.append(d2)
    res_mm = process_eval(fixed_mm, "Fix Multi-miss")
    
    print("\n[Summary] G-mIoU@1 / G-mIoU@3 / Rej-F1 / mR+@5 Gain Table:")
    for label, res in [("Fix Over-detection", res_od), ("Fix FN", res_fn), ("Fix Multi-miss", res_mm)]:
        g1 = res["G-mIoU@1"] - baseline["G-mIoU@1"]
        g3 = res["G-mIoU@3"] - baseline["G-mIoU@3"]
        gR = res["Rej-F1"] - baseline["Rej-F1"]
        gM = res["mR+@5"] - baseline["mR+@5"]
        print(f"  {label:<20}: @1 +{g1:.2f} | @3 +{g3:.2f} | Rej-F1 +{gR:.2f} | mR+@5 +{gM:.2f}")

run_oracle_v2()
