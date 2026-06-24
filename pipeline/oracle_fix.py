#!/usr/bin/env python3
"""
Phase 3: Counterfactual (Oracle) Fix Ranking for Moment-DETR-GMR.

Sequentially fixes one error type at a time, re-runs eval,
and ranks by G-mIoU@1 / mAP / Rej-F1 gain.

Oracle fixes:
  3.2  Fix FP  — perfectly reject all null-set queries
  3.3  Fix Multi-miss — add missed subsequent GT moments to predictions
  3.4  Fix Boundary — set matched pair predictions to exact GT boundaries
  3.5  Fix Over-detection — remove unmatched (extra) predictions

Usage:
  cd /home/guoxiangyu/GMR/generalized-moment-retrieval
  python pipeline/oracle_fix.py
"""

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
OUT_DIR = ROOT / "results/moment_detr_gmr/oracle_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the optimal threshold for G-mIoU evaluation
GMIOU_THRESHOLD = 0.55


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: Path) -> None:
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def save_json(obj: Any, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ─── IoU helpers ──────────────────────────────────────────────────────────────

def temporal_iou(pred: List[float], gt: List[float]) -> float:
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
    iou_thd: float = 0.5,
) -> List[Tuple[int, int, float]]:
    if not preds or not gts:
        return []
    iou_matrix = np.array(
        [[temporal_iou(p, g) for g in gts] for p in preds], dtype=np.float64
    )
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


# ─── Run eval ─────────────────────────────────────────────────────────────────

def run_eval(pred_path: Path, save_path: Path, gmiou_threshold: float = GMIOU_THRESHOLD) -> Dict:
    """Run official eval script and return the brief metrics dict."""
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--submission_path", str(pred_path),
        "--gt_path", str(GT_PATH),
        "--save_path", str(save_path),
        "--gmiou_cls_threshold", str(gmiou_threshold),
        "--cls_thresholds", "0.4", "0.55", "0.6",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  [ERROR] eval failed:\n{result.stderr[:500]}")
        return {}

    with open(save_path) as f:
        data = json.load(f)
    return data.get("brief", data)


def extract_key_metrics(brief: Dict) -> Dict:
    """Extract the three key metrics we compare."""
    return {
        "G-mIoU@1": brief.get("G-mIoU@1", 0.0),
        "mAP": brief.get("mAP", 0.0),
        "Rej-F1": brief.get(f"Rej-F1@{GMIOU_THRESHOLD}", brief.get("Rej-F1@0.6", 0.0)),
    }


# ─── Load data ────────────────────────────────────────────────────────────────

def load_data():
    gt_list = load_jsonl(GT_PATH)
    pred_list = load_jsonl(PRED_PATH)
    pred_map = {d["qid"]: d for d in pred_list}
    gt_map = {d["qid"]: d for d in gt_list}
    return gt_list, pred_list, pred_map, gt_map


# ─── Baseline eval ────────────────────────────────────────────────────────────

def run_baseline(pred_list):
    print("\n=== Baseline Eval ===")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        for d in pred_list:
            f.write(json.dumps(d) + "\n")
        tmp_pred = Path(f.name)
    save_path = OUT_DIR / "baseline_eval.json"
    brief = run_eval(tmp_pred, save_path)
    tmp_pred.unlink()
    metrics = extract_key_metrics(brief)
    print(f"  Baseline: {metrics}")
    return metrics


# ─── Task 3.2: Oracle Fix FP ──────────────────────────────────────────────────

def oracle_fix_fp(pred_list, gt_map):
    """Perfectly reject all true null-set queries: set exist_score=0, clear windows."""
    print("\n=== Oracle Fix FP (perfect null-set rejection) ===")
    fixed = []
    n_fixed = 0
    for d in pred_list:
        d2 = copy.deepcopy(d)
        gt = gt_map.get(d["qid"], {})
        is_negative = len(gt.get("relevant_windows", [])) == 0
        if is_negative:
            d2["pred_exist_score"] = 0.0
            d2["pred_relevant_windows"] = []
            n_fixed += 1
        fixed.append(d2)
    print(f"  Fixed {n_fixed} null-set queries (exist_score→0, windows cleared)")

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        for d in fixed:
            f.write(json.dumps(d) + "\n")
        tmp_pred = Path(f.name)
    save_path = OUT_DIR / "fix_fp_eval.json"
    brief = run_eval(tmp_pred, save_path)
    tmp_pred.unlink()
    metrics = extract_key_metrics(brief)
    print(f"  Fix-FP: {metrics}")
    return metrics


# ─── Task 3.3: Oracle Fix Multi-miss ─────────────────────────────────────────

def oracle_fix_multi_miss(pred_list, gt_map, iou_thd=0.5):
    """Add missed subsequent GT moments as oracle predictions."""
    print(f"\n=== Oracle Fix Multi-miss (add missed GT moments, IoU≥{iou_thd}) ===")
    fixed = []
    n_added_total = 0
    n_queries_fixed = 0

    for d in pred_list:
        d2 = copy.deepcopy(d)
        gt = gt_map.get(d["qid"], {})
        gts = gt.get("relevant_windows", [])

        if len(gts) < 2:
            fixed.append(d2)
            continue

        pred_windows_raw = d.get("pred_relevant_windows", [])
        pred_windows = [[w[0], w[1]] for w in pred_windows_raw]

        # Find unmatched GT windows
        matches = greedy_match(pred_windows, gts, iou_thd=iou_thd)
        matched_gt_idx = {m[1] for m in matches}
        unmatched_gts = [gts[j] for j in range(len(gts)) if j not in matched_gt_idx]

        if unmatched_gts:
            # Add unmatched GT windows as oracle predictions with score=1.0
            extra = [[g[0], g[1], 1.0] for g in unmatched_gts]
            d2["pred_relevant_windows"] = list(pred_windows_raw) + extra
            n_added_total += len(unmatched_gts)
            n_queries_fixed += 1

        fixed.append(d2)

    print(f"  Added {n_added_total} oracle GT moments to {n_queries_fixed} queries")

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        for d in fixed:
            f.write(json.dumps(d) + "\n")
        tmp_pred = Path(f.name)
    save_path = OUT_DIR / "fix_multi_miss_eval.json"
    brief = run_eval(tmp_pred, save_path)
    tmp_pred.unlink()
    metrics = extract_key_metrics(brief)
    print(f"  Fix-Multi-miss: {metrics}")
    return metrics


# ─── Task 3.4: Oracle Fix Boundary ───────────────────────────────────────────

def oracle_fix_boundary(pred_list, gt_map, iou_thd=0.1):
    """Replace matched prediction boundaries with exact GT boundaries (IoU→1.0)."""
    print(f"\n=== Oracle Fix Boundary (matched pairs IoU→1.0) ===")
    fixed = []
    n_replaced = 0

    for d in pred_list:
        d2 = copy.deepcopy(d)
        gt = gt_map.get(d["qid"], {})
        gts = gt.get("relevant_windows", [])

        if not gts:
            fixed.append(d2)
            continue

        pred_windows_raw = list(d.get("pred_relevant_windows", []))
        pred_windows = [[w[0], w[1]] for w in pred_windows_raw]

        # Force-match (low threshold) to find any matched pair
        matches = greedy_match(pred_windows, gts, iou_thd=iou_thd)

        new_windows = [list(w) for w in pred_windows_raw]
        for (i_pred, j_gt, iou) in matches:
            # Replace pred boundary with GT boundary (keep original score)
            g = gts[j_gt]
            old_score = new_windows[i_pred][2] if len(new_windows[i_pred]) > 2 else 1.0
            new_windows[i_pred] = [g[0], g[1], old_score]
            n_replaced += 1

        d2["pred_relevant_windows"] = new_windows
        fixed.append(d2)

    print(f"  Replaced boundaries for {n_replaced} matched pairs")

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        for d in fixed:
            f.write(json.dumps(d) + "\n")
        tmp_pred = Path(f.name)
    save_path = OUT_DIR / "fix_boundary_eval.json"
    brief = run_eval(tmp_pred, save_path)
    tmp_pred.unlink()
    metrics = extract_key_metrics(brief)
    print(f"  Fix-Boundary: {metrics}")
    return metrics


# ─── Task 3.5: Oracle Fix Over-detection ─────────────────────────────────────

def oracle_fix_over_detection(pred_list, gt_map, iou_thd=0.5):
    """Remove all unmatched (extra) predictions; null-set queries: clear all."""
    print(f"\n=== Oracle Fix Over-detection (remove unmatched preds) ===")
    fixed = []
    n_removed = 0

    for d in pred_list:
        d2 = copy.deepcopy(d)
        gt = gt_map.get(d["qid"], {})
        gts = gt.get("relevant_windows", [])

        pred_windows_raw = d.get("pred_relevant_windows", [])

        if not gts:
            # Null-set: remove all predictions, set exist_score to preserve actual score
            d2["pred_relevant_windows"] = []
            # Keep original exist_score (not oracle-fixing the classification)
            n_removed += len(pred_windows_raw)
        else:
            pred_windows = [[w[0], w[1]] for w in pred_windows_raw]
            matches = greedy_match(pred_windows, gts, iou_thd=iou_thd)
            matched_pred_idx = {m[0] for m in matches}

            new_windows = [pred_windows_raw[i] for i in range(len(pred_windows_raw))
                           if i in matched_pred_idx]
            n_removed += len(pred_windows_raw) - len(new_windows)
            d2["pred_relevant_windows"] = new_windows

        fixed.append(d2)

    print(f"  Removed {n_removed} extra predictions")

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        for d in fixed:
            f.write(json.dumps(d) + "\n")
        tmp_pred = Path(f.name)
    save_path = OUT_DIR / "fix_over_detection_eval.json"
    brief = run_eval(tmp_pred, save_path)
    tmp_pred.unlink()
    metrics = extract_key_metrics(brief)
    print(f"  Fix-Over-detection: {metrics}")
    return metrics


# ─── Task 3.6: Oracle Fix All (upper bound) ───────────────────────────────────

def oracle_fix_all(pred_list, gt_map):
    """Apply all oracle fixes simultaneously for upper bound."""
    print(f"\n=== Oracle Fix ALL (upper bound) ===")
    fixed = []

    for d in pred_list:
        gt = gt_map.get(d["qid"], {})
        gts = gt.get("relevant_windows", [])

        if not gts:
            # Perfect rejection
            fixed.append({
                "qid": d["qid"],
                "pred_exist_score": 0.0,
                "pred_relevant_windows": [],
            })
        else:
            # Perfect prediction: use GT windows directly as predictions
            fixed.append({
                "qid": d["qid"],
                "pred_exist_score": 1.0,
                "pred_relevant_windows": [[g[0], g[1], 1.0] for g in gts],
            })

    print(f"  Oracle upper bound: perfect classification + perfect localization")

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        for d in fixed:
            f.write(json.dumps(d) + "\n")
        tmp_pred = Path(f.name)
    save_path = OUT_DIR / "fix_all_eval.json"
    brief = run_eval(tmp_pred, save_path)
    tmp_pred.unlink()
    metrics = extract_key_metrics(brief)
    print(f"  Oracle upper bound: {metrics}")
    return metrics


# ─── Task 3.7: Summary & ranking ─────────────────────────────────────────────

def write_oracle_summary(baseline, fixes: Dict[str, Dict]) -> None:
    print(f"\n=== Task 3.7: Writing Oracle summary ===")

    direction_map = {
        "Fix FP": "candidate verification (exist_head calibration / null-set detection)",
        "Fix Multi-miss": "coverage-aware retrieval (multi-moment detection)",
        "Fix Boundary": "boundary refinement (temporal boundary regression)",
        "Fix Over-detection": "NMS / score thresholding post-processing",
    }

    # Compute gains
    rows = []
    for name, metrics in fixes.items():
        if not metrics:
            continue
        g_miou_gain = metrics.get("G-mIoU@1", 0) - baseline.get("G-mIoU@1", 0)
        map_gain = metrics.get("mAP", 0) - baseline.get("mAP", 0)
        rej_gain = metrics.get("Rej-F1", 0) - baseline.get("Rej-F1", 0)
        rows.append({
            "name": name,
            "G-mIoU@1": metrics.get("G-mIoU@1", 0),
            "mAP": metrics.get("mAP", 0),
            "Rej-F1": metrics.get("Rej-F1", 0),
            "G-mIoU@1_gain": g_miou_gain,
            "mAP_gain": map_gain,
            "Rej-F1_gain": rej_gain,
        })

    rows.sort(key=lambda x: x["G-mIoU@1_gain"], reverse=True)

    # Print table
    print("\n" + "="*70)
    print(f"{'Fix Type':<22} {'G-mIoU@1':>10} {'Gain':>8} {'mAP':>8} {'Gain':>8} {'Rej-F1':>8} {'Gain':>8}")
    print("-"*70)
    print(f"{'Baseline':<22} {baseline.get('G-mIoU@1',0):>10.2f} {'':>8} {baseline.get('mAP',0):>8.2f} {'':>8} {baseline.get('Rej-F1',0):>8.2f} {'':>8}")
    for r in rows:
        print(f"{r['name']:<22} {r['G-mIoU@1']:>10.2f} {r['G-mIoU@1_gain']:>+8.2f} "
              f"{r['mAP']:>8.2f} {r['mAP_gain']:>+8.2f} {r['Rej-F1']:>8.2f} {r['Rej-F1_gain']:>+8.2f}")
    print("="*70)

    # Save ranking JSON
    ranking = {
        "gmiou_threshold_used": GMIOU_THRESHOLD,
        "baseline": baseline,
        "ranked_by_gmiou_gain": rows,
        "recommended_direction": rows[0]["name"] if rows else "N/A",
        "direction_mapping": direction_map,
    }
    save_json(ranking, OUT_DIR / "oracle_ranking.json")

    # Write summary.md
    if rows:
        best = rows[0]
    else:
        best = {}

    lines = [
        "# Oracle Fix Ranking — Moment-DETR-GMR",
        "",
        f"> Evaluation threshold for G-mIoU: τ = {GMIOU_THRESHOLD}",
        "",
        "## Gain Table (ranked by G-mIoU@1 gain)",
        "",
        f"| Fix Type | G-mIoU@1 | Gain | mAP | Gain | Rej-F1 | Gain | Recommended Direction |",
        f"|---------|:--------:|:----:|:---:|:----:|:------:|:----:|----------------------|",
        f"| **Baseline** | {baseline.get('G-mIoU@1',0):.2f} | — | "
        f"{baseline.get('mAP',0):.2f} | — | {baseline.get('Rej-F1',0):.2f} | — | — |",
    ]
    for r in rows:
        name = r["name"]
        dir_ = direction_map.get(name, "—")
        marker = " 🏆" if name == (best.get("name", "") if best else "") else ""
        lines.append(
            f"| **{name}**{marker} | {r['G-mIoU@1']:.2f} | {r['G-mIoU@1_gain']:+.2f} | "
            f"{r['mAP']:.2f} | {r['mAP_gain']:+.2f} | "
            f"{r['Rej-F1']:.2f} | {r['Rej-F1_gain']:+.2f} | {dir_} |"
        )

    oracle_row = fixes.get("Fix All", {})
    if oracle_row:
        lines += [
            f"| **Fix All** (upper bound) | {oracle_row.get('G-mIoU@1',0):.2f} | "
            f"{oracle_row.get('G-mIoU@1',0)-baseline.get('G-mIoU@1',0):+.2f} | "
            f"{oracle_row.get('mAP',0):.2f} | {oracle_row.get('mAP',0)-baseline.get('mAP',0):+.2f} | "
            f"{oracle_row.get('Rej-F1',0):.2f} | {oracle_row.get('Rej-F1',0)-baseline.get('Rej-F1',0):+.2f} | — |",
        ]

    best_name = best.get("name", "N/A") if best else "N/A"
    best_dir = direction_map.get(best_name, "N/A")

    lines += [
        "",
        "---",
        "",
        "## 结论与方向选择",
        "",
        f"**指标增益最大的修复**: `{best_name}`  ",
        f"**对应改进方向**: {best_dir}",
        "",
        "### 映射关系",
        "| 主要瓶颈 | 改进方向 |",
        "|---------|---------|",
        "| 多时刻漏检 | Coverage-aware retrieval (改进 decoder 多样性) |",
        "| 误报为主 | Candidate verification (null-set 判别器) |",
        "| 边界偏移 | Boundary refinement (后处理时序边界回归) |",
        "| 多检 | NMS / 分数阈值 post-processing |",
        "",
        "---",
        "",
        "## 下一步 (Phase 4)",
        f"- 基于 `{best_name}` 的方向: **{best_dir}**",
        "- 设计具体模块, 见 task_plan.md Phase 4",
    ]

    with open(OUT_DIR / "summary.md", "w") as f:
        f.write("\n".join(lines))
    print(f"  [saved] oracle_analysis/summary.md")
    print(f"  [saved] oracle_analysis/oracle_ranking.json")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 3: Counterfactual Oracle Fix Ranking")
    print("=" * 60)

    gt_list, pred_list, pred_map, gt_map = load_data()

    # Baseline
    baseline = run_baseline(pred_list)

    # Oracle fixes
    fixes = {}
    fixes["Fix FP"] = oracle_fix_fp(pred_list, gt_map)
    fixes["Fix Multi-miss"] = oracle_fix_multi_miss(pred_list, gt_map)
    fixes["Fix Boundary"] = oracle_fix_boundary(pred_list, gt_map)
    fixes["Fix Over-detection"] = oracle_fix_over_detection(pred_list, gt_map)
    fixes["Fix All"] = oracle_fix_all(pred_list, gt_map)

    write_oracle_summary(baseline, fixes)

    print("\n" + "=" * 60)
    print("Phase 3 COMPLETE. Results in results/moment_detr_gmr/oracle_analysis/")
    print("=" * 60)


if __name__ == "__main__":
    main()
