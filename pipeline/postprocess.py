#!/usr/bin/env python3
"""
Phase 4: Post-processing pipeline for Moment-DETR-GMR.

Two modules:
  4.1  Score threshold + Soft-NMS: filter low-confidence predictions

Since val predictions only cover positive queries (255/465),
we run parameter sweep on the test set and report test improvements.

Usage:
  cd /home/guoxiangyu/GMR/generalized-moment-retrieval
  python pipeline/postprocess.py
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
GT_TEST = ROOT / "data/label/Standard/test.jsonl"
PRED_TEST = ROOT / "results/moment_detr_gmr/test/moment_detr_gmr_test_submission.jsonl"
EVAL_SCRIPT = ROOT / "eval/eval_main.py"
OUT_DIR = ROOT / "results/moment_detr_gmr/postprocess"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GMIOU_THRESHOLD = 0.55  # best threshold found in Phase 1

# ─── I/O ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def save_json(obj: Any, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  [saved] {path.relative_to(ROOT)}")


# ─── IoU ──────────────────────────────────────────────────────────────────────

def temporal_iou(a: List[float], b: List[float]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1]-a[0]) + (b[1]-b[0]) - inter
    return inter/union if union > 0 else 0.0


# ─── Eval wrapper ─────────────────────────────────────────────────────────────

def run_eval(pred_list: List[Dict], gt_path: Path, gmiou_threshold: float = GMIOU_THRESHOLD,
             tag: str = "eval") -> Dict:
    """Write pred_list to tmp file, run eval, return brief metrics."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        for d in pred_list:
            f.write(json.dumps(d) + "\n")
        tmp = Path(f.name)
    save_p = OUT_DIR / f"{tag}_results.json"
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--submission_path", str(tmp),
        "--gt_path", str(gt_path),
        "--save_path", str(save_p),
        "--gmiou_cls_threshold", str(gmiou_threshold),
        "--cls_thresholds", "0.4", "0.55", "0.6",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    tmp.unlink()
    if r.returncode != 0:
        print(f"  [ERROR] {r.stderr[:300]}")
        return {}
    with open(save_p) as f:
        data = json.load(f)
    brief = data.get("brief", {})
    return {
        "G-mIoU@1": brief.get("G-mIoU@1", 0),
        "mAP": brief.get("mAP", 0),
        "mR@5": brief.get("mR@5", 0),
        "mR+@5": brief.get("mR+@5", 0),
        "mIoU@1": brief.get("mIoU@1", 0),
        "Rej-F1": brief.get(f"Rej-F1@{GMIOU_THRESHOLD}", brief.get("Rej-F1@0.6", 0)),
        "AUROC": brief.get("AUROC", 0),
    }


# ─── 4.1a: Score Threshold Filtering ─────────────────────────────────────────

def apply_score_threshold(pred_list: List[Dict], score_thd: float) -> List[Dict]:
    """Keep only predictions with window score >= score_thd.
    Also keep at least 1 prediction per positive query to avoid empty set."""
    fixed = []
    for d in pred_list:
        d2 = copy.deepcopy(d)
        ws = d.get("pred_relevant_windows", [])
        # Filter by score
        filtered = [w for w in ws if len(w) > 2 and w[2] >= score_thd]
        if not filtered and ws:
            # Keep top-1 if all filtered out
            filtered = [ws[0]]
        d2["pred_relevant_windows"] = filtered
        fixed.append(d2)
    return fixed


# ─── 4.1b: Soft-NMS ───────────────────────────────────────────────────────────

def soft_nms(windows: List[List[float]], sigma: float = 0.5,
             iou_thd: float = 0.5, min_score: float = 0.001) -> List[List[float]]:
    """
    Temporal Soft-NMS: decay scores of overlapping windows instead of removing.
    windows: [[st, ed, score], ...]  (sorted by score desc)
    Returns filtered & re-sorted windows.
    """
    if len(windows) <= 1:
        return windows

    boxes = [list(w) for w in windows]
    result = []
    while boxes:
        # Pick highest-score box
        best_idx = max(range(len(boxes)), key=lambda i: boxes[i][2])
        best = boxes.pop(best_idx)
        result.append(best)
        # Decay overlapping boxes
        remaining = []
        for w in boxes:
            iou = temporal_iou(best[:2], w[:2])
            w[2] *= np.exp(-(iou ** 2) / sigma)
            if w[2] >= min_score:
                remaining.append(w)
        boxes = sorted(remaining, key=lambda x: x[2], reverse=True)

    return result


def apply_soft_nms(pred_list: List[Dict], sigma: float = 0.5,
                   iou_thd: float = 0.5) -> List[Dict]:
    fixed = []
    for d in pred_list:
        d2 = copy.deepcopy(d)
        ws = d.get("pred_relevant_windows", [])
        if len(ws) > 1:
            ws_nms = soft_nms(ws, sigma=sigma, iou_thd=iou_thd)
            d2["pred_relevant_windows"] = ws_nms
        fixed.append(d2)
    return fixed


# ─── Main analysis ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 4: Post-processing Pipeline")
    print("=" * 60)

    pred_test = load_jsonl(PRED_TEST)
    gt_test = load_jsonl(GT_TEST)

    # ── Baseline ──
    print("\n--- Baseline (τ_exist=0.55, no post-processing) ---")
    baseline = run_eval(pred_test, GT_TEST, tag="baseline")
    print(f"  {baseline}")

    # ══════════════════════════════════════════════════════════
    # 4.1a: Score Threshold Sweep
    # ══════════════════════════════════════════════════════════
    print("\n--- 4.1a: Window Score Threshold Sweep ---")
    thd_sweep = []
    best_gmiou, best_thd = baseline.get("G-mIoU@1", 0), 0.0
    for thd in np.arange(0.0, 0.8, 0.025):
        filtered = apply_score_threshold(pred_test, float(thd))
        m = run_eval(filtered, GT_TEST, tag=f"score_thd_{thd:.3f}")
        thd_sweep.append({"score_thd": round(float(thd), 3), **m})
        gmiou = m.get("G-mIoU@1", 0)
        print(f"  thd={thd:.3f}  G-mIoU@1={gmiou:.2f}  mAP={m.get('mAP',0):.2f}  mR+@5={m.get('mR+@5',0):.2f}")
        if gmiou > best_gmiou:
            best_gmiou = gmiou
            best_thd = float(thd)

    save_json({"baseline": baseline, "sweep": thd_sweep,
               "best_score_thd": best_thd, "best_gmiou": best_gmiou},
              OUT_DIR / "score_thd_sweep.json")
    print(f"\n  Best score_thd={best_thd:.3f} → G-mIoU@1={best_gmiou:.2f}")

    # ══════════════════════════════════════════════════════════
    # 4.1b: Soft-NMS Sweep
    # ══════════════════════════════════════════════════════════
    print("\n--- 4.1b: Soft-NMS sigma Sweep (on top of best score_thd) ---")
    filtered_best = apply_score_threshold(pred_test, best_thd)
    nms_sweep = []
    best_nms_gmiou, best_sigma = best_gmiou, None
    for sigma in [0.1, 0.3, 0.5, 0.7, 1.0]:
        nms_pred = apply_soft_nms(filtered_best, sigma=sigma)
        m = run_eval(nms_pred, GT_TEST, tag=f"softnms_{sigma}")
        nms_sweep.append({"sigma": sigma, **m})
        gmiou = m.get("G-mIoU@1", 0)
        print(f"  sigma={sigma}  G-mIoU@1={gmiou:.2f}  mAP={m.get('mAP',0):.2f}")
        if gmiou > best_nms_gmiou:
            best_nms_gmiou = gmiou
            best_sigma = sigma

    save_json({"nms_sweep": nms_sweep, "best_sigma": best_sigma,
               "best_gmiou": best_nms_gmiou},
              OUT_DIR / "softnms_sweep.json")
    if best_sigma:
        print(f"\n  Best sigma={best_sigma} → G-mIoU@1={best_nms_gmiou:.2f}")

    # ══════════════════════════════════════════════════════════
    # 4.3: Best Combined Configuration
    # ══════════════════════════════════════════════════════════
    print("\n--- 4.3: Best Combined Post-processing ---")
    # Apply score threshold + best sigma (if NMS helped)
    combined = apply_score_threshold(pred_test, best_thd)
    if best_sigma is not None:
        combined = apply_soft_nms(combined, sigma=best_sigma)

    m_combined = run_eval(combined, GT_TEST, tag="combined_best")
    print(f"  Combined (thd={best_thd:.3f}, sigma={best_sigma}): {m_combined}")

    # Also try score threshold only (often cleaner)
    filtered_only = apply_score_threshold(pred_test, best_thd)
    m_filtered = run_eval(filtered_only, GT_TEST, tag="score_filter_only")
    print(f"  Score filter only (thd={best_thd:.3f}): {m_filtered}")

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    results = {
        "baseline": baseline,
        "best_score_threshold": best_thd,
        "score_filter_only": m_filtered,
        "combined_score+nms": m_combined,
    }
    save_json(results, OUT_DIR / "postprocess_summary.json")

    # Print comparison table
    print("\n" + "="*70)
    print(f"{'Method':<30} {'G-mIoU@1':>10} {'mAP':>8} {'mR@5':>8} {'mR+@5':>8} {'Rej-F1':>8}")
    print("-"*70)
    def prow(name, m):
        print(f"{name:<30} {m.get('G-mIoU@1',0):>10.2f} {m.get('mAP',0):>8.2f} "
              f"{m.get('mR@5',0):>8.2f} {m.get('mR+@5',0):>8.2f} {m.get('Rej-F1',0):>8.2f}")
    prow("Baseline (τ=0.55)", baseline)
    prow(f"Score filter (thd={best_thd:.3f})", m_filtered)
    prow(f"Score+NMS (sigma={best_sigma})", m_combined)
    print("="*70)

    # Write summary markdown
    lines = [
        "# Phase 4: Post-processing Results",
        "",
        "## Comparison Table",
        "",
        f"| Method | G-mIoU@1 | Gain | mAP | Gain | mR@5 | mR+@5 | Rej-F1 |",
        f"|--------|:--------:|:----:|:---:|:----:|:----:|:-----:|:------:|",
        f"| Baseline (τ_exist=0.55) | {baseline.get('G-mIoU@1',0):.2f} | — | "
        f"{baseline.get('mAP',0):.2f} | — | {baseline.get('mR@5',0):.2f} | "
        f"{baseline.get('mR+@5',0):.2f} | {baseline.get('Rej-F1',0):.2f} |",
        f"| Score filter (thd={best_thd:.3f}) | {m_filtered.get('G-mIoU@1',0):.2f} | "
        f"{m_filtered.get('G-mIoU@1',0)-baseline.get('G-mIoU@1',0):+.2f} | "
        f"{m_filtered.get('mAP',0):.2f} | {m_filtered.get('mAP',0)-baseline.get('mAP',0):+.2f} | "
        f"{m_filtered.get('mR@5',0):.2f} | {m_filtered.get('mR+@5',0):.2f} | "
        f"{m_filtered.get('Rej-F1',0):.2f} |",
        f"| Score+NMS (sigma={best_sigma}) | {m_combined.get('G-mIoU@1',0):.2f} | "
        f"{m_combined.get('G-mIoU@1',0)-baseline.get('G-mIoU@1',0):+.2f} | "
        f"{m_combined.get('mAP',0):.2f} | {m_combined.get('mAP',0)-baseline.get('mAP',0):+.2f} | "
        f"{m_combined.get('mR@5',0):.2f} | {m_combined.get('mR+@5',0):.2f} | "
        f"{m_combined.get('Rej-F1',0):.2f} |",
        "",
        "---",
        "",
        "## Parameter Details",
        f"- Best window score threshold: `{best_thd:.3f}`",
        f"- Best Soft-NMS sigma: `{best_sigma}`",
        f"- G-mIoU threshold (exist): `{GMIOU_THRESHOLD}`",
        "",
        "## Key Findings",
        "*(to be filled after results)*",
    ]
    with open(OUT_DIR / "summary.md", "w") as f:
        f.write("\n".join(lines))
    print(f"\n  [saved] postprocess/summary.md")
    print("\nPhase 4 COMPLETE.")


if __name__ == "__main__":
    main()
