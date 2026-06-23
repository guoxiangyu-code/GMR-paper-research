# Progress Log — Generalized Moment Retrieval Error Analysis

## Session 1 — 2026-06-23: Initial Repository Exploration & Planning

### Actions Completed
- [x] Read `todo.md` — understood the 5-prompt workflow (reproduce → error analysis → quantify → counterfactual → implement)
- [x] Explored full repository structure via 3 concurrent research subagents
- [x] Mapped all directories: `models/`, `training/`, `eval/`, `data/`, `features/`, `results/`, `scripts/`, `configs/`, `docs/`, `soccer_gmr_dataset/`
- [x] Verified JSONL label format (positive/negative/multi-moment samples)
- [x] Counted data splits: Standard train=4,138, val=465, test=1,036; Full train=16,898, val=2,235, test=2,986
- [x] Verified feature counts: 2,288 CLIP visual, 6,952 CLIP text, 2,288 SlowFast
- [x] Read model architecture: Moment-DETR + GMR Adapter (existence head via max-pooled decoder queries → 2-layer MLP)
- [x] Read evaluation code: `eval_main.py` + `metrics.py` with greedy matching, 8 metric types
- [x] Read training/inference scripts and config hierarchy
- [x] Identified two results directories: `moment_detr_gmr/` (local training) and `moment_detr_gmr_official/` (paper results)
- [x] Found existing test predictions (1,035 lines) and val predictions (464 lines)
- [x] Reviewed val metrics: significantly lower than paper test metrics (mAP 8.14 vs 18.69)
- [x] Created `task_plan.md`, `findings.md`, `progress.md`

### Key Discoveries
1. **Two checkpoints exist** — `results/moment_detr_gmr/best.ckpt` (local) vs `results/moment_detr_gmr_official/model_best.ckpt` (official). Need to verify which produces the reported test metrics.
2. **Val mR+@5 = 0.5** (near zero) — multi-moment retrieval is catastrophically weak on val split; this aligns with the paper's identification of multi-moment as the core challenge.
3. **Gate threshold discrepancy** — model config uses 0.3 (`exist_gate_thd`), eval default uses 0.4 (`exist_thres`/`cls_thresholds`).
4. **Full split has massive missing features** — 14,672 of 16,898 train and 1,126 of 2,235 val samples lack features; Standard split is the practical choice.
5. **Feature paths in scripts**: `features/soccer_gmr/{clip,clip_text,slowfast}` — confirmed present with expected file counts.

### Next Steps
- **Phase 0**: Run `eval/eval_main.py` on existing test predictions to reproduce baseline metrics
- **Phase 0**: Clarify which checkpoint produces Table 2 results
- **Phase 1**: Build `error_analysis.py` reusing `greedy_match` from `metrics.py`

## Session 2 — 2026-06-23: Full Error Analysis Pipeline (Phases 0-4)

### Actions Completed
- [x] Phase 0: Reproduced baseline metrics on test (mAP=7.52, G-mIoU@1=35.84) and val (mAP=8.14, G-mIoU@1=35.41)
- [x] Phase 0: Confirmed local checkpoint underperforms paper (mAP 7.52 vs 18.69); official checkpoint directory no longer exists
- [x] Phase 1: Created `eval/error_analysis.py` — per-query error detail table with scenario labels, match counts, IoU, existence scores
- [x] Phase 1: Generated detail JSONL/CSV and summary JSON for both test and val
- [x] Phase 2: Created `eval/quantify_failures.py` — quantified 6 failure modes with distributions
- [x] Phase 2: Key finding: rejection_FP=171 (34.8% of null-set), rejection_FN=190 (34.9% of positive), 84.1% of matches IoU<0.3
- [x] Phase 2: Systematic time offset: predictions average -20s from GT for low-IoU pairs
- [x] Phase 3: Created `eval/counterfactual.py` — oracle fix ranking
- [x] Phase 3: Boundary fix: mAP +92.48, G-mIoU@1 +25.45 (but unachievable — model has systemic misalignment)
- [x] Phase 3: False positive fix: AUROC +27.91, G-mIoU@1 +16.51 (actionable via existence calibration)
- [x] Phase 4: Implemented `models/moment_detr_gmr/exist_calibrator.py` — temperature scaling + adaptive threshold
- [x] Phase 4: Created `eval/phase4_comparison.py` — full ablation study
- [x] Phase 4: Optimal calibration: temp=0.30, thd=0.63 (learned on val, applied to test)
- [x] Phase 4: Test results: G-mIoU@1 35.84→49.64 (+13.80), Rej-F1@0.4 64.01→74.19 (+10.18)
- [x] Phase 4: Cross-reference: rejection_FP reduced 171→22 (-87%), no impact on other error categories

### Key Results Summary

| Metric | Baseline | Calibrated | Gain |
|--------|----------|------------|------|
| G-mIoU@1 | 35.84 | 49.64 | +13.80 |
| Rej-F1@0.4 | 64.01 | 74.19 | +10.18 |
| Acc@0.4 | 65.15 | 68.44 | +3.29 |
| AUROC | 72.09 | 72.09 | +0.00 |
| mAP | 7.52 | 7.52 | +0.00 |

### Files Created/Modified
- `eval/error_analysis.py` — Phase 1 detail table generator
- `eval/quantify_failures.py` — Phase 2 failure mode quantifier with plots
- `eval/counterfactual.py` — Phase 3 oracle fix ranking
- `eval/phase4_comparison.py` — Phase 4 ablation and comparison
- `eval/calibrate_exist.py` — Standalone calibration script
- `models/moment_detr_gmr/exist_calibrator.py` — Core calibration module
- `results/error_analysis/` — All analysis outputs (detail, summary, plots, counterfactual)
- `results/calibrated/` — Calibrated predictions and metrics
- `results/error_analysis/phase4/` — Phase 4 comparison results
