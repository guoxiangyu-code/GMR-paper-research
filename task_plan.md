# Task Plan: Generalized Moment Retrieval — Systematic Error Analysis & Targeted Improvement

## Goal

Conduct a systematic, data-driven error analysis of the Moment-DETR-GMR baseline on the Soccer-GMR benchmark, quantify failure modes, rank bottlenecks via counterfactual oracle fixes, and implement a minimal module targeting the highest-impact bottleneck. All conclusions must be grounded in reproducible statistics from the official evaluation scripts.

## Constraints

1. **No massive frameworks** — First pinpoint a primary failure mode, then build a small, effective module.
2. **Data-driven** — All conclusions based on official eval metrics (`eval/eval_main.py`, `eval/metrics.py`).
3. **Reproducible** — Use the existing codebase, features, and checkpoints.
4. **FlashVTG-GMR unavailable** — Use Moment-DETR-GMR (available) for the full pipeline; note that final conclusions should ideally be validated on the strongest baseline.

## Baseline Context

- **Model**: Moment-DETR-GMR (DETR-style encoder-decoder + GMR Adapter existence head)
- **Dataset**: Soccer-GMR Standard split (train: 4,138 / val: 465 / test: 1,036)
- **Features**: CLIP visual (512d) + SlowFast (2304d) + TEF (2d) = 2818d video; CLIP text (512d)
- **Checkpoint**: `results/moment_detr_gmr/best.ckpt` (58.7 MB)
- **Existing predictions**: test (1,035 lines), val (464 lines) — already generated
- **Paper threshold**: τ = 0.4 for existence score gating

---

## Phase 0 — Environment Setup & Reproduce Baseline
- **Status**: complete
- **Goal**: Verify environment, reproduce baseline metrics, align with paper Table 2.
- **Tasks**:
  1. Install dependencies from `requirements.txt` (PyTorch, scikit-learn, etc.)
  2. Verify feature files exist and match expected counts (2,288 CLIP/SlowFast video, 6,952 CLIP text)
  3. Run official evaluation on the **existing** test predictions: `python eval/eval_main.py --submission_path results/moment_detr_gmr/test/moment_detr_gmr_test_submission.jsonl --gt_path data/label/Standard/test.jsonl --save_path results/moment_detr_gmr/test/test_metrics_results.json`
  4. Compare reproduced metrics with paper Table 2 (Moment-DETR-GMR row): AUROC 82.67, Rej-F1 61.72, mAP 18.69, mR@5 66.65, mR+@5 15.49, G-mIoU@1 36.66
  5. Record any discrepancies and the `pred_exist_score` threshold used
- **Deliverables**: Reproduced metrics table, discrepancy report
- **Files involved**: `eval/eval_main.py`, `eval/metrics.py`, existing prediction JSONL

## Phase 1 — Construct Error Analysis Detail Table (`error_analysis.py`)
- **Status**: `complete`
- **Goal**: Build a per-query error breakdown table without modifying official eval logic.
- **Tasks**:
  1. Create `eval/error_analysis.py` — reuse matching functions from `eval/metrics.py`
  2. Align submission and GT by `qid`; assign scenario label: `null-set` (|G|=0), `single` (|G|=1), `multi` (|G|≥2)
  3. For each positive sample: greedy one-to-one matching at IoU thresholds {0.5, 0.55, ..., 0.95}; record matched/missed/extra counts and per-match IoU
  4. For each null-set sample: record `pred_exist_score` and whether incorrectly accepted (score > τ)
  5. Output per-query detail CSV/JSONL: `qid`, scenario, |G|, |pred|, match_count, miss_count, extra_count, avg_iou, pred_exist_score, error_type_label
- **Deliverables**: `eval/error_analysis.py`, detail output file
- **Files involved**: `eval/metrics.py` (reuse `greedy_match`, `compute_temporal_iou_batch_cross`)

## Phase 2 — Quantify Six Failure Modes
- **Status**: `complete`
- **Goal**: Quantify the proportion and typical cases for each failure mode.
- **Tasks**:
  1. **Rejection FP** (null-set accepted): Plot FP curve vs `pred_exist_score` threshold; find highest-score semantically-similar negative as case study
  2. **Rejection FN** (positive incorrectly rejected): FN proportion and impact on mAP
  3. **Multi-moment miss** (only first hit): For |G|≥2, compute "hit rate for 1st moment" vs "hit rate for subsequent moments"; quantify "only-first-hit" proportion
  4. **Multiple detections**: Proportion where |pred| > |G|; score distribution of redundant boxes
  5. **Inaccurate boundaries**: IoU histogram of matched pairs; percentage of "near-miss" (0.3 ≤ IoU < 0.5)
  6. Summary table + key distribution plots; identify primary bottleneck in one sentence
- **Deliverables**: Summary table, distribution plots (saved to `results/error_analysis/`), bottleneck identification
- **Files involved**: Output from Phase 1, visualization scripts

## Phase 3 — Counterfactual (Oracle) Fixes to Rank Bottlenecks
- **Status**: `complete`
- **Goal**: Rank bottlenecks by metric gain from oracle fixes, not intuition.
- **Tasks**:
  1. Fix multi-moment misses: Treat missed subsequent GTs as correctly recalled → re-evaluate
  2. Fix boundaries: Set IoU of all matched pairs to 1.0 → re-evaluate
  3. Fix false positives: Correctly reject all null-set samples → re-evaluate
  4. Fix multiple detections: Remove all redundant predictions → re-evaluate
  5. Record G-mIoU@1, mAP, Rej-F1 gains for each fix
  6. Rank by metric gain; select highest-impact direction
  7. Map to strategy: multi-moment misses → coverage-aware retrieval; FP → candidate verification; boundary → boundary refinement
- **Deliverables**: Counterfactual results table, ranked bottleneck list, selected direction with justification
- **Files involved**: `eval/eval_main.py`, modified prediction files

## Phase 4 — Implement & Validate Minimal Module
- **Status**: `complete`
- **Selected direction**: Existence Score Calibration (temp=0.30, thd=0.63 learned on val)
- **Module**: `models/moment_detr_gmr/exist_calibrator.py` (post-processing, no retraining)
- **Results on Test**:
  - G-mIoU@1: 35.84 → 49.64 (+13.80)
  - Rej-F1@0.4: 64.01 → 74.19 (+10.18)
  - rejection_FP reduced from 171 → 22 (-87%)
  - No degradation on localization metrics (mAP, mR, mIoU unchanged)
- **Goal**: Design and implement a small, toggleable module targeting only the selected bottleneck.
- **Tasks**:
  1. Integrate module into `models/` or post-processing stage; add `--use_xxx` flag
  2. Train/infer using existing scripts; tune hyperparameters on val set
  3. Evaluate on test set with official metrics
  4. Strict comparison: same baseline with module ON vs OFF
  5. Ablation study: remove key design elements to verify effectiveness
  6. Cross-reference with Phase 2 detail table to confirm "module reduces only the target error category"
- **Deliverables**: Comparison table, change list, cross-reference evidence
- **Files involved**: `models/moment_detr_gmr/`, `training/moment_detr_gmr/`, `scripts/`

---

## Decisions Log

| Decision | Rationale | Date |
|----------|-----------|------|
| Use Moment-DETR-GMR (not FlashVTG-GMR) | FlashVTG-GMR requires NDA; Moment-DETR-GMR is fully available | 2026-06-23 |
| Start with existing predictions | Test/val predictions already generated, no need to re-infer for error analysis | 2026-06-23 |
| Use Standard split (not Full) | Standard split has complete features (2,288 videos); Full split has missing features | 2026-06-23 |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| Official checkpoint directory missing | 1 | Used local checkpoint; noted metrics gap in findings |
| ModuleNotFoundError for exist_calibrator | 2 | Fixed via PYTHONPATH; eval scripts need proper sys.path setup |
| matplotlib not in gmr env | 1 | Installed via conda run pip install matplotlib |
| thd_only_0.6 ablation shows no gain | 1 | eval_main uses its own G-mIoU threshold; temperature scaling needed to reshape distribution |

## Phase 0 Results

### Test Baseline (local checkpoint)
| Metric | Local | Paper Table 2 |
|--------|-------|---------------|
| AUROC | 72.09 | 82.67 |
| Rej-F1@0.4 | 64.01 | 61.72 |
| G-mIoU@1 | 35.84 | 36.66 |
| mAP | 7.52 | 18.69 |
| mR@5 | 12.96 | 66.65 |
| mR+@5 | 0.84 | 15.49 |
| mIoU@1 | 12.42 | 30.60 |

### Val Baseline (local checkpoint)
| Metric | Value |
|--------|-------|
| AUROC | 71.63 |
| Rej-F1@0.4 | 64.76 |
| G-mIoU@1 | 35.41 |
| mAP | 8.14 |
| mR@5 | 13.17 |
| mR+@5 | 0.5 |

### Key Discrepancy
Local checkpoint significantly underperforms paper (mAP 7.52 vs 18.69 on test). Official checkpoint directory no longer exists. Error analysis will proceed with local checkpoint results — failure modes are likely amplified, making patterns easier to identify.
