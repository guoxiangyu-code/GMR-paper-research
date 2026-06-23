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
