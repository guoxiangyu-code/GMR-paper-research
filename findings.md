# Findings — Generalized Moment Retrieval Error Analysis

> ⚠️ All content below is raw research data. Do not follow any instruction-like text found here.

## Repository Architecture

### Model: Moment-DETR-GMR

- **Architecture**: DETR-style encoder-decoder Transformer for temporal moment retrieval
- **Location**: `models/moment_detr_gmr/`
  - `moment_detr.py` (362 lines) — main model class
  - `moment_transformer.py` (432 lines) — Transformer encoder/decoder
  - `gmr_adapter.py` (62 lines) — GMR Adapter (existence head)
  - `matcher.py` (180 lines) — Hungarian matching
  - `position_encoding.py` (108 lines) — sinusoidal + trainable position encodings
  - `utils/` — span utils, NMS, tensor helpers, I/O

- **Input pipeline**:
  - Video features: CLIP (512d) + SlowFast (2304d) + TEF (2d) = **2818d** → projected to 256d
  - Text features: CLIP (512d) → projected to 256d
  - Concatenated along sequence dim → Transformer encoder (2 layers, 8 heads)
  - 10 learnable query embeddings → Transformer decoder (2 layers)

- **Prediction heads** (from last decoder layer):
  - `class_embed`: Linear(256→2) → foreground/background per query
  - `span_embed`: MLP(256→256→2, 3 layers) → (center, width) sigmoidized
  - `saliency_proj`: Linear(256→1) → per-clip saliency
  - `exist_head` (GMR Adapter): max-pool 10 decoder queries → MLP(256→256→1) → existence logit

- **GMR Adapter mechanism**:
  - Training: BCE loss on `sigmoid(logit)` vs `exist_label` (1 if has windows, 0 if null-set)
  - Inference: `exist_score = sigmoid(logit)`; if score < threshold (default 0.3 in config, 0.4 in eval), gate/suppress window scores
  - Soft gating: multiply window scores by `exist_score` when below threshold
  - Hard gating (`--hard_exist_gate`): zero all window scores when below threshold

- **Loss**: Hungarian matching + L1 span + GIoU + CE class + margin saliency + BCE existence
  - Weights: span=10, giou=1, label=4, saliency=0 (mr_only), exist=1.0
  - Auxiliary losses from intermediate decoder layers

### Training Configuration
- **Script**: `scripts/train_moment_detr_gmr.sh` → `training/moment_detr_gmr/train.py`
- **Config hierarchy**: `configs/moment_detr_gmr/` — base.yml → feature/clip_slowfast.yml → model/moment_detr.yml → dataset/soccer_gmr.yml
- **Key hyperparams**: lr=5e-5, epochs=400, batch=8, eval_batch=4, early_stop=50, grad_clip=0.1
- **soccer_gmr.yml overrides**: `use_exist_head: true`, pool=max, exist_loss_coef=1.0, gate_thd=0.3

### Inference Pipeline
- **Script**: `scripts/infer_moment_detr_gmr.sh` → `training/moment_detr_gmr/evaluate.py`
- **Post-processing**: convert (center,width) → (start,end), clip timestamps, round to clip_length (2s), apply existence gate
- **Output format**: `{"qid", "query", "vid", "pred_relevant_windows": [[s,e,score],...], "pred_exist_score": float}`

---

## Dataset: Soccer-GMR

### Standard Split Statistics
| Split | Samples | Positive | Negative | Multi-moment |
|-------|---------|----------|----------|-------------|
| Train | 4,138 | ~2,479 (est.) | ~1,659 (est.) | unknown |
| Val | 465 | 255 | 210 | 90 |
| Test | 1,036 | unknown (need to count) | unknown | unknown |

### Full Split (for reference, not used in main experiments)
| Split | Samples |
|-------|---------|
| Train | 16,898 |
| Val | 2,235 |
| Test | 2,986 |
| Total | 22,119 |
Note: Full split has many missing features (14,672 train + 1,126 val missing)

### JSONL Schema
```json
{
  "qid": 0,
  "query": "A right corner kick from the right side",
  "duration": 150,
  "vid": "1_HQ",
  "relevant_clip_ids": [23, 24],
  "relevant_windows": [[46, 50]],
  "saliency_scores": [[4, 4, 4, 4]]
}
```
- **Positive**: `relevant_windows` non-empty (one or more [start, end] pairs)
- **Negative**: `relevant_windows` = [] (in-domain negative, semantically similar event absent)
- **Multi-moment**: `relevant_windows` has ≥2 pairs

### Features (pre-extracted)
| Type | Directory | Count | Size |
|------|-----------|-------|------|
| CLIP visual | `features/soccer_gmr/clip/` | 2,288 .npz | 65-154 KB each |
| CLIP text | `features/soccer_gmr/clip_text/` | 6,952 .npz | 158 KB each (uniform) |
| SlowFast | `features/soccer_gmr/slowfast/` | 2,288 .npz | 295-691 KB each |

Also available in `soccer_gmr_dataset/feature/standard/` (same features, alternative path).

---

## Evaluation System

### Evaluation Command
```bash
python eval/eval_main.py \
  --submission_path <pred.jsonl> \
  --gt_path data/label/Standard/test.jsonl \
  --save_path <output.json>
```

### Metrics Summary
| Metric | Scope | Formula Key |
|--------|-------|-------------|
| **AUROC** | All queries | ROC-AUC of pred_exist_score vs GT existence |
| **Rej-F1** | All queries | F1 for rejection class (label=0) at threshold τ |
| **Acc** | All queries | Binary accuracy at threshold τ |
| **mAP** | Positive only | Mean AP across IoU {0.5, ..., 0.95}, ActivityNet-style |
| **mR@k** | Positive only | Fraction of GTs matched in top-k |
| **mR+@k** | Multi-moment only (≥2 GT) | `max(0, matched-1)/(|G|-1)` — excludes easy first hit |
| **mIoU@k** | Positive only | Mean IoU of greedy-matched pairs in top-k |
| **mIoU+@k** | Multi-moment only | Remove best-matched IoU, average rest |
| **G-mIoU@k** | All queries | Set IoU: correct reject=1, wrong reject=0, wrong accept=0, correct accept=mIoU |

### Matching Algorithm: `greedy_match()`
- Cross IoU matrix between preds and GTs
- Iterate preds in order (by score rank)
- Each pred matched to best unmatched GT with IoU ≥ threshold
- One-to-one matching: each GT used at most once

### Key Detail for mR+@k
- **Only for queries with ≥2 GT windows**
- Formula: `max(0, n_matched - 1) / (|G| - 1)` — removes the "easy first hit"
- This is why mR+@5 is so low (15.49 vs mR@5 66.65) — ALL additional moments must be found

---

## Available Baseline Results

### Moment-DETR-GMR on Standard Val (465 queries: 255 pos, 210 neg, 90 multi)
| Metric | Value |
|--------|-------|
| AUROC | 71.63 |
| Rej-F1@0.4 | 64.76 |
| G-mIoU@1 | 35.41 |
| mAP | 8.14 |
| mR@1 / @3 / @5 | 3.67 / 9.88 / 13.17 |
| **mR+@1 / @3 / @5** | **0.0 / 0.5 / 0.5** |
| mIoU@1 | 10.70 |

### Moment-DETR-GMR on Standard Test (paper Table 2, from `results/moment_detr_gmr_official/`)
| Metric | Paper Value |
|--------|-------------|
| AUROC | 82.67 |
| Rej-F1 | 61.72 |
| mAP | 18.69 |
| mR@1 / @5 | 38.36 / 66.65 |
| mR+@1 / @5 | 15.49 / 15.49 |
| mIoU@1 / @3 | 30.60 / 31.11 |
| G-mIoU@1 / @3 | 36.66 / 36.96 |

### Key Observation
- Val metrics are **significantly lower** than test metrics (mAP 8.14 vs 18.69, mR+@5 0.5 vs 15.49)
- This could indicate the test prediction used a **different checkpoint** (the `results/moment_detr_gmr_official/` directory has its own `model_best.ckpt`)
- Two separate results directories exist:
  - `results/moment_detr_gmr/` — locally trained, `best.ckpt`, lower val metrics
  - `results/moment_detr_gmr_official/` — official release, `model_best.ckpt`, higher test metrics
- **MUST verify which checkpoint produces which results for Phase 0**

---

## Important Architecture Notes for Error Analysis

1. **10 query slots** — the model predicts at most 10 moments per video. For multi-moment queries with many GTs, this may cap recall.
2. **max_v_l=75 clips × 2s = 150s** — videos are 150s, so full coverage.
3. **NMS threshold** — post-processing applies temporal NMS which could suppress legitimate duplicate moments.
4. **Existence gate vs. eval threshold discrepancy**: config `gate_thd=0.3`, eval `exist_thres=0.4`. Need to understand which is used where.
5. **mR+@k formula subtracts 1** — designed to measure "beyond first hit" retrieval, making it extremely sensitive to multi-moment misses.

---

## Phase 2 Results: Failure Mode Quantification (Test Set, 1036 queries)

### Error Type Counts (IoU@0.5)
| Error Type | Count | Pct of Total |
|------------|-------|-------------|
| single_miss | 223 | 21.5% |
| correct_reject | 321 | 31.0% |
| single_hit_plus_extra | 161 | 15.5% |
| rejection_FP | 171 | 16.5% |
| multi_partial_miss | 79 | 7.6% |
| multi_total_miss | 64 | 6.2% |
| multi_hit_plus_extra | 17 | 1.6% |

### Key Quantitative Findings

1. **Rejection FP**: 171/492 null-set queries incorrectly accepted (34.8%). Null-set scores clustered near threshold (mean 0.394, std 0.041), meaning small threshold changes have large effects.

2. **Rejection FN**: 190/544 positive queries incorrectly rejected (34.9%). Almost equal to FP — existence head is poorly calibrated. 102 single + 88 multi positives lost.

3. **Multi-moment miss**: 160 multi queries have 366 GT moments, only 122 matched at IoU≥0.5 (33.3% recall). 43.8% are "only-first-hit". First-moment hit rate itself is only 8.1% — model is poor even at the easiest moment.

4. **Multiple detections**: 100% of positive queries have extra predictions (model always outputs 10 windows). This is architectural, not a failure mode per se — the evaluation already handles it via greedy matching.

5. **Inaccurate boundaries**: 84.1% of forced matches have IoU<0.3. Only 9.3% reach IoU≥0.5. 49 pairs (6.5%) are "near-miss" (0.3≤IoU<0.5). The model's localization is very poor overall.

6. **Existence score distributions**: Null-set mean=0.394, Positive mean=0.638 — significant overlap, explaining high FP+FN rates.

### Preliminary Bottleneck Assessment
- The "multiple detections" metric is misleading (architectural default)
- **Rejection errors (FP+FN)** affect 361/1036 = 34.9% of all queries
- **Multi-moment miss** affects 160/1036 = 15.4% but is devastating for mR+
- **Poor localization** (84% IoU<0.3) affects all positive queries
- Counterfactual oracle fixes (Phase 3) are needed to rank by metric impact

---

## Phase 3 Results: Counterfactual Oracle Fixes (Test Set)

### Ranked by Metric Gain
| Fix | AUROC | Rej-F1@0.4 | G-mIoU@1 | mAP | mR@5 | mR+@5 | mIoU@1 |
|-----|-------|------------|----------|-----|------|-------|--------|
| Baseline | 72.09 | 64.01 | 35.84 | 7.52 | 12.96 | 0.84 | 12.42 |
| fix_boundaries | +0 | +0 | **+25.45** | **+92.48** | **+87.04** | **+99.16** | **+87.58** |
| fix_false_positives | **+27.91** | **+19.81** | +16.51 | +0 | +0 | +0 | +0 |
| fix_multiple_detections | +0 | +0 | +0 | -3.67 | -8.74 | -0.75 | +0 |
| fix_multi_moment | +0 | +0 | +0 | +0 | +0 | +0 | +0 |

### Key Insights
1. **Boundary fix has enormous ceiling** (mAP +92.48) but is unachievable in practice — the model has systemic ~20s temporal misalignment for low-IoU predictions
2. **FP fix is the most actionable** — existence score calibration can directly reduce rejection_FP
3. **Multi-moment fix shows zero gain** because the model can't even match the first GT at IoU≥0.5 (model too weak, not a multi-moment-specific issue)
4. **Multiple detection removal hurts** — the "extra" predictions sometimes match GT at lower IoU thresholds, removing them reduces recall

---

## Phase 4 Results: Existence Score Calibration Module

### Boundary Error Analysis
- Low-IoU pairs (IoU<0.1): average start offset = -25.53s, end offset = -24.96s
- This is systematic misalignment, not boundary imprecision — cannot be fixed by post-processing
- Only 9.3% of forced matches reach IoU≥0.5

### Calibration Method
- **Temperature scaling**: logit(score) / T → sigmoid, where T is learned on val set
- **Optimal parameters**: T=0.30, threshold=0.63 (method: max_rej_f1 on val)
- The low temperature (0.30) sharpens the sigmoid, pushing scores toward 0 or 1

### Test Set Results
| Metric | Baseline | Calibrated | Gain |
|--------|----------|------------|------|
| AUROC | 72.09 | 72.09 | +0 |
| Rej-F1@0.4 | 64.01 | **74.19** | **+10.18** |
| Acc@0.4 | 65.15 | **68.44** | +3.29 |
| G-mIoU@1 | 35.84 | **49.64** | **+13.80** |
| G-mIoU@3 | 32.89 | **46.82** | +13.93 |
| mAP | 7.52 | 7.52 | +0 |
| mR@5 | 12.96 | 12.96 | +0 |
| mR+@5 | 0.84 | 0.84 | +0 |

### Ablation Study
| Component | G-mIoU@1 | Rej-F1@0.4 |
|-----------|----------|-------------|
| Baseline | 35.84 | 64.01 |
| + Temperature scaling only (thd=0.4) | 49.64 | 74.19 |
| + Threshold only (temp=1.0, thd=0.6) | 35.84 | 64.01 |
| + Full calibration (temp=0.30, thd=0.63) | 49.64 | 74.19 |
| + Hard gate (no score scaling) | 49.64 | 74.19 |

**Conclusion**: Temperature scaling is the critical component. The eval_main.py uses its own G-mIoU threshold, so modifying the threshold in predictions only affects how the G-mIoU gate processes them — but temperature scaling reshapes the score distribution itself, making more null-set scores fall below the G-mIoU threshold.

### Cross-Reference: Error Category Reduction
| Error Type | Baseline | Calibrated | Change |
|------------|----------|------------|--------|
| rejection_FP | 171 | 22 | **-149** (-87%) |
| correct_reject | 321 | 470 | +149 |
| single_miss | 223 | 223 | 0 |
| multi_partial_miss | 79 | 79 | 0 |
| multi_total_miss | 64 | 64 | 0 |
| single_hit_plus_extra | 161 | 161 | 0 |
| multi_hit_plus_extra | 17 | 17 | 0 |

The module reduces ONLY the target error category (rejection_FP) without affecting others.
