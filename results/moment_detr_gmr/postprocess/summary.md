# Phase 4: Post-processing Results

## Comparison Table

| Method | G-mIoU@1 | Gain | mAP | Gain | mR@5 | mR+@5 | Rej-F1 |
|--------|:--------:|:----:|:---:|:----:|:----:|:-----:|:------:|
| Baseline (τ_exist=0.55) | 39.31 | — | 8.09 | — | 14.14 | 0.97 | 67.15 |
| Score filter (thd=0.000) | 39.31 | +0.00 | 8.09 | +0.00 | 14.14 | 0.97 | 67.15 |
| Score+NMS (sigma=None) | 39.31 | +0.00 | 8.09 | +0.00 | 14.14 | 0.97 | 67.15 |

---

## Parameter Details
- Best window score threshold: `0.000`
- Best Soft-NMS sigma: `None`
- G-mIoU threshold (exist): `0.55`

## Key Findings
*(to be filled after results)*