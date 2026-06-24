# Oracle Fix Ranking — Moment-DETR-GMR

> Evaluation threshold for G-mIoU: τ = 0.55

## Gain Table (ranked by G-mIoU@1 gain)

| Fix Type | G-mIoU@1 | Gain | mAP | Gain | Rej-F1 | Gain | Recommended Direction |
|---------|:--------:|:----:|:---:|:----:|:------:|:----:|----------------------|
| **Baseline** | 39.31 | — | 8.09 | — | 67.15 | — | — |
| **Fix All** 🏆 | 91.66 | +52.35 | 100.00 | +91.91 | 100.00 | +32.85 | — |
| **Fix Over-detection** | 58.23 | +18.92 | 20.64 | +12.55 | 67.15 | +0.00 | NMS / score thresholding post-processing |
| **Fix FP** | 51.18 | +11.87 | 8.09 | +0.00 | 80.52 | +13.37 | candidate verification (exist_head calibration / null-set detection) |
| **Fix Boundary** | 44.43 | +5.12 | 38.83 | +30.74 | 67.15 | +0.00 | boundary refinement (temporal boundary regression) |
| **Fix Multi-miss** | 39.31 | +0.00 | 8.09 | +0.00 | 67.15 | +0.00 | coverage-aware retrieval (multi-moment detection) |
| **Fix All** (upper bound) | 91.66 | +52.35 | 100.00 | +91.91 | 100.00 | +32.85 | — |

---

## 结论与方向选择

**指标增益最大的修复**: `Fix All`  
**对应改进方向**: N/A

### 映射关系
| 主要瓶颈 | 改进方向 |
|---------|---------|
| 多时刻漏检 | Coverage-aware retrieval (改进 decoder 多样性) |
| 误报为主 | Candidate verification (null-set 判别器) |
| 边界偏移 | Boundary refinement (后处理时序边界回归) |
| 多检 | NMS / 分数阈值 post-processing |

---

## 下一步 (Phase 4)
- 基于 `Fix All` 的方向: **N/A**
- 设计具体模块, 见 task_plan.md Phase 4