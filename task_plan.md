# /goal — Confidence-Position Disentangled Reranking for Multi-Target Moment Retrieval (Soccer-GMR)

## ROLE
你是一个执行 ML 研究工程的 agent。你将基于已有的 Moment-DETR 骨干，在 Soccer-GMR 数据集上
诊断并修复"多目标坍缩 (multi-target collapse)"问题。你必须严格按 STAGE 顺序执行，每个 STAGE
带 GATE 闸门，未过闸门不得进入下一阶段。所有产物必须落盘、可复现。

## 背景结论 (已由前期诊断确立，作为不可推翻的前提)
- 任务: GMR (Generalized Moment Retrieval)，同一 query 需命中多个事件 (如 进球+黄牌)。
- 骨干: Moment-DETR，span head = 点回归 (center,width)，监督 = L1 + gIoU，**不预测任何分布**。
  => 因此"从 L1 提取边界分布锐度"是伪命题，禁止尝试 (除非到 STAGE 4 换 DFL 头)。
- 病因 = confidence-position entanglement: existence 评分被数据集时序位置先验主导。
  证据: GT=2 切片中第二目标"物理命中率 19.53% (25/128)"，但最终 mR+@5 只有 1.17%。
       top-5 高分 query 的预测窗口在不同样本间近乎固定 ([12-22][46-58][74-90]...) => 位置模板。
- Oracle 上界 = 14.06% (完美重打分)。子弹在膛里，缺的是无位置偏见的扳机。
- 已失败且禁止重走的路: 硬截断(梯度断裂) / CVPR -SA+QD(加剧坍缩) / 排斥Loss(过度设计) /
  encoder 特征平抑 / 整段 mean-pool 跨模态相似度(1.48%) / saliency 窗口均值(1.48%)。

## HARD CONSTRAINTS (违反即视为任务失败)
- [C1] 全程冻结 backbone 与 span head，梯度绝不流入定位。只动 score (重排)。
- [C2] 重打分器输入严禁任何绝对位置 (center / 归一化坐标 / 位置编码)。width 允许 (非绝对位置)。
- [C3] 不再调排斥 Loss，不在 encoder 做特征平抑。
- [C4] 所有指标用跨阈值平均 (τ ∈ {0.3,0.4,0.5,0.6,0.7})，禁止用单点 τ=0.4 冒充平均。
- [C5] 关键配置必须跑 ≥3 seeds，报 mean±std (1% 地板区信号可能低于测量噪声)。
- [C6] 负结果 (failed zero-shot 等) 一律归档，不得删除。

---

## CODE REPOSITORIES (均已核实，2026-06 可用)

| 用途 | 仓库 | clone URL | 何时需要 |
|---|---|---|---|
| 主骨干 (你已在用) | Moment-DETR (NeurIPS'21) | `https://github.com/jayleicn/moment_detr` | STAGE 0-2，确认 span head / saliency / hs / xattn 暴露点 |
| 跨骨干验证-A | EaTR (ICCV'23) | `https://github.com/jinhyunj/EaTR` | 仅 STAGE 3，且 G2 通过后 |
| 跨骨干验证-B | FlashVTG (WACV'25) | `https://github.com/Zhuo-Cao/FlashVTG` | 仅 STAGE 3，且 G2 通过后 |
| 参考/对照实现 | QD-DETR (CVPR'23) | `https://github.com/wjun0830/QD-DETR` | 可选，CVPR -SA+QD 探针对照、特征对齐参考 |

注:
- EaTR / FlashVTG / QD-DETR 的 QVHighlights 特征均沿用 Moment-DETR 的 SlowFast+CLIP 格式，
  这是跨骨干特征对齐的有利条件；但 Soccer-GMR 自有特征是否一致需在 STAGE 3-PREP 显式核对。
- FlashVTG 数据准备遵循 CG-DETR 说明；EaTR/QD-DETR anaconda 环境参考 Moment-DETR 官方。
- 每个跨骨干仓库必须建独立 conda/venv，依赖大概率冲突。

---

## STAGE 0 — 基线与上界自检 (GATE G0)
```
FUNCTION stage0():
    load_ckpt(baseline_Idea1)
    m = evaluate(val, tau_sweep=[0.3,0.4,0.5,0.6,0.7])
    ASSERT |m.mRplus@5 - 1.17%| < 0.3%
    ASSERT |m.G_mIoU@1 - 50.83%| < 1.0%
    oracle = simulate_perfect_rerank("diagnostic_gt2_analysis.json")
    ASSERT |oracle.mRplus@5 - 14.06%| < 0.5%
    SAVE results/anchor.json {baseline, oracle_ceiling=14.06}
GATE G0: 基线+Oracle 均复现 -> STAGE 1; 否则停止报环境不一致。
```

## STAGE 1 — P0-a: Saliency 峰值对比度 zero-shot 重测 (GATE G1, 当天出数)
```
FUNCTION stage1():
    FOR sample IN val:
        sal = saliency_scores(sample)            # 逐帧, 长度 T
        FOR cand i IN top10:
            (s,e)=window_i
            sharpA = max(sal[s:e]) - median(sal[0:T])      # 方案A 差值
            sharpB = max(sal[s:e]) / (median(sal[s:e])+eps)# 方案B 比值
        rerank by sharpA (再单独跑 sharpB)
    LOG results/stage1_saliency_contrast/  (A、B 两组, tau_sweep)
GATE G1 (诊断性, 不阻断):
    若升到 ~3-4%: 标记 saliency 含内容信号 -> 进 STAGE2 特征池(强证据)
    若 <2%:      仍保留为 STAGE2 一个输入特征, 不单独立 claim
    无论如何 -> 继续 STAGE 2
```

## STAGE 2 — P0-b: 离线 Position-Free 重打分头 (主战场, GATE G2)

### 2.1 特征 dump
```
FUNCTION dump_features(split in {train,val}):
    cache=[]
    FOR sample IN split:
        forward(model, sample)   # no_grad, 冻结
        FOR cand i IN topK(=10):
            f={}
            f.hs            = query_embed_i                  # d 维, 最后层 decoder
            f.xattn_entropy = entropy(decoder_last_xattn_i)  # 注意力弥散度
            f.sal_sharp     = peak_contrast(sal, window_i)   # 复用 STAGE1 公式
            f.width         = (e - s)                        # 允许
            f.xmodal_align  = fine_grained_align(hs_i, txt_mem)
                              # 细粒度! token级 max/attn-pool; 禁止整段 mean-pool
            f.label = 1 if maxIoU(window_i, GT_set) >= 0.5 else 0
            f.iou   = maxIoU(window_i, GT_set)               # 仅分析, 不喂模型
            f.exist = existence_score_i                      # 融合用
            f.gt_cnt= len(GT_set); f.qid=sample.qid
            ASSERT 'center' NOT IN f AND 'norm_pos' NOT IN f # [C2]
            cache.append(f)
    SAVE results/rerank_cache_{split}.pt
    LOG feat_dim, #samples, pos/neg ratio per gt_cnt
```

### 2.2 离线训练小 MLP
```
CLASS RerankHead:
    in  = concat[ proj(hs), xattn_entropy, sal_sharp, width, xmodal_align ]
    net = Linear(in,128)->ReLU->Dropout(0.1)->Linear(128,1)
    out = scalar r_i

FUNCTION train_rerank(seed):
    set_seed(seed); head=RerankHead()
    FOR epoch IN E:
        FOR sample(top-K组) IN train_cache:
            P={i:label==1}; N={i:label==0}
            IF |P|==0 or |N|==0: skip
            L = mean_{p in P, n in N} softplus(margin - (r_p - r_n))  # margin=0.5
            backward(L); step()
            ASSERT backbone.grad is None AND span_head.grad is None    # [C1]
        validate_log(epoch)
    SAVE results/rerank_head_seed{seed}.pt
    # 数据红利: 排序监督对 GT=1 同样成立 => 训练信号全量, 非仅 565 条 GT=2
```

### 2.3 评测端融合 + 分层 (GATE G2)
```
FUNCTION evaluate(submission): Rescoring head training (STAGE 2.2) & inference
[x] Define MLP with input dim = 256 (hs) + 1 (entropy) + 1 (sal) + 1 (width) + 1 (align) = 260
[x] Train using cache, MarginRankingLoss(score_p, score_n) + L2_Reg
[x] eval(scores=a*exist + b*r_i, tau_sweep) 对所有tau_sweep的结果取均值
[x] Record conclusions. If < 6%, conclude position is necessary.

Phase 6: Result wrap-up
[x] Append findings to findings.md
[x] Present results to user.

FUNCTION evaluate_rerank(head, val_cache):
    R={}
    R['pure_r'] = eval(scores=r_i, tau_sweep)
    FOR (a,b) IN grid(alpha,beta in [0.3..0.7]):
        R['fuse_%.1f_%.1f'%(a,b)] = eval(scores=a*exist + b*r_i, tau_sweep)
    FOR subset IN {all, gt1, gt2}:
        report mRplus@5, G_mIoU@1, Rej-F1
    ASSERT G_mIoU@1(after) >= G_mIoU@1(baseline) - 1.0%    # 第一目标不许塌
    SAVE results/stage2_main/ (含 3 seeds mean±std)
GATE G2 (主闸门):
    现实目标 6%~10% (不可能贴满 14.06%)
    >=6% 且第一目标未塌 -> 决定性证据成立 -> STAGE 3
    2%~6%               -> 跑 2.4 消融找瓶颈再判, 不许直接进 STAGE3
    <=2%               -> 排查 [C2]位置泄露 / IoU阈值 / 特征粒度, 不许进 STAGE3
```

### 2.4 特征消融 (实验完整性必需)
```
FUNCTION ablation():
    FOR feat IN {hs, xattn_entropy, sal_sharp, width, xmodal_align}:
        train+eval WITHOUT feat -> LOG delta_mRplus@5
    # 关键反例对照: 故意加 center
    train+eval WITH center -> 预期 GT=2 召回不升 (证明 [C2] 必要性)
    SAVE results/stage2_ablation.csv   # 进论文
```

## STAGE 3 — P1: 跨骨干普适性 (GATE G3，仅 G2 通过后)
### 3-PREP 环境搭建
```
FUNCTION decide_start():
    IF G2 未通过: 暂缓, 不要克隆 (主方法没立住前跨骨干只增噪声)
FUNCTION prep(bb in {EaTR, FlashVTG}):
    git clone <对应 URL>; 建独立 env; 装依赖; 下载官方 ckpt
    ASSERT 官方数据(QVHighlights)能复现 paper 数 (±1%)   # 先跑通官方再碰自己数据
    adapt Soccer-GMR:
        [a] 视频特征是否与该骨干一致? 不一致需重抽 (最大隐形成本, 否则比较不公平)
        [b] query 编码接口  [c] 多目标 GT 格式  [d] 拒答样本移植/标 N/A
    暴露中间量: hs / xattn / saliency / existence (缺者置空并记录为"特征可用性差异")
```
### 3-RUN
```
FUNCTION stage3(bb):
    run GT=2 diagnostic FIRST (复用 diagnostic 流程): 物理命中率/置信度断崖/位置模板?
    GATE G3-0:
        复现坍缩 -> 移植 STAGE2 重打分管线复测
        不复现   -> 结论收窄为"点回归头特有" (干净对照, 同样可发)
GATE G3: 坍缩在 >=2 骨干复现 -> claim="通用现象"; 否则 claim 收窄。
```

## STAGE 4 — P2: DFL 分布边界头 (GATE G4，仅 G2 通过后做强)
```
FUNCTION stage4():
    replace span_head -> DFL: start/end 离散成 bins, 预测分布, 期望解码, DFL loss 监督
    sharpness_i = -entropy(start_dist) - entropy(end_dist)   # 内生置信度
    retrain end-to-end; eval tau_sweep; 分层 GT=1/GT=2
GATE G4: 内生 sharpness 重打分 >= 外挂 MLP -> 作论文主方法; 否则 MLP 为主, DFL 入讨论。
```

---

## 实验完整性 CHECKLIST (宣布 DONE 前逐项自检)
```
[ ] 所有指标 tau_sweep 平均, 无单点冒充 ([C4])
[ ] 关键配置 >=3 seeds, mean±std ([C5])
[ ] 三对照齐全: 基线(1.17%) / 本方法 / Oracle(14.06%)
[ ] 分层报告 all / GT=1 / GT=2
[ ] 守门指标未退化: Rej-F1>=74.5%-ε, G-mIoU@1>=50.83%-1%, 第一目标召回未塌
[ ] 特征消融表完整 (含 center 反例)
[ ] 训练日志可证全程梯度未入 backbone/span_head ([C1])
[ ] 每个 run 的 config/seed/ckpt/metrics 落盘可复现
[ ] failed zero-shot (1.48%) 与 saliency-contrast 负结果归档 ([C6])
```

## 产出目录
```
results/
  anchor.json
  stage1_saliency_contrast/
  rerank_cache_{train,val}.pt
  rerank_head_seed{0,1,2}.pt
  stage2_main/
  stage2_ablation.csv
  stage3_cross_backbone/{eatr,flashvtg}/
  stage4_dfl/
  REPORT.md   # 基线->本方法->Oracle, 含 std 与守门指标
```

## EXECUTION ORDER (agent 必须遵守)
G0 -> STAGE1(当天) -> STAGE2(主战场, 卡在此处直到 G2>=6%) -> [G2通过后] STAGE3 / STAGE4。
未过 G2 不得 clone EaTR/FlashVTG。
