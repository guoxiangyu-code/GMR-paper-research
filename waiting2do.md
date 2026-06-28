# 角色与总目标
你是 ML 研究工程师,在 Moment-DETR-GMR(Soccer-GMR 基准)上攻克唯一未解的"绝症":mR+@5≈1% 的多目标坍缩。
参照论文/home/guoxiangyu/Base_code/generalized-moment-retrieval/《Beyond Caption-Based Queries for VMR》(CVPR 2026).pdf。
对应的官方代码仓库为：/home/guoxiangyu/Base_code/Beyond_Caption-Based_Queries_for_Video_Moment_Retrieval
它把该现象命名为active decoder-query collapse,并给出验证过的正解 -SA+QD。本任务分两阶段:先诊断(零训练成本),再实现正解。
全程实验隔离:每个动作在 experiments/{日期}_{阶段}_{轮次}/ 新建文件夹,改脚本复制到该文件夹改副本、跑副本,
不动仓库公共代码;config、diff、预测、指标、图、run.md 全部落盘该文件夹,结论可溯源。

# 已知前提(基线,你当前 Idea 1 双轨制模型)
mR+@5≈1.12%,mR@5≈10.27,mAP≈7.11,Rej-F1(τ=0.6)≈74.5,Avg G-mIoU@1≈36.97。
Idea 1(slot 拒答)已对齐并略超官方,本阶段不得破坏其拒答与定位指标(作为守护指标)。
铁律:严禁再用 Idea 2 的强制顺序匹配 / CountHead;论文 Table 5/6 已证明改匹配(1-to-k/Group/Hybrid)
会激活更多 query 却产生冗余框、retrieved GT 反降。

# ============ 阶段一:诊断(不训练,只测当前模型,复现论文 Fig.5)============
目的:用自己的数据证明 active decoder-query collapse 存在,把"mR+ 低"升级为有机制、可度量的问题。
STEP D1 定义"active query":对每个查询,统计置信度不衰减的 decoder query 数量。
  - 论文定义:confidence 不 vanish 的 query 即 active。实现上取每个 slot 的前景/分类置信度,
    用一个低阈值(如 >0.05,与论文 Fig.L 的 IOU≥0.1 匹配口径一致)判定是否 active;阈值需在 run.md 注明并做敏感性检查。
STEP D2 画核心曲线:x 轴 = GT moment 数(1,2,3,4+ 分桶),y 轴 = 平均 active query 数(带标准差)。
  - 论文基线实测:无论 moment 多少,active query 恒定≈4(Soccer-GMR 上数值可能不同,但应近似水平线)。
  - 同时画第二条参考线:y = GT moment 数(理论需求)。两线交叉点之后即"compute budget 不足"区。
STEP D3 量化坍缩证据(对齐论文 Table 5 的列):统计并报告
  - # active(平均激活 query 数)、%match P(命中任一 GT 的预测占比)、%match GT(被检索到的 GT 占比);
  - 按 moment 数分桶分别给出。预期:active 数不随 moment 增长 → 多目标查询 %match GT 显著偏低。
STEP D4 校准排除(对齐论文 Sec.F):画"query 置信度 vs 回归质量(IoU≥0.1 命中率)"散点。
  - 若低置信 query 的回归质量确实更差(论文结论),则证明问题不在校准、而在"激活的 query 太少",
    从而排除"靠温度缩放/校准救多目标"这条死路,为 -SA+QD 提供 motivation。
阶段一产出:active-vs-moment 曲线图(最有力的 motivation 图)、Table 5 式统计表、校准散点图、diagnosis.md。
验收:能明确陈述"当前模型 active query 数≈N(基本恒定),在 ≥2 moment 查询上 compute budget 不足",即复现坍缩。

# ============ 阶段二:实现 -SA+QD(论文正解,主攻)============
论文把坍缩拆成两个成因,必须两招同时上(Table 7 证明单独一招几乎无效,组合才接近翻倍 active query)。

STEP I1 删除 decoder 自注意力(治 coordination collapse)
  - 论文式(3)→(4):标准层 Q_{l+1}=FFN(CA(SA(Q_l),M)) 改为 Q_{l+1}=FFN(CA(Q_l,M)),即移除每个 decoder 层的 self-attention,
    保留 cross-attention + FFN,损失不变。文件:models/moment_detr_gmr/transformer.py(或 decoder 层定义处)。
  - 理由:SA 让 query 互相"商量"谁负责谁闭嘴,导致多数 query 失活;删掉让每个 query 独立决策。
STEP I2 推理加 NMS(补回删 SA 失去的去重)
  - 删 SA 后无去冗余机制,论文用 NMS 后处理过滤重叠/冗余预测。接到 evaluate.py 双轨制的"正样本全保留+排序"之后:
    对排序候选做时序 NMS(建议 IoU 阈值 0.7,与同类工作一致,需在验证集微调),再输出。空集判定仍由 Idea 1 的 Noisy-OR 负责。
  - 关键:NMS 只去重、不靠分数阈值砍候选,保持你双轨制对 mAP/mR 的保护。
STEP I3 query dropout(治 index collapse)
  - 论文式(5):训练每次迭代以 keep 概率 (1-k) 对可学习 query 做 Bernoulli mask,Q̂ = Q ⊙ M。
  - k=0.25(论文 Table 8 网格搜索最优;k=0.5 会崩到 mAPm≈3.8,务必不要用 0.5)。
  - 加在 query embedding 进 decoder 之前;仅训练期生效,推理关闭。文件:models/moment_detr_gmr/moment_detr.py。
STEP I4 保留 Hungarian 一对一匹配(铁律,治"激活后冗余")
  - 不改匹配器。论文 Table 6 证明:-SA+QD 必须搭配 1-to-1 匹配维持 query 间竞争与多样性;换 1-to-k 会让 query 坍缩成冗余、mAPm 腰斩。
  - 这正是你 Idea 2 强制匹配 mAP 崩到 1.67 的同款病理,严禁重蹈。
STEP I5 与 Idea 1 共存
  - slot 前景头、Noisy-OR 拒答、双轨制推理全部保留;-SA 改的是 decoder 层结构,QD 改的是 query 输入,二者与拒答头不冲突。
  - 训练损失保持 Idea 1 的 slot BCE + 主 VMR 损失;不新增匹配相关损失。

# 消融矩阵(复现论文 Table 7,证明组合必要性)
| 配置 | -SA | +QD | NMS | 主看 |
|---|---|---|---|---|
| #0 当前 Idea1 基线 | - | - | - | active≈N, mR+@5≈1.1 |
| #1 仅 -SA | ✓ | - | ✓ | 论文:单独几乎无效 |
| #2 仅 +QD | - | ✓ | - | 论文:单独几乎无效 |
| #3 -SA+QD(正解) | ✓ | ✓ | ✓ | active 翻倍, mR+@5 显著↑ |
对照组(复现论文 Table 5/6,证明"改匹配"是错路,给 Idea2 盖棺):
| #4 +1-to-k 匹配 | 预期 %match GT 下降、mAP 崩 |
| #5 +Group/Hybrid 匹配 | 预期 active↑但冗余↑ |
另做 QD rate 消融 k∈{0.0,0.25,0.5} 复现 Table 8(0.5 应崩),query 总数消融复现 Table N(峰值约 20)。

# 验收指标(主指标 + 守护指标,缺一不可)
主指标(必须改善):
  - active query 数随 moment 数增长(诊断曲线从"水平线"变"上升线",对齐论文 Fig.5 橙线);
  - mR+@5 相对基线显著上升(论文多目标 mAPm 提升最高 +34.3%,你至少应看到 mR+@5 数倍增长,目标向官方天花板 19 靠拢);
  - 多目标后续命中率(当前 22.5%)明显提升。
守护指标(不得显著恶化,保住 Idea 1 成果):
  - mR@5 / mAP 不下降(论文显示单目标也小幅受益,至少持平);
  - Rej-F1 / Avg G-mIoU 维持在 Idea 1 水平(-SA+QD 不动拒答头,理应不变)。

# 硬约束与留痕
- 固定 seed,统一 epoch/早停;每组记录 config/ckpt/预测 jsonl/eval json/图,写 run.md(改了什么、为什么、命令、结果、PASS/FAIL、归因)。
- 主指标用官方 eval/ 工具,G-mIoU 报跨阈值 AP;NMS/QD 的阈值在 val 上选、test 不调参。
- QD rate 锁 0.25,严禁 0.5;严禁改 Hungarian 为一对多;-SA 只删 decoder 的 self-attn,不动 cross-attn。
- 资源不足时优先级:阶段一诊断 > #3 -SA+QD > 消融对照。未完成项明确标注。




代码参考要求：
明白,我来核对你代码库里 `models/moment_detr_gmr/` 的实际结构,定位 decoder self-attention 调用点、query embedding 注入点和 evaluate.py 的推理出口,这样给出的代码改法才能直接落地而不是凭空猜路径。我先抓取关键文件。

我注意到 `moment_detr.py` 里 import 的是 `moment_transformer`(不是 `transformer`),所以第一个链接抓空了。我再抓真正的 decoder 文件和推理出口,确保改法精确到行。

我已经把三个关键文件的实际实现都核对清楚了,路径和函数名全部对上,可以给你精确到行的改法了。下面每一处都标了文件、定位锚点和论文出处,执行 Agent 可以直接照着落地。

**先说清楚代码现状(三个落点已确认)**

你的 decoder 自注意力在 `models/moment_detr_gmr/moment_transformer.py` 的 `TransformerDecoderLayer.forward_post`(`normalize_before=False`,所以实际走的是 `forward_post`,不是 `forward_pre`);query embedding 在 `moment_detr.py` 的 `forward` 里通过 `self.query_embed.weight` 传进 transformer;双轨制推理出口在 `slot_existence_head.py` 的 `infer_with_slot_rejection`。三处刚好对应 -SA、QD、NMS。

**STEP I1:删 decoder self-attention(改 `moment_transformer.py`)**

定位 `TransformerDecoderLayer.forward_post`,论文式(3)→(4)就是把 `SA` 整块旁路。最安全的做法不是删代码,而是加一个开关 `self.use_sa`,关掉时跳过 self_attn 子层(连同其残差和 norm1),保留 cross-attention + FFN:

````python
# TransformerDecoderLayer.__init__ 末尾加一行(默认 True 保持原行为)
self.use_sa = True   # -SA 实验时置 False

# forward_post 改成:
def forward_post(self, tgt, memory, ...):
    if self.use_sa:
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
    # 删 SA 后直接进 cross-attention(对齐论文式4:FFN(CA(Q,M)))
    tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                               key=self.with_pos_embed(memory, pos),
                               value=memory, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = tgt + self.dropout3(tgt2)
    tgt = self.norm3(tgt)
    return tgt
````

**关键点**:`use_sa=False` 时 `query_pos` 仍要保留(它在 cross-attention 的 query 里用到,是位置先验,不能丢)。用开关而非删行,方便消融矩阵里 #1/#3 一键切换。同时在 `build_transformer` 里把开关透传:`Transformer(..., use_sa=getattr(args,"use_sa",True))`,再在 `Transformer.__init__` 构造 `decoder_layer` 后设 `decoder_layer.use_sa = use_sa`(因为 `_get_clones` 是 deepcopy,要在 clone 前设好)。

**STEP I3:query dropout k=0.25(改 `moment_detr.py`)**

定位 `MomentDETR.forward` 里这一行:

````python
hs, memory = self.transformer(src, ~mask, self.query_embed.weight, pos)
````

在它之前对 `query_embed.weight` 做 Bernoulli mask(论文式5,只训练期生效,keep 概率 `1-k`):

````python
# __init__ 里加:self.query_dropout = float(getattr(args, "query_dropout", 0.0))
q_weight = self.query_embed.weight
if self.training and self.query_dropout > 0:
    keep = 1.0 - self.query_dropout            # k=0.25 → keep=0.75
    mask_q = torch.bernoulli(
        torch.full((q_weight.size(0), 1), keep, device=q_weight.device))
    q_weight = q_weight * mask_q               # 整行置零,Q̂ = Q ⊙ M
hs, memory = self.transformer(src, ~mask, q_weight, pos)
````

**铁律**:`k` 锁 0.25,论文 Table 8 实测 0.5 会让 mAPm 从 23.9 崩到 3.8。`self.training` 保证推理时关闭。mask 按 query(行)级别置零,不是元素级。

**STEP I2:推理 NMS(改 `slot_existence_head.py` 的 `infer_with_slot_rejection`)**

你的双轨制现在是"空集全删 / 正样本按 `slot_fg_prob` 排序全保留"。删 SA 后会冒重叠框,需要在排序后、返回前插一道时序 NMS。在该函数 `else` 分支的 `order` 排序之后加:

````python
def _temporal_nms(spans, scores, iou_thr=0.7):
    # spans: [M,2] (cx,w) 或 (st,ed)，先转 (st,ed)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item(); keep.append(i)
        if order.numel() == 1: break
        rest = order[1:]
        # 计算 i 与 rest 的时序 IoU，去掉 >iou_thr 的
        iou = temporal_iou(spans[i], spans[rest])
        order = rest[iou <= iou_thr]
    return keep
````

接到 `sel`/`sc` 排序之后:`kept = _temporal_nms(sel, sc, iou_thr=0.7)`,再用 `kept` 索引输出。**NMS 只去重不砍分数阈值**,空集判定仍由前面 `n_fg[b] < min_fg` 的 Noisy-OR 负责,这样不破坏你 Idea 1 守住的 mAP/mR。`iou_thr=0.7` 是起点,需在验证集微调、test 不调。

**STEP I4:不动 matcher(铁律)**

`moment_detr.py` 的 `SetCriterion` 用的 `self.matcher(outputs_without_aux, targets)` 就是 Hungarian 一对一,**一行都不要改**。论文 Table 6 证明 -SA+QD 必须搭配 1-to-1 才能维持 query 多样性;换 1-to-k 会让 mAPm 腰斩——正是你 Idea 2 mAP 崩到 1.67 的同款病理。消融矩阵里的 #4/#5(改匹配对照组)只是为了复现论文负面结论、给 Idea 2 盖棺,跑完即弃,不进主线。

**STEP D1/D2:active query 统计脚本(阶段一诊断,零训练)**

加载当前 Idea 1 ckpt,在 test 集前向,统计每个查询的 active query 数和 GT moment 数,画论文 Fig.5 曲线。骨架:

````python
@torch.no_grad()
def diagnose_active_queries(model, loader, thr=0.05, device="cuda"):
    import numpy as np
    rows = []  # (n_gt_moment, n_active)
    model.eval()
    for batch, targets in loader:
        out = model(**batch)
        prob = out["slot_fg_prob"]          # [B,N] 用前景概率判 active
        n_active = (prob > thr).sum(1).cpu().numpy()
        for b in range(prob.size(0)):
            n_gt = len(targets["span_labels"][b].get("spans", []))
            if n_gt > 0:                    # 阶段一只看正样本
                rows.append((n_gt, int(n_active[b])))
    rows = np.array(rows)
    # 按 GT moment 数分桶(1,2,3,4+),求平均 active 数 ± std → 画曲线
    for g in [1, 2, 3]:
        m = rows[rows[:,0]==g]
        if len(m): print(f"GT={g}: active={m[:,1].mean():.2f}±{m[:,1].std():.2f}")
    m4 = rows[rows[:,0]>=4]
    if len(m4): print(f"GT>=4: active={m4[:,1].mean():.2f}±{m4[:,1].std():.2f}")
    np.save("experiments/diag/active_vs_moment.npy", rows)
````

**用 `slot_fg_prob>0.05` 判 active**(对齐论文 Fig.L 的 IOU≥0.1 口径,阈值要在 `diagnosis.md` 注明并做敏感性检查)。预期基线呈水平线 ≈4,不随 GT moment 增长——这就是你复现的 active decoder-query collapse,也是下一篇报告里最值钱的 motivation 图。

**落地顺序与隔离**

把上述改动复制到 `experiments/{日期}_{阶段}_{轮次}/` 下的副本运行,不动仓库公共代码。`use_sa`、`query_dropout`、NMS `iou_thr` 全部走 args 配置,这样消融矩阵 #0(全关)、#1(仅 -SA)、#2(仅 QD)、#3(-SA+QD)、#4/#5(改匹配对照)只需切 config 不改代码。每组记录 config/ckpt/预测/指标/曲线图,写 `run.md`。

**一处要特别提醒的坑**:删 SA(`use_sa=False`)后理论上 `tgt` 的初始值是全零(`tgt = torch.zeros_like(query_embed)`),第一层 cross-attention 的 query 完全靠 `query_pos`(即被 QD mask 过的 `query_embed`)提供区分度——这正是 QD 必须和 -SA 同时上的底层原因:删了 SA 之后 query 之间不再"商量",而 QD 强迫监督分散到更多 query 索引,两者缺一不可(论文 Table 7 单独上几乎无效)。所以消融里 #1 仅 -SA、#2 仅 QD 大概率不动甚至略降,#3 组合才会让 active query 翻倍、mR+@5 起飞,这个结果方向如果复现出来,本身就是对论文的有力验证。

需要的话,我可以接着把 `temporal_iou`(支持 cx-w 与 st-ed 转换)和 `build_transformer`/args 透传 `use_sa`、`query_dropout` 的完整 diff 也写全,让 Agent 直接 patch。


下面把缺的两块补全:`temporal_iou` 工具函数,以及 `use_sa`/`query_dropout` 从 args 一路透传到 decoder layer 的完整 diff。全部对齐你仓库现有的函数名和调用链,Agent 可以直接 patch。

**补丁 A:`moment_transformer.py` —— `use_sa` 透传到每个 decoder layer**

关键陷阱是 `_get_clones` 用的是 `copy.deepcopy`,所以必须在 clone **之前**把 `use_sa` 写进 `decoder_layer`,否则只改到模板那一份、6 层 clone 全是默认值。

````python
# ---- Transformer.__init__ 改 decoder 构造段 ----
def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
             num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
             activation="relu", normalize_before=False,
             return_intermediate_dec=False, use_sa=True):   # ← 新增 use_sa
    super().__init__()
    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    decoder_layer.use_sa = use_sa          # ← 必须在 _get_clones(deepcopy) 之前设
    decoder_norm = nn.LayerNorm(d_model)
    self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                      return_intermediate=return_intermediate_dec)
    self._reset_parameters()
    self.d_model = d_model
    self.nhead = nhead
````

```python
# ---- TransformerDecoderLayer.__init__ 末尾加一行(给 deepcopy 一个默认值兜底)----
        self.normalize_before = normalize_before
        self.use_sa = True          # ← 新增;被 Transformer.__init__ 覆盖
```

`forward_post` 的改法上一轮已给(用 `if self.use_sa:` 包住 self_attn + dropout1 + norm1 三行)。这里补 `forward_pre` 的对称改法,虽然你 `normalize_before=False` 走不到,但留着防止以后切 pre-norm 时静默失配:

````python
# ---- TransformerDecoderLayer.forward_pre 头部 ----
def forward_pre(self, tgt, memory, ...):
    if self.use_sa:
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
    tgt2 = self.norm2(tgt)
    tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                               key=self.with_pos_embed(memory, pos),
                               value=memory, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
    tgt = tgt + self.dropout2(tgt2)
    tgt2 = self.norm3(tgt)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
    tgt = tgt + self.dropout3(tgt2)
    return tgt
````

````python
# ---- build_transformer 透传 use_sa ----
def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=False,
        return_intermediate_dec=True,
        use_sa=bool(getattr(args, "use_sa", True)),   # ← 默认 True 保持原行为
    )
````

**补丁 B:`moment_detr.py` —— query dropout 接 args**

`MomentDETR.__init__` 加字段,`forward` 改 query 注入(上一轮已给),`build_model` 透传:

````python
# ---- MomentDETR.__init__ 形参与字段 ----
def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
             num_queries, input_dropout, aux_loss=False, max_v_l=75, span_loss_type="l1",
             use_txt_pos=False, n_input_proj=2, aud_dim=0, use_exist_head=False,
             exist_pool="max", query_dropout=0.0):          # ← 新增
    ...
    self.query_dropout = float(query_dropout)               # ← 新增字段
````

````python
# ---- build_model 透传 query_dropout(放在 MomentDETR(...) 构造里)----
    model = MomentDETR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        aud_dim=args.a_feat_dim if "a_feat_dim" in args else 0,
        aux_loss=args.aux_loss,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        span_loss_type=args.span_loss_type,
        n_input_proj=args.n_input_proj,
        use_exist_head=bool(getattr(args, "use_exist_head", False)),
        exist_pool=str(getattr(args, "exist_pool", "max")),
        query_dropout=float(getattr(args, "query_dropout", 0.0)),   # ← 新增
    )
````

`forward` 里的 Bernoulli mask 段(上一轮已给)保持不变,锁 `query_dropout=0.25`。

**补丁 C:`temporal_iou` —— NMS 用的时序 IoU(1 对多)**

你的 `pred_spans` 是 `(cx, w)` 归一化格式,仓库已有 `span_cxw_to_xx`(在 `utils/span_utils.py`)。为避免循环依赖,直接在 `slot_existence_head.py` 内部写一个轻量版,输入统一先转 `(st, ed)`:

````python
def _cxw_to_stae(x):
    """[*,2] (cx,w) → (st,ed);已是 (st,ed) 时传 is_cxw=False 跳过。"""
    st = x[..., 0] - x[..., 1] / 2.0
    ed = x[..., 0] + x[..., 1] / 2.0
    return torch.stack([st, ed], dim=-1)

def temporal_iou(box, others, is_cxw=True):
    """box: [2];others: [M,2]。返回 [M] 的 1-vs-M 时序 IoU。"""
    if is_cxw:
        box = _cxw_to_stae(box.unsqueeze(0)).squeeze(0)
        others = _cxw_to_stae(others)
    s1, e1 = box[0], box[1]
    s2, e2 = others[:, 0], others[:, 1]
    inter = (torch.min(e1, e2) - torch.max(s1, s2)).clamp(min=0)
    union = (e1 - s1).clamp(min=0) + (e2 - s2).clamp(min=0) - inter
    return inter / union.clamp(min=1e-6)
````

配合上一轮的 `_temporal_nms`,把它的 `temporal_iou(spans[i], spans[rest])` 调用对齐成 `temporal_iou(spans[i], spans[rest], is_cxw=True)`。

**补丁 D:把 NMS 接进 `infer_with_slot_rejection`**

完整替换该函数的 `else` 分支,保证"空集判定走 Noisy-OR、去重走 NMS"两条逻辑互不干扰:

````python
@torch.no_grad()
def infer_with_slot_rejection(slot_fg_prob, spans, tau_slot=0.5, min_fg=1,
                              use_nms=False, nms_thr=0.7):   # ← 新增两参
    fg_mask = slot_fg_prob > tau_slot
    n_fg = fg_mask.sum(1)
    preds = []
    for b in range(spans.size(0)):
        if n_fg[b] < min_fg:
            preds.append([])                       # Noisy-OR 拒答,空集
        else:
            sel = spans[b][fg_mask[b]]             # [M,2] (cx,w)
            sc = slot_fg_prob[b][fg_mask[b]]       # [M]
            order = torch.argsort(sc, descending=True)
            sel, sc = sel[order], sc[order]
            if use_nms and sel.size(0) > 1:        # 删SA后去重叠框
                kept = _temporal_nms(sel, sc, iou_thr=nms_thr)
                sel, sc = sel[kept], sc[kept]
            preds.append(torch.cat([sel, sc[:, None]], dim=1).tolist())
    return preds
````

````python
def _temporal_nms(spans, scores, iou_thr=0.7):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        iou = temporal_iou(spans[i], spans[rest], is_cxw=True)
        order = rest[iou <= iou_thr]
    return torch.tensor(keep, dtype=torch.long, device=spans.device)
````

**消融矩阵对应的 config 开关一览**

| 实验 | `use_sa` | `query_dropout` | `use_nms` | matcher |
|---|---|---|---|---|
| #0 基线 | True | 0.0 | False | Hungarian |
| #1 仅 -SA | False | 0.0 | False | Hungarian |
| #2 仅 QD | True | 0.25 | False | Hungarian |
| #3 -SA+QD | False | 0.25 | True | Hungarian |
| #4 改匹配(证伪) | False | 0.25 | True | 1-to-k |

`use_nms` 只在 #3/#4 这种会冒重叠框的组合里开;#0\~#2 关掉 NMS 是为了让"删 SA / 加 QD 单独的净效应"不被后处理掩盖,这样 Table 才干净。NMS `nms_thr` 在验证集上扫 `{0.5,0.6,0.7,0.8}`,test 锁定不再动。

**最后一道自检**:patch 完先跑一次 `use_sa=True, query_dropout=0.0, use_nms=False`,确认指标与当前 Idea 1 封板值(Avg G-mIoU@1 36.97 / mR@5 10.27 / mAP 7.11)逐位对齐——只有这个"零改动等价性"通过,才能证明开关接线没有引入副作用,后面 #1\~#4 的对比才有意义。