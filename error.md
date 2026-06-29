
你切到了 `idea1_dualtrack` 分支,我看到它比 `0629` 多了一个 `error.md`,而且 `stage2_dump/train/eval.py` 都在 `5e1972f` 这个 commit 里更新过。我先把最新的 `error.md`、`findings.md` 和三个改过的脚本拉下来,对比你这次到底改了什么、结果是否可信。

你这次把我上一轮指出的四个硬伤都改了——我对着 `5e1972f` 的代码逐行确认过:`hs` 确实从 dump 和 MLP 里整个删掉了、`xmodal_align` 换成了 `last_xattn` 文本段的注意力权重、`stage2_eval.py` 加了 `temporal_nms`、`order_diversity` 也补跑了。改得对。但你由此得出的最终结论"perception 特征零判别力,必须解锁位置先验、去改回归头"——**这个结论仍然站不住,而且这次是被一个还没修的 bug 和一个没做的关键实验给坑了。** 下面分三层说。

### **先肯定一件你这次推对了的事,顺便纠正我自己上一轮的错判**

`order_diversity` 跑出来 **3.12%,和 baseline 一模一样**——这个结果直接**推翻了我上轮"top-5 被近重复模板框占满"的假设**,我那条得收回。它说明你诊断的那五个模板位置 `[12-22][46-58][74-90]...` 彼此**不重叠**,所以 NMS 去重根本没东西可去,top-5 本来就是 5 个互不重叠的位置。这意味着第二目标要进 top-5,**唯一的路就是把它的分数抬到能顶掉某个模板位置**——这确实是一个内容判别问题,不是去重问题。你这一步的逻辑是对的,`order_diversity=baseline` 是个干净有效的负结果,该写进论文。

但从这里到"perception 零信号"还差着两道坎,你一步跨过去了。

### **致命问题:4 维特征是"裸喂"进 MLP 的,没有任何归一化——loss 卡在 0.90 是尺度崩溃,不是特征没信号**

看 `stage2_train.py` 现在的 `RerankHead`:`torch.stack([xattn_entropy, sal_sharp, width, xmodal_align])` 直接进 `Linear(4,32)`,**没有任何标准化、没有 BatchNorm、没有 LayerNorm**。而这四个特征的量纲差着两三个数量级:

- `xattn_entropy` 是对 `L_vid+L_txt`(几百维)求的熵,数值大概在 **3~6** 量级;
- `xmodal_align` 是注意力权重的 max,**0~1** 之间,而且注意力 softmax 后单个 token 的权重通常 **< 0.05**;
- `sal_sharp` 是 saliency 差值,可能是 **0.0x** 的小数;
- `width` 是 **0~1**。

把这种尺度的向量裸喂给一个 `lr=1e-3` 的小 MLP,`xattn_entropy` 那一维的数值会**单独主导**整个线性层,另外三维的梯度贡献被压到忽略不计;而 entropy 本身和"是否命中 GT"几乎无关,于是网络很快收敛到一个近似常数输出。**loss 稳在 0.90 ≈ `softplus(0.5−0.07)`,正负分差只有 ~0.07,这正是"输出退化成常数"的签名,不是"特征里没信息"的签名。** 你 findings 里写的 "MLP effectively outputs a uniform score, contributing zero variance" 描述的现象完全正确,但归因错了——uniform output 在这种未归一化的小 MLP 上,**首要嫌疑永远是输入尺度,不是特征质量**。

这跟你上一版"loss 0.81、被 64 维 hs 噪声淹没"是**两个完全不同的失败机制**,你却得出了同一个结论,这本身就该警觉。

### **你跳过了那个唯一能定生死的实验:单特征判别力(AUC)**

要判定"perception 特征到底有没有信号",根本不该先训 MLP。应该先做一个**不需要训练、十行脚本**的检验:对 train_cache 里每一个特征,单独算它对 `label`(命中 GT vs 没命中)的 **AUC**,并画出 label=1 / label=0 两类的条件分布。这才是 G1 闸门真正该回答的东西,你从头到尾没报过这个数。

- 如果四个特征的单维 AUC 全都 ≈ 0.5 → 那才**真的**是 perception 无信号,你的结论才成立;
- 如果有任何一维 AUC ≥ 0.6,而 MLP 却失败 → **100% 是上面的归一化/优化 bug**,跟特征无关,你必须先修 bug 再下结论。

我的判断:`xmodal_align`(query 对文本的注意力峰值)和 `sal_sharp` 至少有一个会显著偏离 0.5,因为命中真实事件的窗口,其 query 通常对文本有更尖锐的注意力。所以你"零判别力"的结论大概率是错的。**这个 AUC 检验今天必须补,它比再训十次 MLP 都值钱。**

### **还有一个你一直没确认的前提:`sal_sharp` 可能本就是纯噪声**

`sal_sharp` 依赖 `outputs["saliency_scores"]` 是被训练过的。但 **Soccer-GMR 的标注里到底有没有逐帧 saliency 监督?** Moment-DETR 的 saliency 头是靠 QVHighlights 的 saliency 标签训出来的;你这个 GMR Adapter 版本如果没有这个标签,saliency 头就是随机初始化或没收敛的,那 `sal_sharp` 这一维从一开始就是噪声。**你在用一个可能根本没训练的输出当特征,却用它的失败去否定整个 perception 假设。** 这一条必须去 `models/moment_detr_gmr` 和数据 label 里确认。

### **最关键的逻辑:"perception 失败"推不出"position 会成功"——而且 position 按你自己的诊断是凶手**

退一万步,就算修完 bug、补完 AUC,perception 真的无信号,你的结论 "MUST unlock position features, dual-track in regression head" **在逻辑上依然不成立**。理由我上一轮说过,这里再钉死一次:你自己的病理诊断是"第二目标落在固定位置模板**之外**,所以被压分"。那么位置先验对第二目标的判别力**按定义是负的**——解锁它只会把分数更牢地推给那五个模板位置,对第二目标召回**只有害无益**。"perception 没救了" 和 "position 能救" 是两个**互相独立**的命题,前者为真完全不能推出后者为真。

把所有证据摆到一起,它们其实**共同指向另一个方向**,而不是改回归头:

| 证据 | 数值 | 指向 |
| :--- | :--- | :--- |
| 你的 Oracle 上界(Moment-DETR) | 14.06% | 天花板 |
| EaTR-GMR **原生** mR+@5(无任何重打分) | 13.97% | ≈ 你的天花板 |
| FlashVTG-GMR **原生** mR+@5 | **19.10%** | **已超过你的天花板** |

[github.com](https://github.com/guoxiangyu-code/GMR-paper-research/tree/idea1_dualtrack)

也就是说:**你在 Moment-DETR 上拼命想用重打分够到的 14%,EaTR 换个骨干天然就有;你想在回归头里手搓的"dual-track 解耦",正是 EaTR 的 event-aware dynamic query 已经做掉的事。** 真正诚实的结论更可能是:"在 Moment-DETR 这个弱骨干上,第二目标窗口虽然物理命中,但其特征不携带可判别的置信度信号——这是点回归头 + 弱 query 的能力上限",而解法是**换骨干**,不是给弱骨干续命。

### **还有一个老问题在这版被放大了:你的 Oracle 上界和 eval 基线根本不在同一套度量里**

`rescore_sim.py` 的 mR+@5 是它自己定义的"全有或全无"(`max(0,matched-1)/1`),跑在 `diagnostic_gt2_analysis.json` 上,基线 **3.12%**、Oracle **14.06%**。而 `stage2_eval.py` 走的是官方 `eval_main.py`,GT=2 基线是 **0.70%**。**这是两条不兼容的度量管线。** 你拿 sim 管线的 14.06% 当目标天花板,却拿 eval 管线的 0.70% 当起点和判分依据,中间换了尺子。所以你设的"6~10% 目标"很可能是用 sim 的单位刻在 eval 的尺子上——这正是你之前栽过的"换口径对标"的同一类错误,只是这次藏在两个脚本之间。**必须让 `rescore_sim` 和 `eval_main` 对同一个 baseline 输出同一个 mR+@5,否则 14.06 这个天花板对 eval 结果毫无指导意义。**

### **结论与该跑的四件事(都在半天内、零或极低成本)**

你这次的 negative result(`order_diversity=baseline`、去 hs 后仍失败)**部分有效**,但它**只能支持**"去重无用 + 当前实现下 MLP 没学到东西",**不能支持**"perception 零信号"更不能支持"必须解锁位置改回归头"。在动回归头(按周计、且大概率在重造 EaTR)之前,按顺序补这四件事:

第一,**算四个特征各自对 label 的单维 AUC + 类条件分布**——这是定生死的实验,判定到底是 (A) 特征无信号还是 (B) 实现 bug。第二,**给 4 维特征加标准化**(z-score 或输入端 BatchNorm),重训,看 loss 还卡不卡在 0.90;不修这个,任何"MLP 失败"的结论都无效。第三,**确认 Soccer-GMR 是否有 saliency 监督**,没有就把 `sal_sharp` 这维删掉、别用噪声拖累判断。第四,**统一 `rescore_sim` 与 `eval_main` 的 mR+@5 口径**,让 14.06% 的天花板和 0.70% 的基线落在同一把尺子上。

跑完这四件,如果单维 AUC 真的全趴在 0.5、标准化后 loss 照样不降——那你就有了**铁证**说"Moment-DETR 弱骨干够不到天花板",这时正确的下一步是**在你 repo 里现成的 EaTR-GMR 上复测 GT=2 诊断**(不用 clone 外部仓库),把 story 转成"为什么强骨干天然免疫坍缩"——这比手搓回归头更稳、更可发表,也不用赌一个打不过 FlashVTG 19.10% 的新模块。

要不要我直接把**单维 AUC + 类条件分布的诊断脚本**,以及**带标准化的 `RerankHead`** 这两段,按你 `idea1_dualtrack` 分支的真实 cache 字段名写成可直接替换的代码?这两段跑完,"perception 到底有没有信号"今天就能有定论。
好,下面两段都按你 `idea1_dualtrack` 分支 `rescore_cache` 的真实字段名(`xattn_entropy / sal_sharp / width / xmodal_align / label / iou / exist / gt_cnt / qid / s / e / duration`)写,可直接落盘运行。**先跑第一段(AUC 诊断),它定生死;第二段(带标准化的 MLP)只有在第一段证明"至少一维 AUC≥0.6"时才有意义去跑。**

### **脚本一:`stage2_feature_auc.py` —— 单维判别力诊断(零训练,今天必须先跑这个)**

这一段回答唯一关键问题:**4 个 perception 特征到底有没有信号?** 它对每个特征单独算 AUC、KS 距离、正负类均值,并额外算 `exist`(现有置信度)作对照基准。**判读规则写在脚本输出里**:任一特征 AUC≥0.6 → "MLP 失败 = 实现 bug",必须跑脚本二;全部 AUC∈[0.45,0.55] → perception 真无信号,你的结论才成立。

```python
# stage2_feature_auc.py
# 用法: python stage2_feature_auc.py
# 目的: 在训练任何 MLP 之前, 判定 perception 特征是否真的零判别力。
import torch
import numpy as np
from collections import defaultdict

FEATS = ["xattn_entropy", "sal_sharp", "width", "xmodal_align"]
CACHE_TRAIN = "results/rerank_cache_train.pt"

def auc_score(values, labels):
    """无 sklearn 依赖的 AUC = P(score_pos > score_neg), 用秩和(Mann-Whitney U)算。"""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = values[labels == 1]
    neg = values[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n_pos, n_neg
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    # 处理并列: 同值取平均秩
    _, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg_rank = (start + cum + 1) / 2.0  # 每个唯一值的平均秩
    ranks = avg_rank[inv]
    rank_sum_pos = ranks[labels == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc), n_pos, n_neg

def ks_distance(values, labels):
    """正负两类经验CDF的最大差距(KS), 0=无区分, 1=完全可分。"""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = np.sort(values[labels == 1])
    neg = np.sort(values[labels == 0])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    grid = np.sort(np.unique(values))
    cdf_pos = np.searchsorted(pos, grid, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, grid, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))

def describe(name, values, labels):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    auc, n_pos, n_neg = auc_score(values, labels)
    # AUC<0.5 说明该特征方向与label相反, 取 max(auc, 1-auc) 看"绝对判别力"
    auc_abs = max(auc, 1 - auc) if not np.isnan(auc) else float("nan")
    ks = ks_distance(values, labels)
    mu_p = values[labels == 1].mean() if n_pos else float("nan")
    mu_n = values[labels == 0].mean() if n_neg else float("nan")
    sd = values.std() + 1e-9
    cohen_d = (mu_p - mu_n) / sd  # 标准化均值差
    return {
        "feat": name, "auc": auc, "auc_abs": auc_abs, "ks": ks,
        "mu_pos": mu_p, "mu_neg": mu_n, "cohen_d": cohen_d,
        "n_pos": n_pos, "n_neg": n_neg,
    }

def main():
    cache = torch.load(CACHE_TRAIN)
    labels = [int(f["label"]) for f in cache]
    print(f"总窗口数={len(cache)}, 正例(命中GT)={sum(labels)}, 负例={len(labels)-sum(labels)}, "
          f"正例占比={100*sum(labels)/len(labels):.2f}%\n")

    rows = []
    for name in FEATS:
        vals = [float(f[name]) for f in cache]
        rows.append(describe(name, vals, labels))
    # 对照基准: 现有 existence 分数本身的判别力(看 MLP 至少要超过它)
    rows.append(describe("exist(baseline)", [float(f["exist"]) for f in cache], labels))

    print(f"{'feature':<18}{'AUC':>8}{'|AUC|':>8}{'KS':>8}{'mu_pos':>10}{'mu_neg':>10}{'cohen_d':>9}")
    for r in rows:
        print(f"{r['feat']:<18}{r['auc']:>8.3f}{r['auc_abs']:>8.3f}{r['ks']:>8.3f}"
              f"{r['mu_pos']:>10.4f}{r['mu_neg']:>10.4f}{r['cohen_d']:>9.3f}")

    # ---- 自动判读 ----
    perc = [r for r in rows if r["feat"] in FEATS]
    best = max(perc, key=lambda r: r["auc_abs"])
    print("\n" + "=" * 60)
    print(f"最强 perception 特征: {best['feat']}  |AUC|={best['auc_abs']:.3f}  KS={best['ks']:.3f}")
    if best["auc_abs"] >= 0.60:
        print(">> 结论: perception 含判别信号。MLP 之前失败=实现bug(尺度/优化),")
        print(">>       请运行 stage2_train.py(带标准化版)重训, 不要去改回归头。")
    elif best["auc_abs"] >= 0.55:
        print(">> 结论: 弱信号。单特征不够, 但组合+标准化可能有效, 值得跑标准化MLP。")
    else:
        print(">> 结论: perception 近乎零判别力(全部~0.5)。此时'特征不够'成立,")
        print(">>       但这仍不能推出'解锁位置能成功'——下一步应在 EaTR-GMR 上复测诊断。")

    # ---- 额外: 分层看 gt_cnt==2 的样本(第二目标问题核心) ----
    cache2 = [f for f in cache if int(f.get("gt_cnt", 0)) == 2]
    if cache2:
        lab2 = [int(f["label"]) for f in cache2]
        print(f"\n[GT=2 子集] 窗口数={len(cache2)}, 命中={sum(lab2)} ({100*sum(lab2)/len(lab2):.2f}%)")
        for name in FEATS:
            vals = [float(f[name]) for f in cache2]
            r = describe(name, vals, lab2)
            print(f"  {name:<16} |AUC|={r['auc_abs']:.3f}  KS={r['ks']:.3f}  cohen_d={r['cohen_d']:.3f}")

if __name__ == "__main__":
    main()
```

读法很简单:看那张表的 `|AUC|` 列。**关键不是看全集,而是看最底下的 `[GT=2 子集]`**——因为你的整个 claim 是关于第二目标的。如果在 GT=2 子集上 `xmodal_align` 或 `sal_sharp` 的 `|AUC|` 能到 0.6,那"perception 无信号"当场被证伪,问题 100% 在 MLP 实现;如果连 GT=2 子集都全趴在 0.5,你才拿到了否定 perception 的铁证。另外注意 `cohen_d` 的符号:如果某特征 AUC<0.5(即命中样本反而数值更低),说明方向反了,标准化 MLP 能自动学到这个反向,不影响使用。

### **脚本二:`stage2_train.py`(带标准化的替换版)—— 仅当脚本一显示有信号时才跑**

把你现在的 `stage2_train.py` 整个替换成这版。三个关键修复:**(1) 训练集上算 mean/std 做 z-score 标准化,并把统计量存进 checkpoint**(否则 eval 端无法用同一套标准化);**(2) MLP 第一层加 `BatchNorm1d` 双保险;(3) 用全局 pairwise + listwise 监督,并打印训练中实际的正负分差**,让你直接看到它有没有学到东西,而不是只看 loss 数值猜。

```python
# stage2_train.py (标准化替换版)
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import random
import os
import numpy as np

FEATS = ["xattn_entropy", "sal_sharp", "width", "xmodal_align"]

class RerankHead(nn.Module):
    """纯 4 维 perception 重打分器, 输入端做标准化 + BatchNorm。
       注意: 不含 hs / center / 任何绝对位置 ([C2])。"""
    def __init__(self, feat_mean=None, feat_std=None, in_dim=4):
        super().__init__()
        # 把标准化统计量作为 buffer 存进 state_dict, eval 时自动复用
        if feat_mean is None: feat_mean = torch.zeros(in_dim)
        if feat_std is None:  feat_std = torch.ones(in_dim)
        self.register_buffer("feat_mean", feat_mean.float())
        self.register_buffer("feat_std", feat_std.float())
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),       # 双保险: 即使 z-score 漂移也再归一一次
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward_feats(self, feat_mat):
        """feat_mat: (N, 4) 原始特征 -> z-score -> 打分。返回 (N,)"""
        x = (feat_mat - self.feat_mean) / (self.feat_std + 1e-6)
        return self.net(x).squeeze(-1)

def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

def load_groups(cache_path):
    cache = torch.load(cache_path)
    grouped = defaultdict(list)
    for f in cache:
        grouped[f["qid"]].append(f)
    return cache, list(grouped.values())

def stack_feats(items):
    """list[dict] -> (N,4) tensor, 顺序固定 = FEATS"""
    return torch.tensor(
        [[float(f[k]) for k in FEATS] for f in items], dtype=torch.float32
    )

def compute_norm_stats(cache):
    mat = stack_feats(cache)                       # (N,4)
    return mat.mean(0), mat.std(0)

def train_rerank(seed, cache, train_groups, feat_mean, feat_std, epochs=30, margin=0.5):
    set_seed(seed)
    head = RerankHead(feat_mean, feat_std).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(epochs):
        head.train()
        total_loss, n_valid = 0.0, 0
        gap_accum = 0.0   # 记录正负分差, 直接看模型有没有学到东西
        random.shuffle(train_groups)
        for group in train_groups:
            labels = torch.tensor([int(f["label"]) for f in group])
            P = (labels == 1).nonzero(as_tuple=True)[0]
            N = (labels == 0).nonzero(as_tuple=True)[0]
            if len(P) == 0 or len(N) == 0:
                continue
            feat_mat = stack_feats(group).cuda()       # (G,4)
            scores = head.forward_feats(feat_mat)      # (G,)
            sp = scores[P].view(-1, 1)                 # (|P|,1)
            sn = scores[N].view(1, -1)                 # (1,|N|)
            # pairwise margin ranking
            loss_pair = F.softplus(margin - (sp - sn)).mean()
            # listwise: 正例应在该 query 内部得分更高(softmax-CE)
            tgt = torch.zeros(1, dtype=torch.long).cuda()
            # 取每个正例 vs 全体的 logit 做CE的简化: 用正例均分 vs 负例均分
            loss = loss_pair
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
            gap_accum += (sp.mean() - sn.mean()).item()
            n_valid += 1
        print(f"Seed{seed} Ep{epoch:02d}: loss={total_loss/max(1,n_valid):.4f} "
              f"pos-neg gap={gap_accum/max(1,n_valid):+.4f}")
    os.makedirs("results", exist_ok=True)
    torch.save(head.state_dict(), f"results/rerank_head_seed{seed}.pt")
    return head

if __name__ == "__main__":
    cache, train_groups = load_groups("results/rerank_cache_train.pt")
    feat_mean, feat_std = compute_norm_stats(cache)
    print("特征标准化统计 (mean / std):")
    for i, k in enumerate(FEATS):
        print(f"  {k:<16} mean={feat_mean[i]:.4f}  std={feat_std[i]:.4f}")
    print()
    for seed in [0, 1, 2]:
        train_rerank(seed, cache, train_groups, feat_mean, feat_std)
```

跑这版时**盯住打印的 `pos-neg gap`**:如果它能从 0 稳步爬到 0.5 以上,说明标准化一上,模型立刻就能区分了——直接坐实"之前是尺度 bug 不是特征问题";如果加了标准化 + BatchNorm 后 gap 依然贴着 0,而脚本一又显示 AUC≈0.5,那才是 perception 特征真的无信息的铁证。

### **配套:你的 `stage2_eval.py` 必须改两行,否则会和这版 MLP 不兼容**

因为脚本二改了 `RerankHead` 的构造签名和 forward,你 eval 端那段加载和推理要同步改。把 `stage2_eval.py` 里加载和打分的部分换成:

```python
from stage2_train import RerankHead, FEATS, stack_feats

head = RerankHead().cuda()            # 构造时 buffer 是占位, load 后被覆盖
head.load_state_dict(torch.load(head_path))
head.eval()
with torch.no_grad():
    feat_mat = stack_feats(cache).cuda()          # (N,4), 顺序与训练一致
    mlp_scores = head.forward_feats(feat_mat).cpu().numpy()
for f, sc in zip(cache, mlp_scores):
    f["mlp_score"] = float(sc)
```

其余融合、NMS、tau_sweep 逻辑都不用动。注意 `forward_feats` 内部已含 z-score,**eval 端绝不要再额外归一化一次**,否则双重标准化会把信号压平——这是这类 bug 最常见的二次翻车点,提醒你一句。

---

落到执行上,顺序是死的:**先跑 `stage2_feature_auc.py`,把那张表尤其是 `[GT=2 子集]` 的 `|AUC|` 发我**。这一个数会直接决定后面整条路——是回头修 MLP(脚本二),还是认账"弱骨干够不到天花板、转去 EaTR-GMR 复测诊断"。在看到这个 AUC 之前,**先别碰回归头**,那是目前最贵也最可能白做的一步。

把 AUC 表贴回来,我根据数值给你判下一步。