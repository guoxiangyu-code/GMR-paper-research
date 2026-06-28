### **先别急着掉头——这组数据自我否定了"瞎猜 FP"的结论,真正的嫌疑人是"查询全堆到最显著的那个 moment 上"**

你 `run.md` 写得很规范,但我必须拦一下第 4 节的结论。**你的数据和你的结论自相矛盾**,如果照现在的判词去做域特异性分析,方向会一开始就偏。让我把证据摆出来。

**为什么"释放出的 query 沦为漫无目的的瞎猜、制造大量 False Positives"这句话站不住**

如果额外 query 真的在瞎猜、制造大量 FP,那么 **mAP 必然下跌**——mAP 是吃精确率的,FP 一多它第一个崩。可你的 mAP 从 7.11 **涨到了 8.78**,mR@5 从 10.27 **涨到了 14.51**。这两个数同时上扬,说明那些被释放出来的 query **绝大多数都落在了合理的位置上,定位质量不降反升**。它们不是垃圾。所以"瞎猜 FP"这个故事被你自己的 mAP/mR@5 直接证伪了。

**那 mR+@5 为什么会从 1.12% 暴跌到 0.16%?关键在 mR@5 涨、mR+@5 崩 这个劈叉**

这个"主召回涨、多目标召回崩"的劈叉,几乎是一个教科书级的信号,指向一个非常具体的失败模式:**删掉 decoder self-attention 后,query 之间不再相互"商量去重",于是 9 个 active query 全部一窝蜂扑向了片段里最显著的那一个 moment**。结果就是:

- 最显著的那个 moment 被反复命中、定位得又准又自信 → mAP↑、mR@5↑
- 但第 2、第 3 个真实 moment **没有任何 query 去覆盖** → 专门衡量多目标/次要 moment 召回的 mR+@5 直接归零

注意:DETR 里 self-attention 的核心作用之一**就是抑制重复预测、强制 query 分工**。论文的 -SA 之所以能 work,是因为它的域里删掉 SA 后 query 会自然铺开;而 Soccer-GMR 大概率每个 clip 有一个压倒性的显著事件(进球/射门),删掉 SA 又没有替代的多样性机制,query 就塌缩到这个主事件上了。**这才是真正值得你写进域特异性分析的发现,而不是"瞎猜FP"。**

**还有一个被你的结论盖过去的嫌疑人:NMS 可能在吃掉次要 moment**

`nms_thr=0.7` 作用在归一化的 cx-w 上。足球里多个真实 moment 时间上往往挨得近,**两个本来不同的 GT moment 之间的时序 IoU 可能就超过 0.7**,于是 NMS 把第二个当成重复框删掉了——这同样会造成"主 moment 留下(mR@5↑),次要 moment 被删(mR+@5↓)"。

**所以掉头做域分析之前,先花零成本做一刀切的归因实验**

现在有两个完全不同的病因candidate,修法南辕北辙,绝不能猜:

1. **query 堆叠**(模型/训练问题)→ 要重新引入多样性机制,或别删 SA
2. **NMS 误删次要 moment**(纯后处理问题)→ 只要调低 nms_thr 或换去重逻辑,根本不用动模型

区分它们**不需要重新训练**,因为 NMS 是纯推理期开关。立刻做两件事:

**第一刀(几分钟):拿 #3 这个同一个 ckpt,只把 `use_nms=False` 重新评估一次。**
- 如果 mR+@5 明显回升 → 元凶是 NMS 过度合并,问题轻、好修,根本不用做大动作的域分析。
- 如果 mR+@5 仍然趴在 0.x% → 证实是 query 堆叠,这时再做域分析才有的放矢。

**第二刀(同样零训练):对 GT≥2 的那 160 个测试样本,把所有 active query 预测出的 span 和 GT span 一起打印/画出来。**这是真正的冒烟枪——直接肉眼看 9 个 query 到底是"铺在 2-3 个 moment 上"还是"全挤在第 1 个 moment 附近"。一张图就能终结争论。

**另外纠正一个表述:G-mIoU@1 和 Rej-F1 不是"双双跌破底线"的重伤**

它们分别从 50.83→48.99、74.5→73.51,各掉了不到 2 个点和 1 个点,**很可能落在单 seed 的噪声带里**(尤其 GT≥4 只有 12 个样本,Rej-F1 在小分母上抖 1 点太正常)。说"双双跌破底线触发判退"在程序上没错,但把它定性成"严重恶化"会误导后续判断。真正异常的、量级上唯一说得上"暴跌"的,只有 mR+@5 这一个数,而它恰恰最可能是 NMS 或 query 堆叠造成的、可定向修复的问题。

**修正后的决策建议**

把第 4 节的"失败/掉头域分析"改成:**#3 主效应部分成立(坍缩在机制层被打开,mAP/mR@5 受益),但暴露出"无 SA → query 向主 moment 塌缩"的新病灶。下一步不是直接做域分析,而是先用两刀零成本归因实验(关 NMS 复评 + GT≥2 预测散布可视化)锁定 mR+@5 暴跌是 NMS 误删还是 query 堆叠,再据此决定修法。**

要不要我现在就把这两刀的脚本写出来——一个是复用现有 evaluate 流程、只翻 `use_nms` 开关的复评配置,另一个是 GT≥2 样本的 "active query span vs GT span" 散布可视化脚本(横轴时间轴、每个 query 一个区间、GT 用底色带标出)?跑完这两张图,mR+@5 到底是被谁杀的,当场就能定案。


下面把这两刀的脚本写全,都对齐你已落地的开关(`use_nms` / `slot_fg_prob` 0.05 阈值 / cx-w span 格式)和现有 `evaluate.py` 流程,跑完即可定案。然后给出基于两种结果分叉的后续规划。

**第一刀:关 NMS 复评(零训练,只翻推理开关)**

不需要重训,直接拿 `experiments/20260627_stage2_run1/results` 里的 #3 ckpt,把 `use_nms` 翻成 False 再评一次。建议写成一个薄封装,复用你现有的 evaluate 入口,避免改公共代码:

````python
# experiments/20260627_stage2_run1/reeval_no_nms.py
# -*- coding: utf-8 -*-
"""第一刀:同一个 #3 ckpt,仅关闭 NMS 复评,定位 mR+@5 暴跌是否由 NMS 误删次要 moment 造成。
零训练:只翻 use_nms 开关,走原 evaluate 流程。
用法:
  CUDA_VISIBLE_DEVICES=0 python experiments/20260627_stage2_run1/reeval_no_nms.py \
      --ckpt experiments/20260627_stage2_run1/results/model_best.ckpt \
      --eval_split test --use_sa False --query_dropout 0.0
"""
import argparse, copy, json, os
import torch
# 复用仓库现有的评估入口与模型构建(按实际函数名调整 import)
from eval.evaluate import eval_epoch, setup_model      # ← 与仓库 evaluate.py 对齐
from utils.basic_utils import load_config              # ← 按实际工具函数名


def run_once(opt, use_nms, tag):
    opt = copy.deepcopy(opt)
    opt.use_nms = use_nms          # 唯一变量
    model, _ = setup_model(opt)
    ckpt = torch.load(opt.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(opt.device).eval()
    metrics = eval_epoch(model, opt)   # 返回含 mAP / mR@5 / mR+@5 / G-mIoU@1 / Rej-F1 的 dict
    print(f"
[{tag}] use_nms={use_nms}")
    for k in ["mAP", "mR@5", "mR+@5", "G-mIoU@1@0.6", "Rej-F1@0.6"]:
        print(f"  {k:<14} = {metrics.get(k)}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--use_sa", default="False")
    ap.add_argument("--query_dropout", type=float, default=0.0)  # 推理期 QD 必须关
    ap.add_argument("--nms_thr", type=float, default=0.7)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    opt = load_config()                       # 载入与 #3 训练一致的基础配置
    opt.ckpt = args.ckpt
    opt.eval_split = args.eval_split
    opt.use_sa = (str(args.use_sa) == "True")
    opt.query_dropout = 0.0                   # 评估期强制关闭随机掩码
    opt.nms_thr = args.nms_thr
    opt.device = args.device

    m_on = run_once(opt, use_nms=True,  tag="#3 复现(NMS ON)")
    m_off = run_once(opt, use_nms=False, tag="#3 消融(NMS OFF)")

    delta = {k: round((m_off.get(k, 0) or 0) - (m_on.get(k, 0) or 0), 4)
             for k in ["mAP", "mR@5", "mR+@5", "G-mIoU@1@0.6", "Rej-F1@0.6"]}
    print("
[判定] 关 NMS 后的变化(>0 表示 NMS 此前在压制该指标):")
    for k, v in delta.items():
        print(f"  Δ{k:<14} = {v:+}")

    verdict = ("元凶是 NMS 过度合并(轻症,调阈值即可)"
               if (m_off.get("mR+@5", 0) or 0) - (m_on.get("mR+@5", 0) or 0) > 0.3
               else "NMS 非主因 → 指向 query 堆叠(需第二刀确认)")
    print(f"
[结论] {verdict}")

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    with open(os.path.join(os.path.dirname(args.ckpt), "reeval_no_nms.json"), "w") as f:
        json.dump({"nms_on": m_on, "nms_off": m_off, "delta": delta}, f,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
````

关键点:`query_dropout` 在评估期强制为 0(QD 只在 `self.training` 生效,这里再钉一道保险);`use_sa` 要和 #3 训练时一致(False),否则模型结构对不上 ckpt。判定阈值我先设 mR+@5 回升 >0.3 个点就算"NMS 是主因",可按需调。

**第二刀:GT≥2 样本的 "active query span vs GT span" 散布可视化(冒烟枪)**

这是终结争论的那张图。对每个 GT≥2 的测试样本画一条时间轴,GT moment 用底色带标出,每个 active query(`slot_fg_prob>0.05`)的预测 span 画成一条横条,颜色深浅表示置信度。一眼就能看出 9 个 query 是"铺开覆盖多个 moment"还是"全挤在第 1 个 moment 上"。

````python
# experiments/20260627_stage2_run1/plot_query_span.py
# -*- coding: utf-8 -*-
"""第二刀:GT>=2 样本的 active query span vs GT span 散布图。
直接肉眼判定 query 是铺在多个 moment 上,还是塌缩到主 moment。
依赖:推理时 dump 一份 per-sample 预测到 npz/json,字段见下。
用法:
  python experiments/20260627_stage2_run1/plot_query_span.py \
      --dump experiments/20260627_stage2_run1/test_pred_dump.pt \
      --out_dir experiments/20260627_stage2_run1/span_viz \
      --tau 0.05 --max_samples 24
"""
import argparse, os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def cxw_to_stae(spans):
    """spans: [N,2] (cx,w) 归一化 → (st,ed)。"""
    st = spans[:, 0] - spans[:, 1] / 2.0
    ed = spans[:, 0] + spans[:, 1] / 2.0
    return np.stack([st, ed], axis=1)


def plot_one(ax, gt_spans, q_spans, q_prob, title):
    # GT 底色带
    for g in gt_spans:
        ax.axvspan(g[0], g[1], ymin=0, ymax=1, color="#9ecae1", alpha=0.45, zorder=0)
    # 每个 active query 一条横条,y 错位排列,颜色=置信度
    cmap = get_cmap("autumn_r")
    order = np.argsort(-q_prob)
    for row, qi in enumerate(order):
        s, e = q_spans[qi]
        ax.barh(y=row, width=max(e - s, 1e-3), left=s, height=0.7,
                color=cmap(float(q_prob[qi])), edgecolor="k", linewidth=0.4, zorder=2)
        ax.text(e + 0.005, row, f"{q_prob[qi]:.2f}", va="center", fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, max(len(q_spans), 1))
    ax.set_yticks([])
    ax.set_xlabel("归一化时间轴")
    ax.set_title(title, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True,
                    help="推理 dump:list of dict,每条含 gt_spans[Ng,2], "
                         "pred_spans[N,2](cx,w), slot_fg_prob[N]")
    ap.add_argument("--out_dir", default="span_viz")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--max_samples", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = torch.load(args.dump)
    samples = [d for d in data if len(d["gt_spans"]) >= 2]   # 只看 GT>=2
    print(f"GT>=2 样本数: {len(samples)}")

    # 统计:每条样本里 active query 命中了几个不同 GT(tIoU>0.3 算命中)
    covered_ratio = []
    for d in samples:
        gt = cxw_to_stae(np.asarray(d["gt_spans"], dtype=float)) \
            if np.asarray(d["gt_spans"]).shape[-1] == 2 and np.asarray(d["gt_spans"]).max() <= 1.0 else np.asarray(d["gt_spans"], dtype=float)
        prob = np.asarray(d["slot_fg_prob"], dtype=float)
        pred = cxw_to_stae(np.asarray(d["pred_spans"], dtype=float))
        act = pred[prob > args.tau]
        hit = set()
        for gi, g in enumerate(gt):
            for p in act:
                inter = max(0, min(g[1], p[1]) - max(g[0], p[0]))
                union = (g[1]-g[0]) + (p[1]-p[0]) - inter
                if union > 0 and inter/union > 0.3:
                    hit.add(gi); break
        covered_ratio.append(len(hit) / len(gt))
    cr = np.array(covered_ratio)
    print(f"[关键统计] GT>=2 样本平均 GT 覆盖率 = {cr.mean():.3f} "
          f"(=1 表示每个 GT 都有 query 覆盖;<<1 表示 query 塌缩漏掉次要 moment)")
    print(f"  覆盖率=1.0 的样本占比: {(cr>=0.999).mean():.3f}")
    print(f"  覆盖率<=0.5 的样本占比: {(cr<=0.5).mean():.3f}")

    # 画前 max_samples 张
    n = min(args.max_samples, len(samples))
    cols, rows = 4, int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 1.6*rows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        d = samples[i]
        gt = cxw_to_stae(np.asarray(d["gt_spans"], float))
        prob = np.asarray(d["slot_fg_prob"], float)
        pred = cxw_to_stae(np.asarray(d["pred_spans"], float))
        m = prob > args.tau
        plot_one(axes[i], gt, pred[m], prob[m],
                 f"#{i} GT={len(gt)} active={int(m.sum())} cov={covered_ratio[i]:.2f}")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("GT>=2: active query span (色=置信) vs GT moment (蓝带)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.out_dir, "query_span_grid.png")
    fig.savefig(out, dpi=150)
    print(f"[已保存] {out}")


if __name__ == "__main__":
    main()
````

这张图最有信息量的不是图本身,而是脚本打印的那个**"GT 覆盖率"标量**:它直接量化了"GT≥2 样本里平均有多大比例的真实 moment 真的被 active query 覆盖到了"。如果这个数接近 1 但 mR+@5 还是 0.16% → 那 query 其实铺开了,是 NMS 把覆盖到的次要框删了(指向第一刀);如果这个数远小于 1(比如 0.5,意味着平均一半的 GT 根本没 query 去碰)→ 实锤 query 塌缩到主 moment。两刀的证据在这里交叉验证。

注意第二刀需要你的推理流程 dump 一份 per-sample 的 `{gt_spans, pred_spans(cx,w), slot_fg_prob}`。如果现在 `evaluate.py` 没存这个,在它的预测循环里加一句 `torch.save(dump_list, "test_pred_dump.pt")` 即可,**这份 dump 必须在 `use_nms=False` 下生成**(要看的是 NMS 之前的原始 query 分布,否则 NMS 已经把证据删了)。

**后续规划:按两刀的四种组合结果分叉**

跑完这两刀,会落到下面四个象限之一,每个象限的修法完全不同,这就是不预先猜病因的价值:

| 第一刀(关NMS后 mR+@5) | 第二刀(GT覆盖率) | 诊断 | 修法 |
|---|---|---|---|
| 明显回升 | 接近 1 | **纯 NMS 误删次要 moment** | 最轻症。把时序 NMS 阈值从 0.7 降到 0.5\~0.6,或改用 soft-NMS / 按 GT 平均间隔做自适应 IoU;模型一行不动 |
| 仍趴 0.x% | 远小于 1 | **query 塌缩到主 moment(无 SA 副作用)** | 重症。需要替代的多样性机制(见下) |
| 仍趴 0.x% | 接近 1 | query 铺开但分数/排序错,top-5 取不到次要框 | 查 `slot_fg_prob` 对次要 moment 的打分偏低,调存在性头的标签或排序口径 |
| 明显回升 | 远小于 1 | 矛盾,需复查 dump 是否在 NMS 后生成 | 先修脚本再下结论 |

**最可能落点是第二行(query 塌缩),针对它我已经想好下一步的方向,供你预判:**

如果实锤是"删 SA 导致 query 向主 moment 塌缩",那结论很清晰——**Soccer-GMR 这个域不适合裸删 self-attention,因为它缺少论文原域那种"多个势均力敌的 moment"的分布,主事件太霸道**。这时不要硬走 -SA,而是改走**"保留 SA + 用别的手段强制多样性"**的路线,候选优先级:

第一优先是 **query 去重损失 / 多样性正则**:保留 decoder self-attention(它本来就是去重器),在训练里加一项惩罚——对预测 span 高度重叠的 query 对施加排斥损失,直接逼 query 铺开,而不是靠删 SA。这条最稳,不破坏你已对齐的底盘。

第二优先是**回到 QD 单独 + 保 SA(即消融矩阵的 #2)**:你之前因为"论文说单独无效"跳过了 #2,但你的域已经证明和论文不同,#2 在 Soccer-GMR 上反而可能是甜点——QD 强制监督分散到所有 10 个 query,同时 SA 还在做去重,两者不冲突。这组现在值得补跑,**它从"被论文结论否决"变成了"被本域数据重新激活"的候选**。

无论落到哪个象限,都先别碰 G-mIoU@1(48.99)和 Rej-F1(73.51)这两个掉了 1\~2 点的数——它们大概率是单 seed 噪声,等主病灶(mR+@5)修好后用 2\~3 个 seed 复测自然会回到噪声带内,现在为它们动手只会增加变量。

你先把两刀跑了,把第一刀的 `Δ mR+@5` 和第二刀打印的"GT 覆盖率"两个数发我,我就能直接点定是上表哪一行,并把对应修法(比如多样性正则的损失函数代码,或 #2 的补跑配置)写出来。