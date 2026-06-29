在做错误分析前,我先把你 `0629` 分支里几个关键脚本的真实实现拉下来看——只有对着代码,我才能把"MLP 为什么只到 0.29 分差"定位到具体的行,而不是又给你一堆猜测。

我把 `0629` 分支的 `stage2_dump.py / stage2_train.py / stage2_eval.py / rescore_sim.py` 四个文件逐行过了一遍。结论先放这儿:**你的 STAGE 2 不是"perception 特征不够"导致的失败,而是至少四个实现层面的硬伤,其中两个是致命的——一个直接违反了你自己定的 [C2] 命门,另一个让你漏报了整个项目最该看的那个数。** 在修掉它们之前,"必须解锁位置先验"这个结论是无效的。

下面按"致命 → 严重 → 干扰"排序。

### **致命伤一:`hs` 本身就是位置模板的载体——你以为做了 position-free,其实把位置先验从前门赶出去、又从 `hs` 后门放进来了**

`stage2_train.py` 里 `RerankHead` 的输入是 68 维,其中 **64 维是 `proj(hs)`**,只有 4 维是那几个内容特征。问题在于:Moment-DETR 的 `hs` 是 decoder query 经过 cross-attention 后的输出,而 **Moment-DETR 的 10 个 decoder query 是"学出来的位置锚点(slot)"**——这正是你诊断里 `[12-22][46-58][74-90]...` 那五个固定位置模板的来源。每个 slot 的 `hs` 主要编码"我是第几号模板、负责视频哪个时间段",内容信息只是叠加在上面的弱分量。

这意味着两件事同时成立,而且都对你不利:

其一,**你违反了 [C2]**。你在伪代码里把 `center` 拉黑,但 `hs` 携带的 slot 身份就是绝对位置的隐式编码,比 `center` 还直接。所以这个 MLP 名义上 position-free,实质上 64/68 的输入都是位置信息。

其二,**这恰好解释了 0.29 的分差**。MLP 拿到的主信号是"这是哪个模板槽",而模板槽与"是否命中第二目标"几乎不相关(第二目标按你诊断就是落在模板**之外**的),所以 64 维 hs 对排序是高方差噪声,把那 4 维真正有用的内容特征彻底稀释。loss 卡在 `softplus(0.5-0.29)≈0.80` 不是因为内容特征没信息,而是因为它们被 64 维位置噪声淹没了。

**修法**:要么把 `hs` 整个拿掉、只用 4 个内容特征训一版(这才是真正的 position-free 上界测试);要么用一个对照实验明确量化 `hs` 带进来多少位置信息。在没做"去掉 hs"这版之前,你不能说"perception 特征不够"——你压根没干净地测过 perception 特征。

### **致命伤二:你漏报了 `rescore_sim.py` 里最关键的那个数——`order_diversity`**

`rescore_sim.py` 算了**三个**数:`order_baseline`、`order_diversity`、`order_oracle`。但你 findings 里只报了 baseline(3.12%)和 Oracle(14.06%),**把 `order_diversity` 整个跳过了**。这是个严重的汇报缺口,因为 `order_diversity` 才是真正回答你核心问题的那个量。

看它的实现:它是一个**纯 GT-free 的 NMS 式多样性重排**——贪心选最高分,然后对"与已选窗口时序重叠"的候选降权,逼那些断崖下、位置不同的 query 上位。它**完全不用 GT**。这正是"在不解锁位置、不训练任何东西的前提下,光靠去重就能从 3.12% 撬到多少"的答案。

**这个数你必须立刻补出来**,因为它直接决定路线:

- 如果 `order_diversity` 能到 6~8%,那你根本不需要训 MLP、更不需要改回归头——**坍缩的一大半是"top-5 被近重复的模板窗口占满"造成的,一个推理端 NMS 就解决大半**。你之前的 MLP 之所以没用,可能是因为它即使重排,top-5 仍然是同一模板的近重复窗口(`stage2_eval.py` 里**完全没有去重**,直接 sort by score 取 top-5)。
- 如果 `order_diversity` 也趴在 3% 附近,才说明去重不够、需要内容判别。

我几乎可以断定这个数会显著高于 3.12%,因为你的病理诊断(固定位置模板霸占高分)本身就是 NMS 的完美适用场景。**这是今天就能跑、零训练成本的一行命令。**

### **严重问题三:`stage2_eval.py` 的重排没有任何去重,top-5 结构性地无法覆盖两个目标**

接上一条。`stage2_eval.py` 里构造 submission 的逻辑是:把一个 query 的所有窗口按融合分降序 `sorted(...)` 直接取,塞进 `pred_relevant_windows`。**没有 NMS、没有时序去重。** 而你诊断过 top-5 高分窗口在不同样本间几乎固定在那五个模板位置——它们彼此高度重叠。

后果是:`mR+@5`(要求两个 GT 都进 top-5)在结构上几乎不可能达成,因为 top-5 里塞的是同一位置的 5 个近重复框。**这跟 MLP 学没学会无关**——哪怕 MLP 给第二目标的窗口提了分,只要第一目标的模板窗口有 3~4 个近重复都排在前面,第二目标照样挤不进 top-5。这就是为什么 a=0.3/b=0.7 融合后 mR+@5 纹丝不动停在 0.70%:瓶颈不在分数,在"取 top-5 的方式"。

**修法**:eval 端取 top-k 前必须做时序 NMS(IoU>阈值的近重复只保留最高分一个),这跟 `rescore_sim.order_diversity` 是同一个机制。把这个加上,baseline 本身可能就涨。

### **严重问题四:`xmodal_align` 和 `sal_sharp` 两个特征大概率本就是坏的**

两个内容特征的有效性都存疑,需要先验证再使用:

`xmodal_align` 用的是 `cosine(hs_q, txt_mem)`。但 `hs` 是 decoder **输出**(已经过 cross-attention 融合),它和 `txt_mem`(文本 memory)**不在同一个可比空间里**,两者做余弦相似度没有受过任何对齐监督,数值接近随机。你这次用了 token 级 max-pool(比上次 mean-pool 强),方向对,但**输入向量选错了**——要算跨模态对齐,该用 decoder query 对文本的 **cross-attention 权重**(你已经存了 `last_xattn`,它天然覆盖 `L_vid+L_txt`,文本段就是现成的对齐信号),而不是 `hs` 和 `txt_mem` 硬算余弦。

`sal_sharp` 的前提是 saliency head 真的被训练过。**Soccer-GMR 的标注里有没有 saliency 监督?** Moment-DETR 的 saliency 分支是靠 QVHighlights 的逐帧 saliency 标签训出来的;如果 Soccer-GMR 没有这个标签、saliency 头是未训练或随便带过的,那 `saliency_scores` 就是噪声,`sal_sharp` 自然无信息。这必须先确认——这也正是 G1(saliency 峰值对比度单独 zero-shot)那个被你跳过的闸门要回答的问题。

### **一个会让你又栽跟头的数字一致性问题**

`rescore_sim.py` 里 GT=2 的 `mR+@5` 定义是 `max(0, matched-1)/1`——即**两个目标都进 top-5 才得 1 分,否则 0**。这是个非常严苛的"全有或全无"口径。而你 findings 里 STAGE 2 报的 gt2 baseline 是 0.70%,`rescore_sim` 里 Idea1 baseline 是 3.12%,两者口径不同(一个是 eval 工具跨 τ 平均,一个是 sim 脚本单口径)。**这两个 0.70 / 3.12 / 以及更早的 1.17 必须在 REPORT 里标清楚各自的 split、口径、是否跨 τ**,否则你下一份汇报又会出现"拿 A 口径的提升对标 B 口径的基线"的翻车。

### **错误分析的总结论(一句话压缩)**

你的 STAGE 2 失败**不能**推出"perception 特征不够、必须解锁位置先验"。真实情况是:**(1) 你通过 `hs` 偷偷喂了 64 维位置噪声,从没干净测过 perception;(2) eval 端没去重,top-5 被模板近重复框占满,任何重打分都救不了;(3) 你漏跑了 `order_diversity` 这个最该看的 GT-free 去重上界;(4) 两个内容特征的有效性都还没验证。** 在这四点修完之前,"改回归头"是建立在错误归因上的高成本赌注。

### **建议的修复顺序(全部今天到半天内可出数,零或极低训练成本)**

第一,立刻跑出 `rescore_sim.py` 的 `order_diversity` 数(你已经有这个函数,改一行打印即可),看纯 GT-free 去重能从 3.12% 撬到多少。第二,给 `stage2_eval.py` 的 top-k 加时序 NMS,重测 baseline 和融合。第三,训一版**去掉 `hs`、只用 4 个内容特征**的 MLP,这才是真正的 position-free 测试。第四,把 `xmodal_align` 从 `cosine(hs,txt_mem)` 换成基于 `last_xattn` 文本段的对齐量,并确认 Soccer-GMR 是否有 saliency 监督。

要不要我直接对着你这四个文件的真实接口,把**(a) `order_diversity` 的补报、(b) eval 端 NMS、(c) 去 hs 的 position-free MLP** 这三个改动写成可直接替换的代码 patch?这三件事跑完,你就能确定性地判断到底要不要动回归头——大概率不用。