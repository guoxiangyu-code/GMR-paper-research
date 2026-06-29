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
