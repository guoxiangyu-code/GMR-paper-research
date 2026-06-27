# -*- coding: utf-8 -*-
"""Idea 1: 逐 slot NONE 前景分类 + 聚合式拒答。

借鉴 FUTR(arXiv:2205.14022)的 NONE 类与 δ 停止机制(式 17/18),
将样本级 max-pool 标量存在性,替换为 N 个 decoder slot 的前景计数投票。
可插入 models/moment_detr_gmr/moment_detr.py 的 decoder 输出之后。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers=2):
        super().__init__()
        dims = [in_dim] + [hid_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1])
                                    for i in range(num_layers))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class SlotExistenceHead(nn.Module):
    """每个 query slot 独立预测前景(=对应真实 moment)/ 背景(=NONE)。"""

    def __init__(self, d_model: int, hidden: int = None):
        super().__init__()
        hidden = hidden or d_model
        self.fg_head = MLP(d_model, hidden, 1, num_layers=2)

    def forward(self, hs: torch.Tensor):
        """hs: [B, N, D] 最后一层 decoder query 表征。"""
        slot_fg_logit = self.fg_head(hs).squeeze(-1)   # [B, N]
        slot_fg_prob = torch.sigmoid(slot_fg_logit)    # [B, N]
        return slot_fg_logit, slot_fg_prob


def aggregate_existence(slot_fg_prob: torch.Tensor, tau_slot: float = 0.5):
    """由 slot 概率聚合出样本级存在性。"""
    soft_count = slot_fg_prob.sum(dim=1)                          # [B] 期望前景数
    prob_any = 1.0 - torch.prod(1.0 - slot_fg_prob, dim=1)        # [B] P(∃前景)
    hard_count = (slot_fg_prob > tau_slot).sum(dim=1)             # [B]
    pred_exist = hard_count >= 1                                  # 计数投票
    return pred_exist, prob_any, soft_count


def build_slot_labels(slot_fg_logit, matched_slot_idx, gt_is_empty):
    """matched_slot_idx: List[Tensor],每个样本被匹配到 GT 的 slot 下标。"""
    label = torch.zeros_like(slot_fg_logit)            # [B, N] 默认背景
    for b, empty in enumerate(gt_is_empty):
        if not empty and len(matched_slot_idx[b]) > 0:
            label[b, matched_slot_idx[b]] = 1.0
    return label


def existence_loss(slot_fg_logit, slot_fg_prob, matched_slot_idx,
                   gt_is_empty, lambda_any: float = 1.0, eps: float = 1e-6):
    slot_label = build_slot_labels(slot_fg_logit, matched_slot_idx, gt_is_empty)
    l_slot = F.binary_cross_entropy_with_logits(slot_fg_logit, slot_label)

    y_exist = (~torch.as_tensor(gt_is_empty, device=slot_fg_logit.device)).float()
    prob_any = (1.0 - torch.prod(1.0 - slot_fg_prob, dim=1)).clamp(eps, 1 - eps)
    l_any = F.binary_cross_entropy(prob_any, y_exist)
    return l_slot + lambda_any * l_any, {"L_slot": l_slot.item(),
                                         "L_any": l_any.item()}


@torch.no_grad()
def infer_with_slot_rejection(slot_fg_prob, spans, tau_slot=0.5, min_fg=1):
    """spans: [B, N, 2];返回每个样本的预测窗口列表(空集即 [])。"""
    fg_mask = slot_fg_prob > tau_slot                  # [B, N]
    n_fg = fg_mask.sum(1)
    preds = []
    for b in range(spans.size(0)):
        if n_fg[b] < min_fg:
            preds.append([])                           # 所有 slot 判背景 → 空集
        else:
            sel = spans[b][fg_mask[b]]
            sc = slot_fg_prob[b][fg_mask[b]]
            order = torch.argsort(sc, descending=True)
            preds.append(torch.cat([sel[order], sc[order, None]], dim=1).tolist())
    return preds
