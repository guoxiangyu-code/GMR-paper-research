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

@torch.no_grad()
def infer_with_slot_rejection(slot_fg_prob, spans, tau_slot=0.5, min_fg=1,
                              use_nms=False, nms_thr=0.7):
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
