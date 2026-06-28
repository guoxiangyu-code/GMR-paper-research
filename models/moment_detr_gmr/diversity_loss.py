# -*- coding: utf-8 -*-
"""阶段三:时序多样性/排斥损失,治 Soccer-GMR 的 query 引力坍缩。
作用:对"未匹配空闲 query"之间(及空闲 vs 匹配)预测 span 的过度时序重叠施加惩罚,
逼空闲 query 离开主事件——要么去覆盖次要 moment,要么退背景被拒答。
保护:被 Hungarian 匹配上的 query 之间不互相排斥(它们的间距由 GT 决定)。
"""
import torch
from models.moment_detr_gmr.utils.span_utils import (
    generalized_temporal_iou, span_cxw_to_xx)


def temporal_diversity_loss(pred_spans, matched_src_idx, iou_margin=0.5,
                            protect_matched=True, use_giou=False, eps=1e-6):
    """
    pred_spans:      [B, N, 2] (cx, w) 归一化预测 span(用 outputs["pred_spans"])
    matched_src_idx: List[Tensor],每个 batch 样本被 Hungarian 匹配的 query 下标
    iou_margin:      容忍阈值;tIoU 超过此值的部分才被惩罚(适度重叠不罚)
    protect_matched: True 时不惩罚 matched-matched 对
    返回标量 loss。
    """
    B, N, _ = pred_spans.shape
    device = pred_spans.device
    total = pred_spans.new_zeros(())
    n_pairs = 0
    for b in range(B):
        spans_xx = span_cxw_to_xx(pred_spans[b])           # [N,2] (st,ed)
        if use_giou:
            iou = generalized_temporal_iou(spans_xx, spans_xx)  # [N,N], 含负值
            iou = iou.clamp(min=0.0)
        else:
            iou = _pairwise_tiou(spans_xx)                  # [N,N] in [0,1]
        # 只罚超过 margin 的重叠部分
        over = (iou - iou_margin).clamp(min=0.0)            # [N,N]
        # 屏蔽自身对
        eye = torch.eye(N, device=device, dtype=torch.bool)
        over = over.masked_fill(eye, 0.0)
        # 构造 matched 掩码,保护 matched-matched 对
        if protect_matched and matched_src_idx[b].numel() > 0:
            m = torch.zeros(N, dtype=torch.bool, device=device)
            m[matched_src_idx[b]] = True
            mm = m[:, None] & m[None, :]                    # matched-matched
            over = over.masked_fill(mm, 0.0)
        total = total + over.sum()
        n_pairs += N * (N - 1)
    return total / max(n_pairs, 1)


def _pairwise_tiou(spans_xx, eps=1e-6):
    """spans_xx: [N,2] (st,ed) → [N,N] 时序 IoU。"""
    st, ed = spans_xx[:, 0], spans_xx[:, 1]
    inter = (torch.min(ed[:, None], ed[None, :])
             - torch.max(st[:, None], st[None, :])).clamp(min=0)
    len_i = (ed - st).clamp(min=0)
    union = len_i[:, None] + len_i[None, :] - inter
    return inter / union.clamp(min=eps)
