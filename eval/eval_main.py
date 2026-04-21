# -*- coding: utf-8 -*-
"""
Soccer-GMR 评估唯一入口：读取预测与 GT JSONL，输出全部分数值（JSON）。

依赖同目录下的 normalization.py、metrics.py、utils.py。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from metrics import (
    DEFAULT_IOU_THRESHOLDS,
    compute_G_mIoU,
    compute_gmr_cls,
    compute_mAP,
    compute_mIoU,
    compute_mIoU_plus,
    compute_mR,
    compute_mR_plus,
    prepare_submission_for_gmiou,
)
from normalization import load_ts_window_cfg, normalize_ground_truth
from utils import load_jsonl


def evaluate_gmr(
    submission: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    *,
    k_list: Sequence[int] = (1, 3, 5),
    max_pred_windows: int = 10,
    cls_thresholds: Tuple[float, ...] = (0.4, 0.6),
    gmiou_cls_threshold: float = 0.4,
    iou_thds: np.ndarray = DEFAULT_IOU_THRESHOLDS,
    map_num_workers: int = 8,
    verbose: bool = True,
) -> "OrderedDict[str, Any]":
    """
    计算 GMR 全套指标：CLS、G-mIoU@k（k 与 k_list 一致）、
    以及正例上的 mAP / mR / mR+ / mIoU / mIoU+。
    """
    start = time.time()

    n_pos = sum(1 for d in ground_truth if len(d.get("relevant_windows", [])) > 0)
    n_multi = sum(1 for d in ground_truth if len(d.get("relevant_windows", [])) >= 2)
    n_neg = len(ground_truth) - n_pos

    results: "OrderedDict[str, Any]" = OrderedDict()
    brief: "OrderedDict[str, Any]" = OrderedDict()

    cls = compute_gmr_cls(submission, ground_truth, thresholds=cls_thresholds)
    brief["AUROC"] = cls["AUROC"]
    for thd_str, metrics in cls["per_threshold"].items():
        brief[f"Rej-F1@{thd_str}"] = metrics["Rej-F1"]
        brief[f"Acc@{thd_str}"] = metrics["Acc"]
    results["GMR-CLS"] = cls

    gated_sub, gmiou_gate = prepare_submission_for_gmiou(
        submission,
        cls_threshold=gmiou_cls_threshold,
        max_pred_windows=max_pred_windows,
    )
    gmiou_res = compute_G_mIoU(gated_sub, ground_truth, k_list=k_list)
    brief.update(gmiou_res)
    results["G-mIoU_gate"] = gmiou_gate
    results["G-mIoU_detail"] = gmiou_res

    pos_qids = {d["qid"] for d in ground_truth if len(d.get("relevant_windows", [])) > 0}
    gt_pos = [d for d in ground_truth if d["qid"] in pos_qids]
    sub_pos = [d for d in submission if d.get("qid") in pos_qids]

    if len(gt_pos) == 0:
        raise ValueError("无正例样本，无法计算定位类指标（mAP / mR / mIoU 等）。")

    map_res = compute_mAP(
        sub_pos,
        gt_pos,
        iou_thds=iou_thds,
        max_pred_windows=max_pred_windows,
        num_workers=map_num_workers,
    )
    m_r_res = compute_mR(sub_pos, gt_pos, k_list=k_list, iou_thds=iou_thds)
    m_r_plus_res = compute_mR_plus(sub_pos, gt_pos, k_list=k_list, iou_thds=iou_thds)
    miou_res = compute_mIoU(sub_pos, gt_pos, k_list=k_list)
    miou_plus_res = compute_mIoU_plus(sub_pos, gt_pos, k_list=k_list)

    brief["mAP"] = map_res["mAP"]
    for k in k_list:
        brief[f"mR@{k}"] = m_r_res[f"mR@{k}"]
    for k in k_list:
        brief[f"mR+@{k}"] = m_r_plus_res.get(f"mR+@{k}", 0.0)
    for k in k_list:
        brief[f"mIoU@{k}"] = miou_res[f"mIoU@{k}"]
    for k in k_list:
        brief[f"mIoU+@{k}"] = miou_plus_res.get(f"mIoU+@{k}", 0.0)

    results["brief"] = brief
    results["mAP_detail"] = map_res
    results["mR_detail"] = m_r_res
    results["mR+_detail"] = m_r_plus_res
    results["mIoU_detail"] = miou_res
    results["mIoU+_detail"] = miou_plus_res
    results["stats"] = {
        "num_total": len(ground_truth),
        "num_positive": n_pos,
        "num_negative": n_neg,
        "num_multi_instance": n_multi,
        "num_single_instance": n_pos - n_multi,
        "k_list": list(k_list),
        "cls_thresholds": list(cls_thresholds),
        "gmiou_cls_threshold": gmiou_cls_threshold,
        "eval_time_sec": round(time.time() - start, 2),
    }

    if verbose:
        print(
            f"[eval_main] {n_pos} positive ({n_pos - n_multi} single + {n_multi} multi), "
            f"{n_neg} negative, time={time.time() - start:.1f}s"
        )
        print(json.dumps(brief, indent=2, ensure_ascii=False))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Soccer-GMR Evaluation（GMR 全指标）")
    parser.add_argument("--submission_path", type=str, required=True, help="预测 JSONL")
    parser.add_argument("--gt_path", type=str, required=True, help="GT JSONL")
    parser.add_argument("--save_path", type=str, required=True, help="输出结果 JSON")
    parser.add_argument(
        "--gt_ts_window_cfg",
        type=str,
        default=None,
        help="若 GT 为 timestamps 形态，需提供时间窗展宽配置 JSON",
    )
    parser.add_argument(
        "--k_list",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="mR / mR+ / mIoU / mIoU+ / G-mIoU 的 k 列表（默认 1 3 5）",
    )
    parser.add_argument(
        "--max_pred_windows",
        type=int,
        default=10,
        help="mAP 与 G-mIoU 门控时最多保留的预测窗数量（默认 10）",
    )
    parser.add_argument(
        "--cls_thresholds",
        type=float,
        nargs="+",
        default=[0.4, 0.6],
        help="GMR-CLS 中 Rej-F1 / Acc 报告的阈值列表",
    )
    parser.add_argument(
        "--gmiou_cls_threshold",
        type=float,
        default=0.4,
        help="计算 G-mIoU@k 时用于门控存在分数的阈值 \\tau（默认 0.4）",
    )
    parser.add_argument(
        "--map_num_workers",
        type=int,
        default=8,
        help="mAP 多进程 worker 数（<=1 或样本少则自动单线程）",
    )
    parser.add_argument("--not_verbose", action="store_true", help="静默运行")
    args = parser.parse_args()

    verbose = not args.not_verbose

    submission = load_jsonl(args.submission_path)
    gt_raw = load_jsonl(args.gt_path)
    ts_cfg = load_ts_window_cfg(args.gt_ts_window_cfg)

    # GMR：保留 GT 中空集样本，用于 CLS 与 G-mIoU
    gt, gt_stats = normalize_ground_truth(gt_raw, ts_cfg, drop_empty_gt=False)

    pred_qids = {e["qid"] for e in submission if isinstance(e, dict) and "qid" in e}
    gt_qids = {e["qid"] for e in gt}
    shared = pred_qids & gt_qids

    submission = [e for e in submission if e.get("qid") in shared]
    gt = [e for e in gt if e.get("qid") in shared]

    if verbose:
        print(f"[eval_main] GT: {json.dumps(gt_stats, ensure_ascii=False)}")
        print(
            f"[eval_main] shared={len(shared)}, "
            f"gt_only={len(gt_qids - pred_qids)}, "
            f"pred_only={len(pred_qids - gt_qids)}"
        )

    if len(shared) == 0:
        raise ValueError("submission 与 GT 无交集 qid，无法评估。")

    results = evaluate_gmr(
        submission,
        gt,
        k_list=tuple(args.k_list),
        max_pred_windows=args.max_pred_windows,
        cls_thresholds=tuple(args.cls_thresholds),
        gmiou_cls_threshold=args.gmiou_cls_threshold,
        iou_thds=DEFAULT_IOU_THRESHOLDS,
        map_num_workers=args.map_num_workers,
        verbose=verbose,
    )

    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    if verbose:
        print(f"[eval_main] Saved -> {args.save_path}")


if __name__ == "__main__":
    main()
