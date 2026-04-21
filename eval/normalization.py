# -*- coding: utf-8 -*-
"""
Soccer-GMR 评估用：Ground Truth 规范化。

支持两种标注形态：
  - 直接提供 relevant_windows: [[st, ed], ...]
  - moment.type == "clips" / "timestamps" 的原始结构（需配合 ts_cfg 展开时间戳窗）

本模块仅做数据结构与合法性整理，不包含任何指标计算。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple


def sanitize_windows(
    windows: Optional[List],
    duration: Optional[float] = None,
) -> List[List[float]]:
    """
    将 GT 时间窗列表清洗为合法、去重、按起点排序的 [st, ed] 列表。

    Args:
        windows: 原始窗列表。
        duration: 若提供，则将起止裁剪到 [0, duration]（时长未知则跳过裁剪）。
    """
    cleaned: List[List[float]] = []
    for w in windows or []:
        if w is None or len(w) != 2:
            continue
        try:
            st, ed = float(w[0]), float(w[1])
        except (TypeError, ValueError):
            continue
        if duration is not None:
            try:
                dur = float(duration)
                st, ed = max(0.0, st), min(dur, ed)
            except (TypeError, ValueError):
                pass
        if ed <= st:
            continue
        cleaned.append([st, ed])
    cleaned.sort(key=lambda x: (x[0], x[1]))
    deduped: List[List[float]] = []
    last: Optional[List[float]] = None
    for w in cleaned:
        if last is None or w[0] != last[0] or w[1] != last[1]:
            deduped.append(w)
        last = w
    return deduped


def load_ts_window_cfg(cfg_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """从 JSON 文件加载时间戳展宽规则；无路径则返回 None。"""
    if cfg_path is None:
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"ts_window_cfg must be a JSON object, got: {type(cfg)}")
    return cfg


def get_pre_post_by_query_type(
    ts_cfg: Optional[Dict[str, Any]],
    query_type: Any,
) -> Tuple[float, float]:
    """
    根据 query_type 解析 timestamps 展宽的前向/后向秒数。
    若未命中 by_query_type，则回退到 default；再缺则报错。
    """
    if ts_cfg is None:
        raise ValueError("GT 为 timestamps 类型但未提供 ts_cfg（需 --gt_ts_window_cfg）")
    default = ts_cfg.get("default", None)
    by_qt = ts_cfg.get("by_query_type", {}) or {}
    rule = by_qt.get(str(query_type), None) if query_type is not None else None
    if rule is None:
        rule = default
    if rule is None:
        raise ValueError(f"missing ts window rule for query_type={query_type}")
    return float(rule.get("pre", 6.0)), float(rule.get("post", 2.0))


def gt_record_to_relevant_windows(
    d: Dict[str, Any],
    ts_cfg: Optional[Dict[str, Any]],
) -> List[List[float]]:
    """
    将单条 GT 记录解析为 relevant_windows（秒级 [st, ed] 列表）。

    若顶层已有 relevant_windows 则直接使用（与 moment 字段互斥场景下优先前者）。
    """
    if "relevant_windows" in d:
        raw = d["relevant_windows"]
        return raw if isinstance(raw, list) else []

    moment = d.get("moment") or {}
    mtype = moment.get("type", None)
    value = moment.get("value", None)
    if mtype is None:
        return []

    if mtype == "clips":
        if value is None:
            return []
        # 单窗可能被写成二元组而非列表包一层
        if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(
            value[0], (int, float)
        ):
            return [list(value)]
        return list(value or [])

    if mtype == "timestamps":
        query_type = d.get("query_type", None)
        pre, post = get_pre_post_by_query_type(ts_cfg, query_type)
        windows: List[List[float]] = []
        for t in value or []:
            try:
                windows.append([float(t) - pre, float(t) + post])
            except (TypeError, ValueError):
                continue
        return windows

    raise ValueError(f"Unknown moment.type={mtype}")


def normalize_ground_truth(
    gt_raw: List[Dict[str, Any]],
    ts_cfg: Optional[Dict[str, Any]],
    *,
    drop_empty_gt: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    将原始 GT JSONL 行列表规范为 {qid, relevant_windows} 列表。

    Args:
        gt_raw: 原始记录列表。
        ts_cfg: 时间戳窗配置（与 load_ts_window_cfg 返回值一致）。
        drop_empty_gt: True 时丢弃无正例窗的样本（GMR 主评估通常保留空集，故调用方传入 False）。
    """
    stats = {"total": len(gt_raw), "kept": 0, "dropped_empty": 0, "dropped_invalid": 0}
    normalized: List[Dict[str, Any]] = []

    for d in gt_raw:
        if not isinstance(d, dict) or "qid" not in d:
            stats["dropped_invalid"] += 1
            continue
        windows = gt_record_to_relevant_windows(d, ts_cfg)
        windows = sanitize_windows(windows, duration=d.get("duration"))
        if drop_empty_gt and len(windows) == 0:
            stats["dropped_empty"] += 1
            continue
        normalized.append({"qid": d["qid"], "relevant_windows": windows})
        stats["kept"] += 1

    return normalized, stats
