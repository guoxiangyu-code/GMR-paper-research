from __future__ import annotations


import argparse
import logging
import os
import pprint
import sys
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import BaseOptions
from dataset import StartEndDataset, prepare_batch_inputs, start_end_collate
from models.moment_detr_gmr.moment_detr import build_model as build_model_moment_detr
from models.moment_detr_gmr.gmr_adapter import apply_existence_gate
from models.moment_detr_gmr.utils.basic_utils import AverageMeter, save_json, save_jsonl
from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx
from models.moment_detr_gmr.slot_existence_head import infer_with_slot_rejection
from postprocessing import PostProcessorDETR
from standalone_eval.eval import eval_submission

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename):
    logger.info("Saving/evaluating predictions")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_jsonl(submission, submission_path)

    if opt.eval_split_name == "val":
        metrics = eval_submission(submission, gt_data)
        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path]

    return metrics, latest_file_paths

@torch.no_grad()
def compute_mr_results(epoch_i, model, eval_loader, opt, criterion=None):
    del epoch_i
    loss_meters = defaultdict(AverageMeter)
    mr_res = []

    for batch in tqdm(eval_loader, desc="compute moment scores"):
        query_meta = batch[0]
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)
        outputs = model(**model_inputs)

        pred_spans = outputs["pred_spans"].cpu()
        prob = F.softmax(outputs["pred_logits"], -1)
        scores = prob[..., 0].cpu()

        pred_exist_scores = None
        if "pred_exist_logits" in outputs:
            pred_exist_scores = torch.sigmoid(outputs["pred_exist_logits"]).detach().cpu()
            threshold = float(getattr(opt, "exist_gate_thd", 0.4))
            scores = apply_existence_gate(
                scores,
                pred_exist_scores,
                threshold,
                hard=getattr(opt, "hard_exist_gate", False),
            )

        # slot_fg_prob = outputs["slot_fg_prob"].cpu()
        # preds_slot_list = infer_with_slot_rejection(slot_fg_prob, pred_spans, tau_slot=0.5)

        for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans, scores)):
            if getattr(opt, "split", getattr(opt, "eval_split_name", "")) == "test" and not getattr(opt, "use_nms", False):
                if not hasattr(opt, "_test_pred_dump"):
                    opt._test_pred_dump = []
                # We will use meta["relevant_windows"] directly
                gt_spans = meta.get("relevant_windows", [])
                opt._test_pred_dump.append({
                    "gt_spans": gt_spans,
                    "duration": meta.get("duration", 150),
                    "pred_spans": spans.tolist(),
                    "slot_fg_prob": score.tolist()
                })

            if getattr(opt, "use_nms", False):
                from models.moment_detr_gmr.slot_existence_head import _temporal_nms
                kept = _temporal_nms(spans, score, iou_thr=getattr(opt, "nms_thr", 0.7))
                spans = spans[kept]
                score = score[kept]
            
            # Always output all N spans with their original scores
            spans_xx = span_cxw_to_xx(spans) * meta["duration"]
            cur_ranked_preds = torch.cat([spans_xx, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]

            cur_query_pred = {
                "qid": meta["qid"],
                "query": meta["query"],
                "vid": meta["vid"],
                "pred_relevant_windows": cur_ranked_preds,
            }
            if "pred_exist_score" in outputs:
                cur_query_pred["pred_exist_score"] = float(f"{float(outputs['pred_exist_score'][idx]):.4f}")
            elif pred_exist_scores is not None:
                cur_query_pred["pred_exist_score"] = float(f"{float(pred_exist_scores[idx]):.4f}")
            mr_res.append(cur_query_pred)

        if criterion is not None:
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_dict["loss_overall"] = float(losses)
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

    post_processor = PostProcessorDETR(
        clip_length=opt.clip_length,
        min_ts_val=0,
        max_ts_val=float(getattr(opt, "max_ts_val", 150)),
        min_w_l=1,
        max_w_l=float(getattr(opt, "max_ts_val", 150)),
        move_window_method="left",
        process_func_names=("clip_ts", "round_multiple"),
    )
    if hasattr(opt, "_test_pred_dump"):
        dump_path = os.path.join(getattr(opt, "results_dir", "."), "test_pred_dump.pt")
        torch.save(opt._test_pred_dump, dump_path)
        
    return post_processor(mr_res), loss_meters

def eval_epoch(epoch_i, model, eval_dataset, opt, save_submission_filename, criterion=None):
    logger.info("Generate submissions")
    model.eval()
    if criterion is not None:
        criterion.eval()

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
    )

    submission, eval_loss_meters = compute_mr_results(epoch_i, model, eval_loader, opt, criterion)
    metrics, latest_file_paths = eval_epoch_post_processing(
        submission,
        opt,
        eval_dataset.data,
        save_submission_filename,
    )
    return metrics, eval_loss_meters, latest_file_paths

def build_model(opt):
    return build_model_moment_detr(opt)

def setup_model(opt):
    logger.info("setup model/optimizer/scheduler")
    model, criterion = build_model(opt)

    if opt.device == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)

    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad]}],
        lr=opt.lr,
        weight_decay=opt.wd,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)
    return model, criterion, optimizer, lr_scheduler

def build_dataset_config(opt, data_path, load_labels):
    keep_empty_gt = bool(getattr(opt, "use_exist_head", False)) if load_labels else True
    return EasyDict(
        dset_name=opt.dset_name,
        domain=None,
        data_path=data_path,
        ctx_mode=opt.ctx_mode,
        v_feat_dirs=opt.v_feat_dirs,
        a_feat_dirs=None,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        v_feat_types=opt.v_feat_types,
        a_feat_types=None,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        max_a_l=opt.max_a_l,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=load_labels,
        mr_only=True,
        keep_empty_gt=keep_empty_gt,
    )

def start_inference(opt):
    logger.info("Setup config, data and model...")
    cudnn.benchmark = True
    cudnn.deterministic = False

    load_labels = opt.eval_split_name == "val"
    eval_dataset = StartEndDataset(**build_dataset_config(opt, opt.eval_path, load_labels=load_labels))

    checkpoint = torch.load(opt.model_path, weights_only=False)
    ckpt_state = checkpoint.get("model")
    if ckpt_state is None:
        raise ValueError(f"Checkpoint missing key 'model': {opt.model_path}")

    has_exist_head = any("exist_head" in k for k in ckpt_state.keys())
    if bool(getattr(opt, "use_exist_head", False)) != has_exist_head:
        logger.warning(
            "Config/checkpoint mismatch for existence head. config.use_exist_head=%s, ckpt_has_exist_head=%s. "
            "Using checkpoint setting.",
            bool(getattr(opt, "use_exist_head", False)),
            has_exist_head,
        )
        opt.use_exist_head = has_exist_head

    model, criterion, _, _ = setup_model(opt)
    model.load_state_dict(ckpt_state)
    logger.info("Model checkpoint: %s", opt.model_path)
    if not load_labels:
        criterion = None

    save_submission_filename = f"moment_detr_gmr_{opt.eval_split_name}_submission.jsonl"
    with torch.no_grad():
        metrics, _, _ = eval_epoch(None, model, eval_dataset, opt, save_submission_filename, criterion)

    if opt.eval_split_name == "val" and metrics is not None:
        logger.info("metrics_no_nms %s", pprint.pformat(metrics["brief"], indent=4))

def parse_args():
    parser = argparse.ArgumentParser(description="Run Moment-DETR-GMR inference on Soccer-GMR features.")
    parser.add_argument("--model", "-m", default="moment_detr", choices=["moment_detr"])
    parser.add_argument("--dataset", "-d", default="soccer_gmr", choices=["soccer_gmr"])
    parser.add_argument("--feature", "-f", default="clip_slowfast", choices=["clip_slowfast"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["val", "test"])
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--t_feat_dir", type=str, default=None)
    parser.add_argument("--v_feat_dirs", type=str, nargs="+", default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hard_exist_gate", action="store_true", help="Zero all window scores when p_exist is below threshold.")
    parser.add_argument("--use_nms", action="store_true")
    parser.add_argument("--nms_thr", type=float, default=0.7)
    parser.add_argument("--use_sa", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--query_dropout", type=float, default=0.0)
    parser.add_argument("--use_diversity", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--div_coef", type=float, default=0.5)
    parser.add_argument("--div_margin", type=float, default=0.5)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    option_manager = BaseOptions(args.model, args.dataset, args.feature, resume=None)
    option_manager.parse()
    opt = option_manager.option

    for name in ["model_path", "eval_path", "t_feat_dir", "results_dir", "device", "hard_exist_gate", "use_nms", "nms_thr", "use_sa", "query_dropout", "use_diversity", "div_coef", "div_margin"]:
        value = getattr(args, name)
        if value is not None:
            setattr(opt, name, value)
    if args.v_feat_dirs is not None:
        opt.v_feat_dirs = args.v_feat_dirs
    if args.results_dir is not None:
        os.makedirs(opt.results_dir, exist_ok=True)
    else:
        os.makedirs(opt.results_dir, exist_ok=True)

    opt.model_path = args.model_path
    opt.eval_split_name = args.split
    start_inference(opt)
