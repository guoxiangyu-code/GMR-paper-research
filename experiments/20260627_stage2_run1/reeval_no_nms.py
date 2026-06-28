# -*- coding: utf-8 -*-
"""第一刀:同一个 #3 ckpt,仅关闭 NMS 复评,定位 mR+@5 暴跌是否由 NMS 误删次要 moment 造成。
零训练:只翻 use_nms 开关,走原 evaluate 流程。
用法:
  CUDA_VISIBLE_DEVICES=0 python experiments/20260627_stage2_run1/reeval_no_nms.py \
      --ckpt experiments/20260627_stage2_run1/results/best.ckpt \
      --eval_split test --use_sa False --query_dropout 0.0
"""
import argparse, copy, json, os, sys
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "training/moment_detr_gmr"))
from training.moment_detr_gmr.evaluate import eval_epoch, setup_model
from training.moment_detr_gmr.train import BaseOptions

def run_once(opt, use_nms, tag):
    opt = copy.deepcopy(opt)
    opt.use_nms = use_nms
    opt.eval_split_name = opt.eval_split
    
    # We need eval_dataset. Let's just import and build it
    from training.moment_detr_gmr.dataset import StartEndDataset
    from training.moment_detr_gmr.evaluate import build_dataset_config
    
    data_path = f"data/label/Standard/{opt.eval_split}.jsonl"
    eval_dataset_config = build_dataset_config(opt, data_path, load_labels=True)
    eval_dataset = StartEndDataset(**eval_dataset_config)
    
    model, criterion, _, _ = setup_model(opt)
    ckpt = torch.load(opt.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(opt.device).eval()
    
    metrics, _, _ = eval_epoch(0, model, eval_dataset, opt, f"tmp_{tag}.jsonl", criterion)
    print(f"\n[{tag}] use_nms={use_nms}")
    if metrics:
        brief = metrics["brief"]
        for k in ["mAP", "mR@5", "mR+@5", "G-mIoU@1@0.6", "Rej-F1@0.6"]:
            val = brief.get(k)
            # if G-mIoU@1@0.6 not present, we can look at metrics['brief']['G-mIoU@1'] assuming threshold=0.6 was used
            # We can just extract from the full eval output since `eval_main.py` might be needed for guarding metrics
            print(f"  {k:<14} = {val}")
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--use_sa", default="False")
    ap.add_argument("--query_dropout", type=float, default=0.0)
    ap.add_argument("--nms_thr", type=float, default=0.7)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    opt_manager = BaseOptions("moment_detr", "soccer_gmr", "clip_slowfast")
    opt_manager.parse()
    opt = opt_manager.option
    
    opt.ckpt = args.ckpt
    opt.eval_split = args.eval_split
    opt.use_sa = (str(args.use_sa) == "True")
    opt.query_dropout = 0.0
    opt.nms_thr = args.nms_thr
    opt.device = args.device
    opt.t_feat_dir = "features/soccer_gmr/clip_text"
    opt.v_feat_dirs = ["features/soccer_gmr/clip", "features/soccer_gmr/slowfast"]
    opt.results_dir = "experiments/20260627_stage2_run1"

    m_on = run_once(opt, use_nms=True,  tag="NMS_ON")
    m_off = run_once(opt, use_nms=False, tag="NMS_OFF")
    
    os.system("python eval/eval_main.py --submission_path experiments/20260627_stage2_run1/tmp_NMS_ON.jsonl --gt_path data/label/Standard/test.jsonl --save_path tmp_eval_on.json --gmiou_cls_threshold 0.6")
    os.system("python eval/eval_main.py --submission_path experiments/20260627_stage2_run1/tmp_NMS_OFF.jsonl --gt_path data/label/Standard/test.jsonl --save_path tmp_eval_off.json --gmiou_cls_threshold 0.6")
    
    with open("tmp_eval_on.json") as f:
        metrics_on = json.load(f)
    with open("tmp_eval_off.json") as f:
        metrics_off = json.load(f)

    delta = {k: round(metrics_off.get(k, 0) - metrics_on.get(k, 0), 4)
             for k in ["mAP", "mR@5", "mR+@5", "G-mIoU@1", "Rej-F1@0.6"]}
    print("\n[判定] 关 NMS 后的变化(>0 表示 NMS 此前在压制该指标):")
    for k, v in delta.items():
        print(f"  Δ{k:<14} = {v:+}")

    verdict = ("元凶是 NMS 过度合并(轻症,调阈值即可)"
               if metrics_off.get("mR+@5", 0) - metrics_on.get("mR+@5", 0) > 0.3
               else "NMS 非主因 → 指向 query 堆叠(需第二刀确认)")
    print(f"\n[结论] {verdict}")

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    with open(os.path.join(os.path.dirname(args.ckpt), "reeval_no_nms.json"), "w") as f:
        json.dump({"nms_on": metrics_on, "nms_off": metrics_off, "delta": delta}, f,
                  ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
