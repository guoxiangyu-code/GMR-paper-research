import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from easydict import EasyDict

import sys
sys.path.append("training/moment_detr_gmr")
from training.moment_detr_gmr.train import setup_model, build_dataset_config
from training.moment_detr_gmr.dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
import argparse

@torch.no_grad()
def diagnose_active_queries(model, loader, thr=0.05, device="cuda"):
    rows = []  # (n_gt_moment, n_active)
    model.eval()
    for batch in tqdm(loader, desc="Diagnosing active queries"):
        # batch is a tuple: (batch_meta, batched_data)
        batch_meta = batch[0]
        model_inputs, targets = prepare_batch_inputs(batch[1], device)
                
        out = model(**model_inputs)
        prob = out["slot_fg_prob"]          # [B,N]
        n_active = (prob > thr).sum(1).cpu().numpy()
        
        for b in range(prob.size(0)):
            # span_labels has GT spans
            n_gt = len(targets["span_labels"][b]["spans"]) if "span_labels" in targets else 0
            if n_gt > 0:                    # 阶段一只看正样本
                rows.append((n_gt, int(n_active[b])))
                
    rows = np.array(rows)
    # 按 GT moment 数分桶(1,2,3,4+),求平均 active 数 ± std
    print(f"\n--- Active Query Analysis (Threshold={thr}) ---")
    for g in [1, 2, 3]:
        m = rows[rows[:,0]==g]
        if len(m): print(f"GT={g}: active={m[:,1].mean():.2f}±{m[:,1].std():.2f} (count={len(m)})")
    m4 = rows[rows[:,0]>=4]
    if len(m4): print(f"GT>=4: active={m4[:,1].mean():.2f}±{m4[:,1].std():.2f} (count={len(m4)})")
    
    os.makedirs("experiments/diag", exist_ok=True)
    np.save("experiments/diag/active_vs_moment.npy", rows)
    print("Saved to experiments/diag/active_vs_moment.npy")

def parse_args():
    parser = argparse.ArgumentParser(description="Run diagnosis")
    parser.add_argument("--model", "-m", default="moment_detr", choices=["moment_detr"])
    parser.add_argument("--dataset", "-d", default="soccer_gmr", choices=["soccer_gmr"])
    parser.add_argument("--feature", "-f", default="clip_slowfast", choices=["clip_slowfast"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["val", "test"])
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--t_feat_dir", type=str, default=None)
    parser.add_argument("--v_feat_dirs", type=str, nargs="+", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--thr", type=float, default=0.05, help="active threshold")
    parser.add_argument("--use_sa", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--query_dropout", type=float, default=0.0)
    return parser.parse_args()

if __name__ == "__main__":
    from config import BaseOptions
    args = parse_args()
    option_manager = BaseOptions(args.model, args.dataset, args.feature, resume=None)
    option_manager.parse()
    opt = option_manager.option
    
    for name in ["eval_path", "t_feat_dir", "device", "use_sa", "query_dropout", "model_path"]:
        value = getattr(args, name)
        if value is not None:
            setattr(opt, name, value)
    if args.v_feat_dirs is not None:
        opt.v_feat_dirs = args.v_feat_dirs
        
    print("Setup config, data and model...")
    load_labels = True
    eval_dataset = StartEndDataset(**build_dataset_config(opt, opt.eval_path, load_labels=load_labels))
    
    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
    )
    
    model, _, _, _ = setup_model(opt)
    checkpoint = torch.load(opt.model_path, map_location="cpu", weights_only=False)
    
    ckpt_state = checkpoint.get("model")
    if ckpt_state is None:
        raise ValueError(f"Checkpoint missing key 'model': {opt.model_path}")
    model.load_state_dict(ckpt_state)
    model.to(opt.device)
    print(f"Loaded model from {opt.model_path}")
    
    diagnose_active_queries(model, eval_loader, thr=args.thr, device=opt.device)
