import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from training.moment_detr_gmr.config import BaseOptions
from training.moment_detr_gmr.dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from models.moment_detr_gmr.moment_detr import build_model
from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx, temporal_iou

from easydict import EasyDict

def build_dataset_config(opt, data_path, load_labels):
    keep_empty_gt = bool(getattr(opt, "use_exist_head", False)) if load_labels else True
    return EasyDict(
        dset_name=opt.get("dset_name", "soccer_gmr"),
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
        max_a_l=getattr(opt, "max_a_l", 75),
        clip_len=getattr(opt, "clip_length", 2),
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=load_labels,
        mr_only=True,
        keep_empty_gt=keep_empty_gt,
    )

@torch.no_grad()
def dump_split_features(split_name, opt, model, save_path):
    # Determine split path
    if split_name == "train":
        data_path = "data/label/Standard/train.jsonl"
    elif split_name == "val":
        data_path = "data/label/Standard/val.jsonl"
    else:
        data_path = "data/label/Standard/test.jsonl"
        
    dataset = StartEndDataset(**build_dataset_config(opt, data_path, load_labels=True))
    loader = DataLoader(
        dataset, collate_fn=start_end_collate, batch_size=opt.eval_bsz,
        num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory
    )
    
    cache = []
    
    for batch in tqdm(loader, desc=f"dump {split_name}"):
        query_meta = batch[0]
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)
        outputs = model(**model_inputs)
        
        pred_spans = outputs["pred_spans"].cpu() # (bs, num_queries, 2) [cx, w]
        hs = outputs["hs"].cpu() # (bs, num_queries, d)
        last_xattn = outputs["last_xattn"].cpu() # (bs * num_heads, num_queries, L_vid+L_txt)
        txt_mem = outputs["txt_mem"].cpu() # (bs, L_txt, d)
        saliency_scores = outputs["saliency_scores"].cpu() # (bs, L_vid)
        
        prob = F.softmax(outputs["pred_logits"], -1)
        exist_scores = prob[..., 0].cpu() # (bs, num_queries)
        
        bs = pred_spans.shape[0]
        num_queries = pred_spans.shape[1]
        
        # reshape last_xattn to (bs, num_heads, num_queries, seq_len)
        num_heads = last_xattn.shape[0] // bs
        seq_len = last_xattn.shape[-1]
        last_xattn = last_xattn.view(bs, num_heads, num_queries, seq_len)
        # average across heads to get (bs, num_queries, seq_len)
        last_xattn = last_xattn.mean(dim=1)
        
        for b in range(bs):
            meta = query_meta[b]
            if "span_labels" in targets and b < len(targets["span_labels"]) and "spans" in targets["span_labels"][b]:
                gt_spans_cxw = targets["span_labels"][b]["spans"].cpu()
                gt_spans_xx = span_cxw_to_xx(gt_spans_cxw)
            else:
                gt_spans_xx = torch.empty((0, 2))
            gt_cnt = len(gt_spans_xx)
            
            # get sequence lengths
            L_vid = saliency_scores.shape[1]
            L_txt = txt_mem.shape[1]
            txt_mask = model_inputs["src_txt_mask"][b].bool().cpu() # 1 for valid
            
            for q in range(num_queries):
                cx, w = pred_spans[b, q].tolist()
                s, e = cx - w/2, cx + w/2
                
                # sal_sharp: peak contrast
                st_idx = max(0, int(s * L_vid))
                ed_idx = min(L_vid, int(e * L_vid))
                if ed_idx > st_idx:
                    sal_sharp = saliency_scores[b, st_idx:ed_idx].max() - saliency_scores[b].median()
                else:
                    sal_sharp = torch.tensor(0.0)
                
                # xattn_entropy
                xattn_q = last_xattn[b, q] # (seq_len)
                xattn_q = torch.clamp(xattn_q, 1e-9, 1.0)
                xattn_entropy = -(xattn_q * torch.log(xattn_q)).sum()
                
                # xmodal_align: token level max pool over txt
                # txt_mem is (L_txt, d), hs_q is (d)
                hs_q = hs[b, q]
                sim = F.cosine_similarity(hs_q.unsqueeze(0), txt_mem[b], dim=-1) # (L_txt)
                sim = sim.masked_fill(~txt_mask, -1.0)
                xmodal_align = sim.max()
                
                # maxIoU
                pred_xx = torch.tensor([[s, e]])
                if gt_cnt > 0:
                    iou, _ = temporal_iou(pred_xx, gt_spans_xx)
                    max_iou = iou.max().item()
                else:
                    max_iou = 0.0
                
                label = 1 if max_iou >= 0.5 else 0
                
                f = {
                    "hs": hs_q,
                    "xattn_entropy": xattn_entropy.item(),
                    "sal_sharp": sal_sharp.item(),
                    "width": w,
                    "xmodal_align": xmodal_align.item(),
                    "label": label,
                    "iou": max_iou,
                    "exist": exist_scores[b, q].item(),
                    "gt_cnt": gt_cnt,
                    "qid": meta.get("qid", None),
                    "s": s,
                    "e": e,
                    "duration": meta["duration"]
                }
                cache.append(f)
                
    torch.save(cache, save_path)
    print(f"Dumped {len(cache)} queries to {save_path}")

def main():
    base_opt = BaseOptions("moment_detr", "soccer_gmr", "clip_slowfast", None)
    base_opt.parse()
    opt = base_opt.opt
    opt.pin_memory = False
    opt.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, criterion = build_model(opt)
    checkpoint = torch.load("results/moment_detr_gmr/best.ckpt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(opt.device)
    model.eval()
    
    os.makedirs("results", exist_ok=True)
    
    # Dump test (val equivalent)
    dump_split_features("test", opt, model, "results/rerank_cache_test.pt")
    # Dump train
    dump_split_features("train", opt, model, "results/rerank_cache_train.pt")

if __name__ == "__main__":
    main()
