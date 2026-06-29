import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import random
import os

class RerankHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Input features: xattn_entropy (1), sal_sharp (1), width (1), xmodal_align (1)
        self.mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, xattn_entropy, sal_sharp, width, xmodal_align):
        x = torch.stack([xattn_entropy, sal_sharp, width, xmodal_align], dim=-1)
        return self.mlp(x).squeeze(-1)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)

def prepare_data(cache_path):
    cache = torch.load(cache_path)
    # group by qid
    grouped = defaultdict(list)
    for f in cache:
        qid = f["qid"]
        grouped[qid].append(f)
    return list(grouped.values())

def train_rerank(seed, train_groups, epochs=20):
    set_seed(seed)
    head = RerankHead().cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    margin = 0.5
    
    for epoch in range(epochs):
        head.train()
        total_loss = 0
        valid_samples = 0
        random.shuffle(train_groups)
        
        for group in train_groups:
            # group is a list of features for one query
            xattn_entropy = torch.tensor([f["xattn_entropy"] for f in group]).cuda()
            sal_sharp = torch.tensor([f["sal_sharp"] for f in group]).cuda()
            width = torch.tensor([f["width"] for f in group]).cuda()
            xmodal_align = torch.tensor([f["xmodal_align"] for f in group]).cuda()
            labels = torch.tensor([f["label"] for f in group]).cuda()
            
            P = (labels == 1).nonzero(as_tuple=True)[0]
            N = (labels == 0).nonzero(as_tuple=True)[0]
            
            if len(P) == 0 or len(N) == 0:
                continue
                
            xattn_p = xattn_entropy[P]
            sal_p = sal_sharp[P]
            w_p = width[P]
            align_p = xmodal_align[P]
            
            xattn_n = xattn_entropy[N]
            sal_n = sal_sharp[N]
            w_n = width[N]
            align_n = xmodal_align[N]
            
            scores_p = head(xattn_p, sal_p, w_p, align_p).view(-1, 1) # (num_P, 1)
            scores_n = head(xattn_n, sal_n, w_n, align_n).view(1, -1) # (1, num_N)
            
            # Broadcast to compute all pairs
            # margin ranking loss: max(0, margin - (scores_p - scores_n))
            loss = F.softplus(margin - (scores_p - scores_n)).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            valid_samples += 1
            
        print(f"Seed {seed} Epoch {epoch}: Loss {total_loss/max(1, valid_samples):.4f}")
        
    os.makedirs("results", exist_ok=True)
    torch.save(head.state_dict(), f"results/rerank_head_seed{seed}.pt")
    return head

if __name__ == "__main__":
    train_groups = prepare_data("results/rerank_cache_train.pt")
    for seed in [0, 1, 2]:
        train_rerank(seed, train_groups)
