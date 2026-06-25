import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(gt_path, sub_path):
    with open(gt_path, 'r') as f:
        gts = [json.loads(line) for line in f]
    
    with open(sub_path, 'r') as f:
        subs = [json.loads(line) for line in f]
        
    sub_dict = {s['qid']: s['pred_exist_score'] for s in subs if 'pred_exist_score' in s}
    
    single, multi, null = [], [], []
    gt_single, gt_multi, gt_null = [], [], []
    
    for gt in gts:
        qid = gt['qid']
        if qid not in sub_dict: continue
        score = sub_dict[qid]
        
        windows = gt.get('relevant_windows', [])
        if not windows:
            null.append(score)
            gt_null.append(0.0)
        elif len(windows) == 1:
            single.append(score)
            gt_single.append(1.0)
        else:
            multi.append(score)
            gt_multi.append(1.0)
            
    return (single, multi, null), (gt_single, gt_multi, gt_null)

gt_path = 'data/label/Standard/test.jsonl'
off_sub = 'eval/example/example_test_submission.jsonl'
my_sub = 'results/moment_detr_gmr/test/moment_detr_gmr_test_submission.jsonl'

(off_single, off_multi, off_null), _ = load_data(gt_path, off_sub)
(my_single, my_multi, my_null), (gt_single, gt_multi, gt_null) = load_data(gt_path, my_sub)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

bins = np.linspace(0, 1, 50)

# Plot GT
axes[0].hist(gt_single, bins, alpha=0.5, label='Single (GT=1.0)')
axes[0].hist(gt_multi, bins, alpha=0.5, label='Multi (GT=1.0)')
axes[0].hist(gt_null, bins, alpha=0.5, label='Null-set (GT=0.0)')
axes[0].set_title('Ground Truth (GT) Exist Label')
axes[0].legend()

# Plot official
axes[1].hist(off_single, bins, alpha=0.5, label='Single')
axes[1].hist(off_multi, bins, alpha=0.5, label='Multi')
axes[1].hist(off_null, bins, alpha=0.5, label='Null-set')
axes[1].set_title('Official Submission pred_exist_score')
axes[1].legend()

# Plot reproduced
axes[2].hist(my_single, bins, alpha=0.5, label='Single')
axes[2].hist(my_multi, bins, alpha=0.5, label='Multi')
axes[2].hist(my_null, bins, alpha=0.5, label='Null-set')
axes[2].set_title('Reproduced Submission pred_exist_score')
axes[2].legend()

os.makedirs('results/moment_detr_gmr/error_analysis/figures', exist_ok=True)
plt.savefig('results/moment_detr_gmr/error_analysis/figures/distribution_compare.png')

print("Distribution plots saved to results/moment_detr_gmr/error_analysis/figures/distribution_compare.png")
