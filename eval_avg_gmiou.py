import os
import json
import matplotlib.pyplot as plt

base_exp_dir = "/home/guoxiangyu/Base_code/generalized-moment-retrieval/experiments"
eval_script = "/home/guoxiangyu/GMR-Idea1-DualTrack/eval/eval_main.py"
gt_path = "/home/guoxiangyu/GMR-Idea1-DualTrack/data/label/Standard/test.jsonl"

# Known values from previous evaluations
results = {
    "Baseline": 4.49,
    "Official": 35.84,
}

exps = {
    "Idea_1": "/home/guoxiangyu/GMR-Idea1-DualTrack/results/moment_detr_gmr/moment_detr_gmr_test_submission.jsonl",
    "Probe_-SA": f"{base_exp_dir}/20260627_stage2_run2/results/moment_detr_gmr_test_submission.jsonl",
    "Div_0.1": f"{base_exp_dir}/20260627_stage3_coef0.1/results/moment_detr_gmr_test_submission.jsonl",
    "Div_0.3": f"{base_exp_dir}/20260627_stage3_run1_coef0.3/results/moment_detr_gmr_test_submission.jsonl",
    "Div_0.5": f"{base_exp_dir}/20260627_stage3_coef0.5/results/moment_detr_gmr_test_submission.jsonl",
    "Div_1.0": f"{base_exp_dir}/20260627_stage3_coef1.0/results/moment_detr_gmr_test_submission.jsonl",
}

for name, sub_path in exps.items():
    if not os.path.exists(sub_path):
        print(f"Skipping {name}, no submission file.")
        continue
    
    gmiou_sum = 0
    valid_count = 0
    
    for thresh in [0.4, 0.6, 0.8]:
        save_path = f"/tmp/eval_{name}_{thresh}.json"
        cmd = f"python {eval_script} --submission_path '{sub_path}' --gt_path '{gt_path}' --save_path '{save_path}' --gmiou_cls_threshold {thresh} > /dev/null 2>&1"
        os.system(cmd)
        
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                metrics = json.load(f)
                gmiou = metrics['brief'].get('G-mIoU@1', 0)
                gmiou_sum += gmiou
                valid_count += 1
                
    if valid_count == 3:
        avg_gmiou = gmiou_sum / 3.0
        # Remap name for display
        display_name = name.replace("_", " ")
        results[display_name] = avg_gmiou
        print(f"{display_name}: Avg G-mIoU@1 = {avg_gmiou:.2f}%")
    else:
        print(f"{name}: Failed to compute for all thresholds.")

# Write the data to a markdown table file
with open("/home/guoxiangyu/Base_code/generalized-moment-retrieval/final_report/avg_gmiou_data.md", "w") as f:
    f.write("| 实验配置 | Avg G-mIoU@1 (τ=0.4, 0.6, 0.8 平均) |\n")
    f.write("| :--- | :--- |\n")
    for name, score in results.items():
        f.write(f"| {name} | {score:.2f}% |\n")

# Plotting
names = list(results.keys())
scores = list(results.values())

plt.figure(figsize=(10, 6))
bars = plt.bar(names, scores, color=['#cccccc', '#f5c242', '#e74c3c', '#9b59b6', '#3498db', '#2980b9', '#1f618d', '#154360'])

# Highlight Idea 1
bars[names.index("Idea 1")].set_edgecolor('black')
bars[names.index("Idea 1")].set_linewidth(2)

plt.title('Average G-mIoU@1 across Strict Thresholds (τ=0.4, 0.6, 0.8)', fontsize=14, pad=15)
plt.ylabel('Avg G-mIoU@1 (%)', fontsize=12)
plt.xlabel('Experiments', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 60)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("/home/guoxiangyu/Base_code/generalized-moment-retrieval/final_report/plot_avg_gmiou.png", dpi=300)
print("Plot saved to plot_avg_gmiou.png")
