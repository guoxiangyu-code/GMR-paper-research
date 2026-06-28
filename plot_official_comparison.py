import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Baseline (No Rej)', 'Official (Moment-DETR-GMR)', 'Idea 1 (Double-Track)']
peak_gmiou = [4.49, 51.64, 51.76]
avg_gmiou = [4.49, 46.37, 37.09]
mr_plus = [0.97, 0.84, 1.12]

x = np.arange(len(labels))
width = 0.25

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Peak and Avg G-mIoU on primary y-axis
rects1 = ax1.bar(x - width, peak_gmiou, width, label='Peak G-mIoU@1', color='#3498db')
rects2 = ax1.bar(x, avg_gmiou, width, label='Avg G-mIoU@1 (across \u03c4)', color='#2980b9')

ax1.set_ylabel('G-mIoU@1 Score (%)', fontsize=12)
ax1.set_ylim(0, 60)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Create secondary y-axis for mR+@5 since it's much smaller (~1%)
ax2 = ax1.twinx()
rects3 = ax2.bar(x + width, mr_plus, width, label='mR+@5 (Multi-target)', color='#e74c3c')
ax2.set_ylabel('mR+@5 Score (%)', fontsize=12, color='#e74c3c')
ax2.set_ylim(0, 1.5)
ax2.tick_params(axis='y', labelcolor='#e74c3c')

# Add labels to bars
def autolabel(rects, ax, is_secondary=False):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color='#e74c3c' if is_secondary else 'black')

autolabel(rects1, ax1)
autolabel(rects2, ax1)
autolabel(rects3, ax2, is_secondary=True)

# Legends
lines, labels_ax1 = ax1.get_legend_handles_labels()
lines2, labels_ax2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels_ax1 + labels_ax2, loc='upper left', fontsize=11)

plt.title('Baseline vs Official vs Idea 1: Accuracy vs Multi-Target Recall', fontsize=14, pad=15)
plt.tight_layout()

# Save plot
plt.savefig("/home/guoxiangyu/Base_code/generalized-moment-retrieval/final_report/plot_official_vs_idea1.png", dpi=300)
print("Plot saved to plot_official_vs_idea1.png")
