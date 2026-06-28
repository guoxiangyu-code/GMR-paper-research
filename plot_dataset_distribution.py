import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

def count_gt_distribution(file_path):
    counts = {0: 0, 1: 0, 2: 0, 3: 0, ">=4": 0}
    total = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            num_windows = len(data.get("relevant_windows", []))
            
            if num_windows >= 4:
                counts[">=4"] += 1
            else:
                counts[num_windows] += 1
            total += 1
    return counts, total

def main():
    train_file = "data/label/Standard/train.jsonl"
    test_file = "data/label/Standard/test.jsonl"
    
    train_counts, train_total = count_gt_distribution(train_file)
    test_counts, test_total = count_gt_distribution(test_file)
    
    print(f"Train total queries: {train_total}")
    print(f"Train distribution: {train_counts}")
    print(f"Test total queries: {test_total}")
    print(f"Test distribution: {test_counts}")
    
    labels = ['0', '1', '2', '3', '>=4']
    train_values = [train_counts[0], train_counts[1], train_counts[2], train_counts[3], train_counts[">=4"]]
    test_values = [test_counts[0], test_counts[1], test_counts[2], test_counts[3], test_counts[">=4"]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, train_values, width, label=f'Train (Total: {train_total})', color='#4C72B0')
    rects2 = ax.bar(x + width/2, test_values, width, label=f'Test (Total: {test_total})', color='#DD8452')
    
    ax.set_ylabel('Number of Queries')
    ax.set_xlabel('Number of Actions (GT Moments) per Query')
    ax.set_title('Distribution of Multiple Actions per Query in Soccer-GMR')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    os.makedirs("experiments/diag", exist_ok=True)
    out_path = "experiments/diag/dataset_distribution.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
