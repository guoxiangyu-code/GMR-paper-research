import json
import os

gt_path = 'data/label/Standard/test.jsonl'
out_path = 'data/label/Standard/test_gt2.jsonl'

count = 0
with open(gt_path, 'r') as f_in, open(out_path, 'w') as f_out:
    for line in f_in:
        data = json.loads(line.strip())
        if 'relevant_windows' in data and len(data['relevant_windows']) == 2:
            f_out.write(json.dumps(data) + '\n')
            count += 1

print(f"Saved {count} GT=2 queries to {out_path}.")
