from bert_score import score
import csv
import json

csv_path = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/llama/batch_SSv2_result/folder_summary1-100.csv"

predictions = []
references = []

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pred = row.get("result", "").strip()
        gt_json = row.get("ground_truth_json", "").strip()
        if pred and gt_json:
            try:
                label = json.loads(gt_json).get("label", "").strip()
                if label:
                    predictions.append(pred)
                    references.append(label)
            except json.JSONDecodeError:
                continue

# BERTScoreの実行（model_type変更で日本語対応も可）
P, R, F1 = score(predictions, references, lang="en", verbose=True)

print(f"Avg BERTScore F1: {F1.mean().item():.4f}")
