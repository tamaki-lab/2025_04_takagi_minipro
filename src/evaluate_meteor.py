import csv
import json
import evaluate
import nltk
import os

# # nltk リソースをダウンロード（初回のみ）
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('omw-1.4')

# METEORのロード
meteor = evaluate.load("meteor")

csv_path = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/llama/batch_SSv2_result/folder_summary1-100.csv"

references = []
predictions = []

if not os.path.exists(csv_path):
    print(f"CSV not found: {csv_path}")
    exit()

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pred = row.get("result", "").strip()
        gt_json = row.get("ground_truth_json", "").strip()

        if pred and gt_json:
            try:
                gt_dict = json.loads(gt_json)
                ref = gt_dict.get("label", "").strip()
                if ref:
                    references.append(ref)
                    predictions.append(pred)
            except json.JSONDecodeError:
                print(f"Invalid JSON in row: {row['folder']}")

print(f"Loaded {len(predictions)} predictions and {len(references)} references.")

if not predictions:
    print("No valid data to evaluate.")
else:
    results = meteor.compute(predictions=predictions, references=references)
    print("=== METEOR Evaluation ===")
    print(f"Samples Evaluated: {len(predictions)}")
    print(f"METEOR Score: {results['meteor']:.4f}")
