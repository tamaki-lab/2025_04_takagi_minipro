import csv
import json
import evaluate
import nltk
import os
import re
import hydra
from omegaconf import DictConfig

# nltk のリソース（初回のみ）
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('omw-1.4')

# METEORのロード
meteor = evaluate.load("meteor")


def extract_label_and_placeholders(text):
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return ""
        label = data.get("label", "")
        placeholders = data.get("placeholders", [])
        ph_words = [re.sub(r"[^\w\s]", "", str(p)) for p in placeholders]
        return (label + " " + " ".join(ph_words)).strip()
    except BaseException:
        return ""


@hydra.main(config_path="../config", config_name="llama", version_base=None)
def main(cfg: DictConfig):
    csv_path = cfg.csv_path  # YAMLから取得

    references = []
    predictions = []

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        exit()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = extract_label_and_placeholders(row.get("result", ""))
            ref = extract_label_and_placeholders(row.get("ground_truth_json", ""))

            if pred and ref:
                predictions.append(pred)
                references.append(ref)

    print(f"Loaded {len(predictions)} predictions and {len(references)} references.")

    if not predictions:
        print("No valid data to evaluate.")
    else:
        results = meteor.compute(predictions=predictions, references=references)
        print("=== METEOR Evaluation ===")
        print(f"Samples Evaluated: {len(predictions)}")
        print(f"METEOR Score: {results['meteor']:.4f}")


if __name__ == "__main__":
    main()
