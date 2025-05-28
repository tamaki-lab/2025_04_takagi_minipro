import csv
import json
import evaluate
import nltk
import os
import re
import hydra
from omegaconf import DictConfig

# nltk.download("wordnet")
# nltk.download("punkt")
# nltk.download("omw-1.4")

meteor = evaluate.load("meteor")


def extract_label_placeholders(text):
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return "", "", ""
        label = data.get("label", "").strip()
        placeholders = data.get("placeholders", [])
        ph_words = [re.sub(r"[^\w\s]", "", str(p)) for p in placeholders]
        ph_text = " ".join(ph_words).strip()
        combined = (label + " " + ph_text).strip()
        return label, ph_text, combined
    except BaseException:
        return "", "", ""


@hydra.main(config_path="../config", config_name="llama", version_base=None)
def main(cfg: DictConfig):
    csv_path = cfg.csv_path

    refs_label, preds_label = [], []
    refs_ph, preds_ph = [], []
    refs_combined, preds_combined = [], []

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        exit()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_l, pred_ph, pred_combined = extract_label_placeholders(row.get("result", ""))
            ref_l, ref_ph, ref_combined = extract_label_placeholders(row.get("ground_truth_json", ""))

            if pred_l and ref_l:
                preds_label.append(pred_l)
                refs_label.append(ref_l)
            if pred_ph and ref_ph:
                preds_ph.append(pred_ph)
                refs_ph.append(ref_ph)
            if pred_combined and ref_combined:
                preds_combined.append(pred_combined)
                refs_combined.append(ref_combined)

    print(f"Loaded {len(preds_combined)} predictions and {len(refs_combined)} references.")

    # 1. Label only
    results_label = meteor.compute(predictions=preds_label, references=refs_label)
    print("\n=== METEOR Score for Label only ===")
    print(f"Samples Evaluated: {len(preds_label)}")
    print(f"METEOR Score: {results_label['meteor']:.4f}")

    # 2. Placeholders only
    results_ph = meteor.compute(predictions=preds_ph, references=refs_ph)
    print("\n=== METEOR Score for Placeholders only ===")
    print(f"Samples Evaluated: {len(preds_ph)}")
    print(f"METEOR Score: {results_ph['meteor']:.4f}")

    # 3. Combined (Label + Placeholders)
    results_combined = meteor.compute(predictions=preds_combined, references=refs_combined)
    print("\n=== METEOR Score for Label + Placeholders Combined ===")
    print(f"Samples Evaluated: {len(preds_combined)}")
    print(f"METEOR Score: {results_combined['meteor']:.4f}")


if __name__ == "__main__":
    main()
