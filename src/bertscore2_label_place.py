import pandas as pd
import json
import re
from bert_score import score
import hydra
from omegaconf import DictConfig


def extract_label_placeholders(text):
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return "", ""
        label = data.get("label", "")
        placeholders = data.get("placeholders", [])
        ph_words = [re.sub(r"[^\w\s]", "", str(p)) for p in placeholders]
        return label, " ".join(ph_words)
    except BaseException:
        return "", ""


@hydra.main(config_path="../config", config_name="llama", version_base=None)
def main(cfg: DictConfig):
    csv_path = cfg.csv_path
    df = pd.read_csv(csv_path)

    # ラベルとプレースホルダを抽出
    df["pred_label"], df["pred_ph"] = zip(*df["result"].map(extract_label_placeholders))
    df["gt_label"], df["gt_ph"] = zip(*df["ground_truth_json"].map(extract_label_placeholders))

    # 有効データのみ抽出
    valid_rows = df[(df["pred_label"] != "") & (df["gt_label"] != "")]

    # 1. labelのみでBERTScore
    preds_label = valid_rows["pred_label"].tolist()
    refs_label = valid_rows["gt_label"].tolist()
    P_l, R_l, F1_l = score(preds_label, refs_label, lang="en", verbose=True)
    print("=== BERTScore for Label only ===")
    print(f"Samples Evaluated: {len(F1_l)}")
    print(f"Average BERTScore F1: {F1_l.mean().item():.4f}\n")

    # 2. placeholdersのみでBERTScore
    preds_ph = valid_rows["pred_ph"].tolist()
    refs_ph = valid_rows["gt_ph"].tolist()
    P_ph, R_ph, F1_ph = score(preds_ph, refs_ph, lang="en", verbose=True)
    print("=== BERTScore for Placeholders only ===")
    print(f"Samples Evaluated: {len(F1_ph)}")
    print(f"Average BERTScore F1: {F1_ph.mean().item():.4f}\n")

    # 3. label + placeholders 全体でBERTScore
    preds_combined = (valid_rows["pred_label"] + " " + valid_rows["pred_ph"]).tolist()
    refs_combined = (valid_rows["gt_label"] + " " + valid_rows["gt_ph"]).tolist()
    P_c, R_c, F1_c = score(preds_combined, refs_combined, lang="en", verbose=True)
    print("=== BERTScore for Label + Placeholders Combined ===")
    print(f"Samples Evaluated: {len(F1_c)}")
    print(f"Average BERTScore F1: {F1_c.mean().item():.4f}")


if __name__ == "__main__":
    main()
