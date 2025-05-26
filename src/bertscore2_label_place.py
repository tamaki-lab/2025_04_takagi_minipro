import pandas as pd
import json
import re
from bert_score import score
import hydra
from omegaconf import DictConfig


def extract_label_and_placeholders(text):
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
    df["pred_label"], df["pred_ph"] = zip(*df["result"].map(extract_label_and_placeholders))
    df["gt_label"], df["gt_ph"] = zip(*df["ground_truth_json"].map(extract_label_and_placeholders))

    # 有効データのみ抽出
    valid_rows = df[(df["pred_label"] != "") & (df["gt_label"] != "")]
    preds = (valid_rows["pred_label"] + " " + valid_rows["pred_ph"]).tolist()
    refs = (valid_rows["gt_label"] + " " + valid_rows["gt_ph"]).tolist()

    # BERTScore 計算
    P, R, F1 = score(preds, refs, lang="en", verbose=True)

    # 出力のみ
    print(f"Samples Evaluated: {len(F1)}")
    print(f"Average BERTScore F1: {F1.mean().item():.4f}")


if __name__ == "__main__":
    main()
