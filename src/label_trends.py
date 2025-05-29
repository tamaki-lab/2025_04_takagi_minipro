import pandas as pd
import json
import re
import hydra
from omegaconf import DictConfig


def extract_label_word_count(json_text):
    try:
        data = json.loads(json_text)
        label = data.get("label", "")
        words = re.findall(r'\b\w+\b', label)
        return len(words)
    except Exception:
        return None


@hydra.main(config_path="../config", config_name="llama", version_base=None)
def main(cfg: DictConfig):
    csv_path = cfg.csv_path  # YAMLからパスを取得
    df = pd.read_csv(csv_path)

    # predのlabel語数
    df["label_word_count"] = df["result"].apply(extract_label_word_count)
    avg_label_word_count = df["label_word_count"].dropna().mean()

    # GTのlabel語数
    df["gt_label_word_count"] = df["ground_truth_json"].apply(extract_label_word_count)
    avg_gt_label_word_count = df["gt_label_word_count"].dropna().mean()

    print(f"Average word count in result 'label': {avg_label_word_count:.2f}")
    print(f"Average word count in ground truth 'label': {avg_gt_label_word_count:.2f}")

    # 結果の保存や可視化が必要ならここに追加


if __name__ == "__main__":
    main()
